import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from timeit import default_timer as timer

from pycuda.compiler import SourceModule
mod = SourceModule("""
__global__ void func(float *a, float *b, size_t N)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
  {
    return;
  }
  float temp_a = a[i];
  float temp_b = b[i];
  a[i] = (temp_a * 10 + 2 ) * ((temp_b + 2) * 10 - 5 ) * 5;
  // a[i] = a[i] + b[i];
}
""")

func = mod.get_function("func")

def test(N):
    # N = 1024 * 1024 * 90   # float: 4M = 1024 * 1024

    print("N = %d" % N)

    N = np.int32(N)

    a = np.random.randn(N).astype(np.float32)
    b = np.random.randn(N).astype(np.float32)
    # copy a to aa
    aa = np.empty_like(a)
    aa[:] = a
    # GPU run
    nTheads = 256
    nBlocks = int( ( N + nTheads - 1 ) / nTheads )
    start = timer()
    func(
            drv.InOut(a), drv.In(b), N,
            block=( nTheads, 1, 1 ), grid=( nBlocks, 1 ) )
    run_time = timer() - start
    print("gpu run time %f seconds " % run_time)
    # cpu run
    start = timer()
    aa = (aa * 10 + 2 ) * ((b + 2) * 10 - 5 ) * 5
    run_time = timer() - start

    print("cpu run time %f seconds " % run_time)

    # check result
    r = a - aa
    print( min(r), max(r) )

def main():
  for n in range(1, 10):
    N = 1024 * 1024 * (n * 10)
    print("------------%d---------------" % n)
    test(N)



def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep




if __name__ == '__main__':
    # main()
    # import pycuda.gpuarray as gpuarray
    # import pycuda.driver as cuda
    # import pycuda.autoinit
    # import numpy
    #
    # a_gpu = gpuarray.to_gpu(numpy.random.randn(4, 4).astype(numpy.float32))
    # a_doubled = (2 * a_gpu).get()
    # print a_doubled
    # print a_gpu

    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy

    from pycuda.compiler import SourceModule

    mod = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x;
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = mod.get_function("multiply_them")

    a = numpy.random.randn(400).astype(numpy.float32)
    b = numpy.random.randn(400).astype(numpy.float32)

    dest = numpy.zeros_like(a)
    multiply_them(
        drv.Out(dest), drv.In(a), drv.In(b),
        block=(400, 1, 1), grid=(1, 1))

    print dest - a * b




# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

# ------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# ------------------------------------------------------------------

from neon.backends.util.source_module import SourceModule
from pycuda.tools import context_dependent_memoize


@context_dependent_memoize
def _get_nms_kernel():

    code = r"""
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
int const threadsPerBlock = sizeof(unsigned int) * 8;
__device__ inline float devIoU(float const * const a, float const * const b,
                               int const offset) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + offset, 0.f), height = max(bottom - top + offset, 0.f);
  float interS = width * height;
  float Sa = (a[2] - a[0] + offset) * (a[3] - a[1] + offset);
  float Sb = (b[2] - b[0] + offset) * (b[3] - b[1] + offset);
  return interS / (Sa + Sb - interS);
}
__global__ void nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned int *dev_mask,
                           const bool normalized) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;
  // if (row_start > col_start) return;
  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
  // if boxes are not normalized to image dim, we use an offset of 1 in
  // calculating the box width and height.
  const float offset = normalized ? 0 : 1;
  __shared__ float block_boxes[threadsPerBlock * 5];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();
  if (threadIdx.x < row_size) {
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 5;
    int i = 0;
    unsigned int t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_boxes + i * 5, offset) > nms_overlap_thresh) {
        t |= 1UL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}
"""

    module = SourceModule(code)
    kernel = module.get_function("nms_kernel")
    sig = "1I 1f 2P 1b"
    kernel.prepare(sig)
    return kernel
