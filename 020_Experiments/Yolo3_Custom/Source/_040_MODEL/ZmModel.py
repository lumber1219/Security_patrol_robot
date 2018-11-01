# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 AM:10:05
# @Author  : ZM
# @Email   : zhangmeng_cumt@163.com
# @File    : Model.py
# @Software: Template
# @Descript: Model


from AITemplate import B_Model
from interface import implements
from Utils import predict_utils
import colorsys
import os
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from Source._040_MODEL.Model_utils import yolo_eval, yolo_body
# from Source._040_MODEL.Model_utils import yolo_eval, yolo_body, tiny_yolo_body
from Utils.utils import letterbox_image
import os
from keras.utils import multi_gpu_model




# class PaModel(implements(B_Model)):
class PaModel():
    # 程序外部参数的定义
    _defaults = {
        "model_path": '/home/lumber/Graduation_Project/Detect/Security_patrol_robot/020_Experiments/Yolo3_Custom/model_data/yolo.h5',
        "anchors_path": '/home/lumber/Graduation_Project/Detect/Security_patrol_robot/020_Experiments/Yolo3_Custom/model_data/yolo_anchors.txt',
        "classes_path": '/home/lumber/Graduation_Project/Detect/Security_patrol_robot/020_Experiments/Yolo3_Custom/model_data/coco_classes.txt',
        "score": 0.3,
        "iou": 0.45,
        "model_image_size": (416, 416),
        "gpu_num": 1,
    }

    def __init__(self,save_path=None,save_best_only=True):
        #-------------------------------------------------
        #paras defination
        #-------------------------------------------------
        self.save_path = save_path
        self.save_best_only = save_best_only
        self.model_path = self._defaults['model_path']
        self.anchors_path = self._defaults['anchors_path']
        self.classes_path = self._defaults['classes_path']
        self.score = self._defaults['score']
        self.iou = self._defaults['iou']
        self.model_image_size = self._defaults['model_image_size']
        self.gpu_num = self._defaults['gpu_num']
        # self.mode_image_size
        #-------------------------------------------------
        #
        #-------------------------------------------------
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()


    def _get_class(self):
        ''' return all the class names
        :return:
        '''
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        '''return anchors because of k-means
        :return:
        '''
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            print('Need Yolo_tiny')
            # self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
            #     if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            # self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        self.colors = predict_utils.generate_colors(len(self.class_names))


        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes



    def detect_image(self, image):
        start = timer()

        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='/home/lumber/Graduation_Project/Detect/Security_patrol_robot/020_Experiments/Yolo3_Custom/font/FiraMono-Medium.otf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right, bottom))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image


if __name__ == "__main__":
    yolo = PaModel()
    while True:
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image)
                r_image.show()
        yolo.close_session()
