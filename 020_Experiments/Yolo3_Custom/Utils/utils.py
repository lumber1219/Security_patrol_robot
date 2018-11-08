# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 PM:23:00
# @Author  : ZM
# @Email   : zhangmeng_cumt@163.com
# @File    : utils.py
# @Software: Template
# @Descript: utils

from functools import reduce
from PIL import Image

def compose(*funcs):
    ''' Compose arbitrarily many functions, evaluated left to right.
        Reference: https://mathieularose.com/function-composition-in-python/
    :param funcs: functions, (conv, BN, Relu), Relu(BN(conv(x)))
    :return:
    '''
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    w= float(w)
    h= float(h)
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, (int((w-nw)//2), int((h-nh)//2)))
    return new_image
