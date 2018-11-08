# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 PM:22:00
# @Author  : ZM
# @Email   : zhangmeng_cumt@163.com
# @File    : predict_util.py
# @Software: Template
# @Descript: Model


import colorsys
import numpy as np


def generate_colors(color_numbers):
    '''Generate colors for drawing bounding boxes.
    :return:
    '''
    hsv_tuples = [(x / float(color_numbers), 1., 1.)
                  for x in range(color_numbers)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),colors))
    np.random.seed(10101)       # Fixed seed for consistent colors across runs. np.random.seed(number) same number, same np.random.shuffle()
    np.random.shuffle(colors)   # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)        # Reset seed to default.

    return colors
