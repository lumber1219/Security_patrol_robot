# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 下午7:07
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : Feature.py
# @Software: Template
# @Descript: Feature

from AITemplate import B_Feature
from interface import implements

class Feature(implements(B_Feature)):
    def __init__(self):
        pass

    def set_data(self, data):
        pass

    def is_existed(self,path):
        return False

    def feature_engineer(self):
        pass

    def load_feature_from_path(self, path):
        pass

    def get_feature(self):
        pass

    def save_feature_to_path(self, path, rewrite=False):
        pass


if __name__ == "__main__":
    pass