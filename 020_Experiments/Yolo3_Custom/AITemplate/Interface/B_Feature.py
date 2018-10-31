# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 下午5:27
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : B_Feature.py
# @Software: Template
# @Descript: B_Feature

from interface import Interface

class B_Feature(Interface):
    def __init__(self):
        pass

    def set_data(self,data):
        pass

    def is_existed(self, path):
        pass

    def feature_engineer(self):
        pass

    def load_feature_from_path(self, path):
        pass

    def get_feature(self):
        pass

    def save_feature_to_path(self,path,rewrite=False):
        pass

if __name__ == "__main__":
    pass