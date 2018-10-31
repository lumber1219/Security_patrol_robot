# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 下午5:37
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : B_Model.py
# @Software: Template
# @Descript: B_Model

from interface import Interface

class B_Model(Interface):
    def __init__(self, save_path=None, save_best_only=True):
        self.save_path = save_path
        self.save_best_only = save_best_only

    def load(self, weights_path=None, model_path=None):
        pass

    def train(self, feature):
        pass

    def predict(self):
        pass

    def score(self):
        pass

    def model_visual(self):
        pass

    def result_visual(self):
        pass

    def create_model(self, weights_path=None, model_path=None):
        pass


if __name__ == "__main__":
    pass