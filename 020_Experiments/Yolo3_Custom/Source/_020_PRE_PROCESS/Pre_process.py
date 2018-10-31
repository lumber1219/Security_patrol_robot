# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 下午7:17
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : Pre_process.py
# @Software: Template
# @Descript: Pre_process

from AITemplate import B_Pre
from interface import implements

class Pre_process(implements(B_Pre)):
    def __init__(self):
        self._data
    def process(self,data):
        pass
    def get_data(self):
        return self._data

if __name__ == "__main__":
    pass