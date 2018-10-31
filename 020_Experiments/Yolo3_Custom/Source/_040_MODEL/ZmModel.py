# -*- coding: utf-8 -*-
# @Time    : 2018/10/31 AM:10:05
# @Author  : ZM
# @Email   : zhangmeng_cumt@163.com
# @File    : Model.py
# @Software: Template
# @Descript: Model


from AITemplate import B_Model
from interface import implements




class PaModel(implements(B_Model)):
    def __init__(self,save_path=None,save_best_only=True):
        self.save_path = save_path
        self.save_best_only = save_best_only




