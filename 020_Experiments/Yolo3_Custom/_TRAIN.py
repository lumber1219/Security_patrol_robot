# -*- coding: utf-8 -*-
# @Time    : 2018/8/7 下午5:10
# @Author  : ZM
# @Email   : zhangmeng001@pingan.com.cn
# @File    : _TRAIN.py
# @Software: Template
# @Descript: _TRAIN

import os
from Source import *
from Source._040_MODEL.Ctpn_model import Ctpn_model
from Source._010_DATA.Data import Data

if __name__ == "__main__":
    # data_path = "."
    # feature_path = "."
    # model_path = "."
    #
    # # 获取数据
    # data = Data.Data()
    # data.read_data_from_file(data_path)
    #
    # # 数据预处理
    # pre_process = Pre_process.Pre_process()
    # pre_process.process(data)
    #
    # # 提取特征
    # feature = Feature.Feature()
    # if not feature.is_existed(feature_path):
    #     feature.set_data(pre_process)
    #     feature.feature_engineer()
    #     feature.save_feature_to_path(feature_path,rewrite=True)
    # else:
    #     feature.load_feature_from_path(feature_path)
    #
    # # 模型建立,训练
    # model = Model.Model(save_path=model_path,save_best_only=True)
    # model.load()
    # model.train(feature)
    # model.model_visual()
    # model.result_visual()
    # model.score()

    BASE_PATH = '/home/huaili_jia/OCR_Lumber/datasets/VOCdevkit/VOC2007'


    Data = Data(BASE_PATH)
    model = Ctpn_model()
    model.load()
    model.train(Data)

