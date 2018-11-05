# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 下午5:10
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : _RUN.py
# @Software: Template
# @Descript: _RUN

from Source import *
from PIL import Image
from Source._040_MODEL import ZmModel
import time
import cv2
import numpy as np
if __name__ == "__main__":
    pass
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
    # model.model_visual()
    # model.predict()
    #
    # # 后处理
    # post_process = Post_process.Post_process()
    # post_process.process(feature)
    # result = post_process.get_data()

    base_path = ''

    # 模型建立
    model = ZmModel.PaModel()
    model.load()

    # img = input('Input image filename:')
    #
    # image = Image.open(img)
    # start_time = time.time()
    # out_boxes, out_scores, out_classes = model.predict(image)
    # img_new = model.result_visual(image, out_boxes, out_scores, out_classes)
    # print('Need time is', time.time()-start_time)
    # img_new.show()
    # model.close_session()

    # while True:
    #     while True:
    #         img = input('Input image filename:')
    #         try:
    #             image = Image.open(img)
    #         except:
    #             print('Open Error! Try again!')
    #             continue
    #         else:
    #             r_image = yolo.detect_image(image)
    #             r_image.show()
    #     yolo.close_session()
    #
    # while True:
    #     pass

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = Image.fromarray(frame)
        out_boxes, out_scores, out_classes = model.predict(frame)
        img_new = model.result_visual(frame, out_boxes, out_scores, out_classes)
        # img_new.show()
        cv2.imshow('1',np.array(img_new))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            model.close_session()
            break


