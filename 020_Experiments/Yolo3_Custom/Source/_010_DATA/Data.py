# -*- coding: utf-8 -*-
# @Time    : 2018/7/19 下午5:50
# @Author  : ZZZ
# @Email   : zuxinxing531@pingan.com.cn
# @File    : data.py
# @Software: Template
# @Descript: data

# from AITemplate import B_Data
# from interface import implements
import os
import random
import cv2
import numpy as np
from Queue import Queue
from glob import glob
from threading import Thread
import xmltodict
from AITemplate import B_Data

# random.shuffle

class Data():
    def __init__(self,data_path):
        '''数据类初始化
        '''
        self.imgs = []
        self.gtboxes = []
        self.filenames = []

        self.data_path = data_path

        self.batch_size = 1
        self.thread_num = 2

        self.record_point = 0


        # record and image_label queue
        self.record_queue = Queue(maxsize=10000)
        self.image_label_queue = Queue(maxsize=512)

        self.samples_paths_all = glob(self.data_path + os.sep +'JPEGImages' +os.sep +'**')
        self.record_number = len(self.samples_paths_all)

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)  # 需要多少次

        t_record_producer = Thread(target= self.sample_path_record)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()


    def sample_path_record(self):
        while True:
            if self.record_point % self.record_number == 0:
                self.record_point = 0
                random.shuffle(self.samples_paths_all)

            self.record_queue.put(self.samples_paths_all[self.record_point])
            self.record_point += 1

    def record_customer(self):
        while True:
            item_image = self.record_queue.get()
            item_label = self.data_path + os.sep + 'Annotations' + os.sep + os.path.splitext(os.path.split(item_image)[1])[0] + '.xml'
            out = self.read_data_from_file(item_image,item_label)
            self.image_label_queue.put(out)

    def batch(self):
        img_batch = []
        gt_boxes_batch = []
        objects_num = []
        for i in range(self.batch_size):
            img, gt_boxes = self.image_label_queue.get()
            img_batch.append(img)
            gt_boxes_batch.append(gt_boxes)
        img_batch = np.asarray(img_batch, dtype=np.float32)
        img_batch = img_batch / 255 * 2 - 1
        # print(gt_boxes_batch)
        gt_boxes_batch = np.asarray(gt_boxes_batch, dtype=np.float32)
        return img_batch, gt_boxes_batch



    def read_data_from_file(self, IMAGE_PATH,LABEL_PATH,data_size=-1,format="default", *args):
        """从文件读取数据并保存到self._data中

        Args:
            data_path:训练数据路径
            format:读取后返回的数据格式(由开发者自定义)
            *args:

        Returns:
            self.imgs:训练数据
            self.rgns:标签数据
        """
        gtbox,_ = self.readxml(LABEL_PATH)
        img = cv2.imread(IMAGE_PATH)

        return [img, gtbox]


    def readxml(self,path):
        gtboxes = []
        imgfile = ''
        with open(path, 'rb') as f:
            xml = xmltodict.parse(f)
            bboxes = xml['annotation']['object']
            if (type(bboxes) != list):
                x1 = bboxes['bndbox']['xmin']
                y1 = bboxes['bndbox']['ymin']
                x2 = bboxes['bndbox']['xmax']
                y2 = bboxes['bndbox']['ymax']
                gtboxes.append((int(x1), int(y1), int(x2), int(y2)))
            else:
                for i in bboxes:
                    x1 = i['bndbox']['xmin']
                    y1 = i['bndbox']['ymin']
                    x2 = i['bndbox']['xmax']
                    y2 = i['bndbox']['ymax']
                    gtboxes.append((int(x1), int(y1), int(x2), int(y2)))

            imgfile = xml['annotation']['filename']
        return np.array(gtboxes), imgfile


    def get_data(self, *args):
        """
        获取数据并返回
        Args:
            *args:

        Returns:
            从文件读取到的数据
        """
        return (self.imgs,self.gtboxes,self.filenames)

    def visual(self,start_index,num,rand=True):
        """
        对数据和标签进行可视化
        Args:
            start_index:开始序号
            num:可视化样本数量
            rand:序号是否随机挑选

        Returns:
        """
        for i in range(num):
            pos_i = start_index+i
            if rand == True:
                pos_i = random.randint(0,99999)%len(self.gtboxes)
            boxes = self.gtboxes[pos_i]
            img = self.imgs[pos_i]
            for i in range(boxes.shape[0]):
                # print i, ":", boxes[i]
                cv2.rectangle(img, (boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), (255, 0, 0), thickness=1)
            cv2.imshow(self.filenames[pos_i],img)
        cv2.waitKey(0)

if __name__ == "__main__":
    pass
