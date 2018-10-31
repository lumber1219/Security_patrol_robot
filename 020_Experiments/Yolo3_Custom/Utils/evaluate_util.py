# -*- coding: utf-8 -*-

import numpy as np

# ground truth object rectangles Gi, i = 1..|G|
# detected object rectangles Dj, j = 1..|D|

def DetEval(gt_boxes, predict_boxes, tr=0.8, tp=0.4):   # 为何tr比tp大这么多，检测大的框总比检测框不够大的好！论文中设置的是这个
    '''参考：Object Count / Area Graphs for the Evaluation of Object Detection and Segmentation Algorithms
    @gt_boxes:
    @predict_boxes:
    @tr: is the constraint on area recall , belong [0,1]
    @tp: is the constraint on area precision, belong [0,1]
    :return:
    @recall
    @precision
    '''
    # tr = 0.6
    # tp = 0.8

    recall_matrix, precision_matrix = recall_precision_matrix(gt_boxes, predict_boxes)
    match_G, match_D = cal_match_G_D(recall_matrix, precision_matrix, tr, tp)

    recall = np.sum(match_G) / gt_boxes.shape[0]
    precision = np.sum(match_D) / predict_boxes.shape[0]

    return recall, precision

def cal_match_G_D(recall_matrix, precision_matrix, tr, tp):
    '''
    :param recall_matrix:
    :param precision_matrix:
    :param tr:
    :param tp:
    :return:
    '''
    '''
    match_G:
    if G_i match one D：即recall、precision中同一行只有一个元素大于tr, tp
        1
    elseif G_i match several D: precision的第i行有多个元素大于tp，且对应recall中多个元素的和大于tr
        f_sc(k)
    elseif G_i match zero D:
        0
        
    match_D:
    if D_j match one G：
        1
    elseif D_i match several G:即recall、precision中同一列只有一个元素大于tr, tp
        f_sc(k)
    elseif D_i match zero G:
        0  
    '''
    match_G = np.zeros([recall_matrix.shape[0]])
    match_D = np.zeros([precision_matrix.shape[1]])
    for i in range(recall_matrix.shape[0]):
        inds_over_tp_per_row = np.where(precision_matrix[i, :] > tp)
        if len(inds_over_tp_per_row) == 0 or np.sum(recall_matrix[i,inds_over_tp_per_row]) < tr:
            match_G[i] = 0
        else:
            if np.shape(inds_over_tp_per_row)[1] == 1:
                match_G[i] = 1
            else:
                match_G[i] = 0.7

    for j in range(precision_matrix.shape[1]):
        inds_over_tr_per_column = np.where(recall_matrix[:, j] > tr)
        if len(inds_over_tr_per_column) == 0 or np.sum(precision_matrix[inds_over_tr_per_column, j] < tp):
            match_D[j] = 0
        else:
            if np.shape(inds_over_tr_per_column)[1] ==1:
                match_D[j] = 1
            else:
                match_D[j] = 1

    return match_G, match_D



def recall_precision_matrix(gt_boxes, predict_boxes):
    '''
    :param gt_boxes: m * 4,  xmin1,ymin1,xmax1,ymax1
    :param predict_boxes: n*4,   xmin2,ymin2,xmax2,ymax2
    :return:
    :param recall_matrix: [gt_boxes.shape[0], predict_boxes.shape[0]]
    '''
    area_gt_boxes = (gt_boxes[:,2] - gt_boxes[:,0]) * (gt_boxes[:,3] - gt_boxes[:,1])
    area_predict_boxes = (predict_boxes[:, 2] - predict_boxes[:,0]) * (predict_boxes[:,3] - predict_boxes[:,1])

    recall_matrix = np.zeros([gt_boxes.shape[0], predict_boxes.shape[0]])
    precision_matrix = np.zeros([gt_boxes.shape[0], predict_boxes.shape[0]])
    intersection_area_matrix = np.zeros([gt_boxes.shape[0], predict_boxes.shape[0]])
    for i in range(gt_boxes.shape[0]):
        intersection_area_matrix[i,:] = intersection_area(gt_boxes[i,:], predict_boxes)

    for i in range(gt_boxes.shape[0]):
        recall_matrix[i, :] = intersection_area_matrix[i, :] / area_gt_boxes[i]

    for i in range(predict_boxes.shape[0]):
        precision_matrix[:, i] = intersection_area_matrix[:,i] / area_predict_boxes[i]

    return recall_matrix, precision_matrix


def intersection_area(gt_box, predict_boxes):
    '''计算recall matrix, precision matrix的每一行, 矩阵大小[|G|, |D|]
    gt_box:         xmin1,ymin1,xmax1,ymax1
    predict_boxes:    xmin2,ymin2,xmax2,ymax2
    area_gt_box:    gt_box的面积
    area_predict_box:
    :return: 交/gt_box面积
    '''
    x1 = np.maximum(gt_box[0], predict_boxes[:,0])
    x2 = np.minimum(gt_box[2], predict_boxes[:,2])
    y1 = np.maximum(gt_box[1], predict_boxes[:,1])
    y2 = np.minimum(gt_box[3], predict_boxes[:,3])
    intersection_area = np.maximum((x2 - x1), 0) * np.maximum((y2-y1), 0)
    # recall_matrix_row = intersection_area / area_gt_box
    # precision_matrix_row = intersection_area  / area_predict_box

    return intersection_area


def cal_iou(box1,box1_area, boxes2,boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0],boxes2[:,0])
    x2 = np.minimum(box1[2],boxes2[:,2])
    y1 = np.maximum(box1[1],boxes2[:,1])
    y2 = np.minimum(box1[3],boxes2[:,3])

    intersection = np.maximum(x2-x1,0) * np.maximum(y2-y1,0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1,boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box

    """
    area1 = (boxes1[:,0] - boxes1[:,2]) * (boxes1[:,1] - boxes1[:,3])
    area2 = (boxes2[:,0] - boxes2[:,2]) * (boxes2[:,1] - boxes2[:,3])

    overlaps = np.zeros((boxes1.shape[0],boxes2.shape[0]))

    #calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i],area1[i],boxes2,area2)

    return overlaps





if __name__ == '__main__':
    import cv2

    gt_boxes = np.array([
        [30,30,70,70],
        [160,10,200,70]
    ])

    predict_boxes = np.array([
        [25,31,65,78],
        [20,20,80,40],
        [30,32,70,72]
    ])

    recall, precision =  DetEval(gt_boxes, predict_boxes)
    print ('recall:{}, precision:{}'.format(recall, precision))
    img = np.zeros((512, 512, 3), np.uint8)
    # fill the image with white
    img.fill(255)
    gt_boxes = gt_boxes.astype(int)
    predict_boxes = predict_boxes.astype(int)
    for i in gt_boxes:
        print(i)
        cv2.rectangle(img,(i[0],i[1]), (i[2], i[3]),(25, 25, 112),1)
    for i in predict_boxes:
        cv2.rectangle(img, (i[0],i[1]), (i[2], i[3]),(255, 0, 0),1)
        print(i)
    cv2.imshow('image', img)
    cv2.waitKey(0)














