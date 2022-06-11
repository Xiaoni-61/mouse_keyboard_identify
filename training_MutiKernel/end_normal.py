#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 10/5/2022 22:20
# @Author: xiaoni
# @File  : end_normal.py

'''
不使用核函数
计算accuracy 分成正负类进行计算 之前先平衡正负类的数量
'''
import random
import numpy as np
import os
import logging
import torch
from numpy import mean
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, label_binarize
import copy
from sklearn.model_selection import train_test_split
from MKLpy.preprocessing import rescale_01
from sklearn.metrics import accuracy_score, roc_auc_score


def takeLast(elem):
    return elem[-1]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG  # 设置日志输出格式
                        , filename="normal_SVM.log"  # log日志输出的文件位置和文件名
                        , filemode="w"  # 文件的写入格式，w为重新写入文件，默认是追加
                        ,
                        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                        # -8表示占位符，让输出左对齐，输出长度都为8位
                        , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                        )
    peopleName = ['chenyu', 'dongzhenyu', 'leishengchuan', 'liyajie', 'maoyu', 'niudanyang', 'renqiuchen',
                  'shenyanping', 'wangcheng', 'wanghe']
    pwdpath = os.getcwd()
    for t in range(2, 11):

        logging.debug("-----------------" + str(t * 30) + "s----------------------")
        logging.debug("----------------- C=1 ,gamma=5 ----------------------")
        path1 = '/data/mouse_data/' + str(t * 30) + 's_sum/' + str(t * 30) + 's_sum.txt'
        pathsum1 = pwdpath + path1
        print('preprocessing data...', end='')
        logging.info("preprocessing data...")
        data_mouse = np.loadtxt(pwdpath + path1)
        data_mouse1, data_mouse_label = np.split(data_mouse, indices_or_sections=(len(data_mouse[0]) - 1,),
                                                 axis=1)

        path2 = '/data/keyboard_data/' + str(t * 30) + 's_sum/' + str(t * 30) + 's_sum.txt'
        pathsum2 = pwdpath + path2
        print(pathsum2)
        try:
            data_keyboard = np.loadtxt(pwdpath + path2)
        except ValueError:
            continue
        data_keyboard1, data_keyboard_label = np.split(data_keyboard, indices_or_sections=(len(data_keyboard[0]) - 1,),
                                                       axis=1)
        print('done')
        logging.info("preprocessing data...done")
        # data_keyboard1 = data_keyboard1[:, :111]
        data1 = np.concatenate((data_mouse1, data_keyboard1), axis=1)

        Y1 = np.squeeze(data_mouse_label)
        Y1 = np.array([int(i) for i in Y1])

        j = 0
        for line in data1:
            if not (np.any(line)):  # 如果全是0
                data1 = np.delete(data1, j, axis=0)
                Y1 = np.delete(Y1, j, axis=0)
                j = j - 1
            j = j + 1

        '''
        '''
        accuracy_1 = []

        for pp in range(10):
            Y_backup = copy.deepcopy(Y1)
            Y_backup = torch.tensor(Y_backup)
            Y = Y_backup.unsqueeze(1)
            data = copy.deepcopy(data1)
            data_balance = np.concatenate((data, Y), axis=1)
            for i1 in range(len(data_balance)):
                if data_balance[i1][-1] == pp:
                    data_balance[i1][-1] = 1
                    continue
                else:
                    data_balance[i1][-1] = 0
            data_balance = sorted(data_balance, key=lambda x: (x[-1]), reverse=True)  # 逆序从1开始排序
            for i2 in range(len(data_balance)):
                if data_balance[i2][-1] == 1:
                    continue
                else:
                    break
            data_balance = np.array(data_balance)
            Positive_sample, Negative_sample = np.split(data_balance, indices_or_sections=(i2,), axis=0)
            Negative_sample = random.sample(list(Negative_sample), i2)
            Negative_sample = np.array(Negative_sample)
            # Negative_sample = np.random.choice(Negative_sample, size=i2, replace=False)

            data_processed = np.concatenate((Positive_sample, Negative_sample), axis=0)
            data, Y = np.split(data_processed, indices_or_sections=(len(data_processed[0]) - 1,), axis=1)

            Y = np.squeeze(Y)
            Y = np.array([int(i) for i in Y])

            data = rescale_01(data)

            Xtr, Xte, Ytr, Yte = train_test_split(data, Y, test_size=0.15, random_state=60)
            Ytr_backup = copy.deepcopy(Ytr)
            Yte_backup = copy.deepcopy(Yte)

            Ytr_person_backup = copy.deepcopy(Ytr)
            Yte_person_backup = copy.deepcopy(Yte)

            base_learner = svm.SVC(C=1, cache_size=2000, decision_function_shape='ovr')
            # compute homogeneous polynomial kernels with degrees 0,1,2,...,10.
            base_learner.fit(Xtr, Ytr.ravel())
            accuracy = base_learner.score(Xte, Yte)
            accuracy_1.append(accuracy)
            print(str(t * 30) + " "+(str)(pp) + " accuracy: " + (str)(accuracy))
        accuracy_1_mean = mean(accuracy_1)
        logging.info(str(t * 30) + " mean accuracy:" + str(accuracy_1_mean))
