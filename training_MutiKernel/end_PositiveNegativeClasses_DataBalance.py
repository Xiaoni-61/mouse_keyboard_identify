#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 10/5/2022 上午 10:23
# @Author: xiaoni
# @File  : end_PositiveNegativeClasses_DataBalance.py

'''
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


def takeLast(elem):
    return elem[-1]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG  # 设置日志输出格式
                        , filename="PositiveNegativeClasses_DataBalance_accuracy.log"  # log日志输出的文件位置和文件名
                        # , filemode="w"  # 文件的写入格式，w为重新写入文件，默认是追加
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
        accuracy_2 = []
        accuracy_3 = []
        accuracy_4 = []
        accuracy_5 = []
        accuracy_6 = []
        accuracy_7 = []
        accuracy_8 = []
        accuracy_9 = []

        for pp in range(10):
            Y_backup = copy.deepcopy(Y1)
            Y_backup = torch.tensor(Y_backup)
            Y = Y_backup.unsqueeze(1)
            data= copy.deepcopy(data1)
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
            Negative_sample=np.array(Negative_sample)
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

            data_mouse_tr, data_keyboard_tr = np.split(Xtr, indices_or_sections=(43,), axis=1)
            data_mouse_te, data_keyboard_te = np.split(Xte, indices_or_sections=(43,), axis=1)
            base_learner = svm.SVC(C=1, cache_size=2000)
            # compute homogeneous polynomial kernels with degrees 0,1,2,...,10.



            '''
            mouse:linear_kernel keyboard:linear_kernel
            '''
            logging.info("computing linear_kernel Kernels...")
            print('computing linear_kernel Kernels...', end='')
            from MKLpy.metrics import pairwise

            Yte = copy.deepcopy(Yte_backup)
            Ytr = copy.deepcopy(Ytr_backup)


            KLtr = []
            KLte = []

            KLtr.append(pairwise.linear_kernel(data_mouse_tr))
            KLte.append(pairwise.linear_kernel(data_mouse_te, data_mouse_tr))
            KLtr.append(pairwise.linear_kernel(data_keyboard_tr))
            KLte.append(pairwise.linear_kernel(data_keyboard_te, data_keyboard_tr))

            # KLtr1 = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(3, 5)]
            # KLte1 = [pairwise.homogeneous_polynomial_kernel(Xte, Xtr, degree=d) for d in range(3, 5)]
            # KLtr1 = [pairwise.linear_kernel(Xtr)]
            print('done')

            # MKL algorithms
            from MKLpy.algorithms import AverageMKL, EasyMKL, KOMD

            print('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            # K_average = clf.solution.ker_matrix  # the combined kernel matrix

            score1 = clf.score(KLte, Yte)
            logging.info('AverageMKL ' + str(t * 30) + 's score:' + str(score1))
            print('AverageMKL ' + str(t * 30) + ' score:' + str(score1))

            # evaluate the solution
            from sklearn.metrics import accuracy_score, roc_auc_score

            y_pred = clf.predict(KLte)  # predictions
            y_score = clf.decision_function(KLte)  # rank
            accuracy = accuracy_score(Yte, y_pred)
            accuracy_1.append(accuracy)
            # Y_pred_prob = clf.predict_proba(KLte)

            logging.info('mouse:linear_kernel keyboard:linear_kernel')
            logging.info('Accuracy score:' + str(accuracy))
            print('mouse:linear_kernel keyboard:linear_kernel Accuracy score: %.3f' % (accuracy))

            '''
            mouse:rbf_kernel keyboard:rbf_kernel
            '''

            logging.info("computing rbf_kernel Kernels...")
            print('computing rbf_kernel Kernels...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.rbf_kernel(data_mouse_tr, gamma=2))
            KLte.append(pairwise.rbf_kernel(data_mouse_te, data_mouse_tr, gamma=2))
            KLtr.append(pairwise.rbf_kernel(data_keyboard_tr, gamma=2))
            KLte.append(pairwise.rbf_kernel(data_keyboard_te, data_keyboard_tr, gamma=2))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')

            from sklearn.metrics import accuracy_score, roc_auc_score

            y_pred = clf.predict(KLte)  # predictions
            accuracy = accuracy_score(Yte, y_pred)
            accuracy_2.append(accuracy)

            logging.info('mouse:rbf_kernel keyboard:rbf_kernel')
            logging.info('Accuracy score:' + str(accuracy))

            print('mouse:rbf_kernel keyboard:rbf_kernel Accuracy score: %.3f' % (accuracy))

            '''
                    polynomial_kernel
            '''
            Yte = Yte_backup
            Ytr = Ytr_backup

            logging.info("computing mouse:polynomial_kernel keyboard:polynomial_kernel...")
            print('computing mouse:polynomial_kernel keyboard:polynomial_kernel...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.polynomial_kernel(data_mouse_tr))
            KLte.append(pairwise.polynomial_kernel(data_mouse_te, data_mouse_tr))
            KLtr.append(pairwise.polynomial_kernel(data_keyboard_tr))
            KLte.append(pairwise.polynomial_kernel(data_keyboard_te, data_keyboard_tr))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            score1 = clf.score(KLte, Yte)
            y_pred = clf.predict(KLte)  # predictions

            accuracy = accuracy_score(Yte, y_pred)
            logging.info("mouse:polynomial_kernel keyboard:polynomial_kernel score:" + str(score1))
            print("mouse:polynomial_kernel keyboard:polynomial_kernel score:" + str(score1))

            from sklearn.metrics import accuracy_score, roc_auc_score

            accuracy_3.append(accuracy)

            '''
            mouse:linear_kernel keyboard:rbf_kernel
            '''
            Yte = Yte_backup
            Ytr = Ytr_backup

            logging.info("computing rbf_kernel Kernels...")
            print('computing rbf_kernel Kernels...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.linear_kernel(data_mouse_tr))
            KLte.append(pairwise.linear_kernel(data_mouse_te, data_mouse_tr))
            KLtr.append(pairwise.rbf_kernel(data_keyboard_tr))
            KLte.append(pairwise.rbf_kernel(data_keyboard_te, data_keyboard_tr))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            score1 = clf.score(KLte, Yte)
            logging.info("mouse:linear_kernel keyboard:rbf_kernel score:" + str(score1))
            print("mouse:linear_kernel keyboard:rbf_kernel score:" + str(score1))

            accuracy_4.append(score1)

            '''
                mouse:rbf_kernel keyboard:linear_kernel
            '''
            Yte = Yte_backup
            Ytr = Ytr_backup

            logging.info("computing mouse:rbf_kernel keyboard:linear_kernel...")
            print('computing mouse:rbf_kernel keyboard:linear_kernel...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.rbf_kernel(data_mouse_tr, gamma=2))
            KLte.append(pairwise.rbf_kernel(data_mouse_te, data_mouse_tr, gamma=2))
            KLtr.append(pairwise.linear_kernel(data_keyboard_tr))
            KLte.append(pairwise.linear_kernel(data_keyboard_te, data_keyboard_tr))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            score1 = clf.score(KLte, Yte)
            logging.info("mouse:rbf_kernel keyboard:linear_kernel score:" + str(score1))
            print("mouse:rbf_kernel keyboard:linear_kernel score:" + str(score1))

            accuracy_5.append(score1)

            '''
                    mouse:linear_kernel keyboard:polynomial_kernel
            '''
            Yte = Yte_backup
            Ytr = Ytr_backup

            logging.info("computing mouse:polynomial_kernel keyboard:polynomial_kernel...")
            print('computing mouse:polynomial_kernel keyboard:polynomial_kernel...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.linear_kernel(data_mouse_tr))
            KLte.append(pairwise.linear_kernel(data_mouse_te, data_mouse_tr))
            KLtr.append(pairwise.polynomial_kernel(data_keyboard_tr))
            KLte.append(pairwise.polynomial_kernel(data_keyboard_te, data_keyboard_tr))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            score1 = clf.score(KLte, Yte)
            logging.info("mouse:linear_kernel keyboard:polynomial_kernel score:" + str(score1))
            print("mouse:linear_kernel keyboard:polynomial_kernel score:" + str(score1))

            accuracy_6.append(score1)

            '''
                mouse:polynomial_kernel keyboard:linear_kernel
            '''
            Yte = Yte_backup
            Ytr = Ytr_backup

            logging.info("computing mouse:polynomial_kernel keyboard:polynomial_kernel...")
            print('computing mouse:polynomial_kernel keyboard:polynomial_kernel...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.polynomial_kernel(data_mouse_tr))
            KLte.append(pairwise.polynomial_kernel(data_mouse_te, data_mouse_tr))
            KLtr.append(pairwise.linear_kernel(data_keyboard_tr))
            KLte.append(pairwise.linear_kernel(data_keyboard_te, data_keyboard_tr))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            score1 = clf.score(KLte, Yte)
            logging.info("mouse:polynomial_kernel keyboard:linear_kernel score:" + str(score1))
            print("mouse:polynomial_kernel keyboard:linear_kernel score:" + str(score1))

            accuracy_7.append(score1)

            '''
                    mouse:polynomial_kernel keyboard:rbf_kernel
            '''
            Yte = Yte_backup
            Ytr = Ytr_backup

            logging.info("computing mouse:polynomial_kernel keyboard:polynomial_kernel...")
            print('computing mouse:polynomial_kernel keyboard:polynomial_kernel...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.polynomial_kernel(data_mouse_tr))
            KLte.append(pairwise.polynomial_kernel(data_mouse_te, data_mouse_tr))
            KLtr.append(pairwise.rbf_kernel(data_keyboard_tr, gamma=2))
            KLte.append(pairwise.rbf_kernel(data_keyboard_te, data_keyboard_tr, gamma=2))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            score1 = clf.score(KLte, Yte)
            logging.info("mouse:polynomial_kernel keyboard:rbf_kernel score:" + str(score1))
            print("mouse:polynomial_kernel keyboard:rbf_kernel score:" + str(score1))

            accuracy_8.append(score1)

            '''
                mouse:rbf_kernel keyboard:polynomial_kernel 
            '''
            Yte = Yte_backup
            Ytr = Ytr_backup

            logging.info("computing mouse:polynomial_kernel keyboard:polynomial_kernel...")
            print('computing mouse:polynomial_kernel keyboard:polynomial_kernel...', end='')
            from MKLpy.metrics import pairwise

            KLtr = []
            KLte = []
            KLtr.append(pairwise.rbf_kernel(data_mouse_tr, gamma=2))
            KLte.append(pairwise.rbf_kernel(data_mouse_te, data_mouse_tr, gamma=2))
            KLtr.append(pairwise.polynomial_kernel(data_keyboard_tr))
            KLte.append(pairwise.polynomial_kernel(data_keyboard_te, data_keyboard_tr))
            print('done')

            from MKLpy.algorithms import AverageMKL, EasyMKL, \
                KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

            logging.info('training AverageMKL...')
            print('training AverageMKL...', end='')

            clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
            logging.info('training AverageMKL...done')
            print('done')
            score1 = clf.score(KLte, Yte)
            logging.info("mouse:rbf_kernel keyboard:polynomial_kernel  score:" + str(score1))
            print("mouse:rbf_kernel keyboard:polynomial_kernel  score:" + str(score1))

            accuracy_9.append(score1)

        print(str(t * 30) + "over!")
        accuracy_1_mean = mean(accuracy_1)
        accuracy_2_mean = mean(accuracy_2)
        accuracy_3_mean = mean(accuracy_3)
        accuracy_4_mean = mean(accuracy_4)
        accuracy_5_mean = mean(accuracy_5)
        accuracy_6_mean = mean(accuracy_6)
        accuracy_7_mean = mean(accuracy_7)
        accuracy_8_mean = mean(accuracy_8)
        accuracy_9_mean = mean(accuracy_9)

        logging.info(str(t * 30) + " accuracy_1:" + str(accuracy_1_mean))
        logging.info(str(t * 30) + " accuracy_2:" + str(accuracy_2_mean))
        logging.info(str(t * 30) + " accuracy_3:" + str(accuracy_3_mean))
        logging.info(str(t * 30) + " accuracy_4:" + str(accuracy_4_mean))
        logging.info(str(t * 30) + " accuracy_5:" + str(accuracy_5_mean))
        logging.info(str(t * 30) + " accuracy_6:" + str(accuracy_6_mean))
        logging.info(str(t * 30) + " accuracy_7:" + str(accuracy_7_mean))
        logging.info(str(t * 30) + " accuracy_8:" + str(accuracy_8_mean))
        logging.info(str(t * 30) + " accuracy_9:" + str(accuracy_9_mean))
