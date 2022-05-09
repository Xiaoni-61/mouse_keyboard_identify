#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 8/5/2022 下午 2:15
# @Author: xiaoni
# @File  : end.py

import numpy as np
import os
import logging
import torch
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler, label_binarize

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG  # 设置日志输出格式
                        , filename="end.log"  # log日志输出的文件位置和文件名
                        # , filemode="w"  # 文件的写入格式，w为重新写入文件，默认是追加
                        , format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                        # -8表示占位符，让输出左对齐，输出长度都为8位
                        , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                        )
    pwdpath = os.getcwd()
    for t in range(1, 2):

        path = '/data/mouse_data/' + str(t * 300) + 's_sum/' + str(t * 300) + 's_sum.txt'
        pathsum = pwdpath + path
        data = np.loadtxt(pwdpath + path)
        X, Y = np.split(data, indices_or_sections=(len(data[0]) - 1,), axis=1)
        Y = np.squeeze(Y)
        Y = np.array([int(i) for i in Y])
        print("asd")
        print('done')

        '''
        WARNING: be sure that your matrix is not sparse! EXAMPLE:
        from sklearn.datasets import load_svmlight_file
        X,Y = load_svmlight_file(...)
        X = X.toarray()
        '''

        # preprocess data
        logging.info("preprocessing data...")
        print('preprocessing data...', end='')

        from MKLpy.preprocessing import normalization, rescale_01

        # Xsc = MinMaxScaler()
        # X = Xsc.fit_transform(X)

        X = rescale_01(X)  # feature scaling in [0,1]
        # X = normalization(X)  # ||X_i||_2^2 = 1

        # train/test split
        from sklearn.model_selection import train_test_split

        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.25, random_state=2)
        print('done')

        # compute homogeneous polynomial kernels with degrees 0,1,2,...,10.
        logging.info("computing Homogeneous Polynomial Kernels...")
        print('computing Homogeneous Polynomial Kernels...', end='')
        from MKLpy.metrics import pairwise

        Xtr = torch.tensor(Xtr)
        KLtr = [pairwise.rbf_kernel(Xtr, gamma=d) for d in range(1, 3)]
        KLte = [pairwise.rbf_kernel(Xte, Xtr, gamma=d) for d in range(1, 3)]

        # KLtr1 = [pairwise.homogeneous_polynomial_kernel(Xtr, degree=d) for d in range(3, 5)]
        # KLte1 = [pairwise.homogeneous_polynomial_kernel(Xte, Xtr, degree=d) for d in range(3, 5)]
        # KLtr1 = [pairwise.linear_kernel(Xtr)]
        print('done')

        # MKL algorithms
        from MKLpy.algorithms import AverageMKL, EasyMKL, \
            KOMD  # KOMD is not a MKL algorithm but a simple kernel machine like the SVM

        logging.info('training AverageMKL...')

        print('training AverageMKL...')
        print('training AverageMKL...', end='')
        base_learner = svm.SVC(cache_size=2000)
        # clf.fit(KLtr, Ytr)
        clf = AverageMKL(learner=base_learner).fit(KLtr, Ytr)  # a wrapper for averaging kernels
        logging.info('training AverageMKL...done')
        print('done')
        # K_average = clf.solution.ker_matrix  # the combined kernel matrix

        score1 = clf.score(KLte, Yte)
        logging.info("AverageMKL 300s score:" + str(score1))
        print("AverageMKL 300s score:" + str(score1))

        # evaluate the solution
        from sklearn.metrics import accuracy_score, roc_auc_score

        y_pred = clf.predict(KLte)  # predictions
        y_score = clf.decision_function(KLte)  # rank
        accuracy = accuracy_score(Yte, y_pred)

        # Y_pred_prob = clf.predict_proba(KLte)

        y_score_sum = []
        for kk in range(len(y_score)):
            y_score_sum.append(y_score[kk])

        y_score_sum = np.array(list(y_score_sum))

        y_score_sum = y_score_sum.transpose()

        # sum1=y_score.sum(axis=1)
        Yte = label_binarize(Yte, classes=clf.classes_)
        roc_auc = roc_auc_score(Yte, y_score_sum, multi_class='ovr')  # , multi_class='ovr'
        logging.info('Accuracy score:' + str(accuracy))
        logging.info('roc AUC score:' + str(roc_auc))
        print('Accuracy score: %.3f, roc AUC score: %.3f' % (accuracy, roc_auc))
