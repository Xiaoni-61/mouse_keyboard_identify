#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/5/2022 下午 4:49
# @Author: xiaoni
# @File  : test.py
import os

import numpy as np


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


data = np.array(np.arange(12).reshape(3, 4))
data = np.insert(data, 1, [1, 1, 1, 1], 0)
data = np.insert(data, 1, [1, 1, 1, 1], 0)
data = np.insert(data, 1, [0, 0, 0, 0, 0], 1)
data = np.insert(data, 1, [0, 0, 0, 0, 0], 1)
data = np.insert(data, 1, [0, 0, 0, 0, 0], 1)

num = 0
length = len(data[0])
j = 0
while j < (len(data[0]) - num):
    a = data[:, j]

    for p in range(len(a)):
        if a[p] == 0:
            continue
        else:
            break
    if p == (len(a) - 1):
        data = np.delete(data, j, axis=1)
        print(j)
        j = j - 1
        num = num + 1
        # for t1 in range(len(data)):
        #     data[t1].pop(j)
    j = j + 1
t = 1
print("asd")
datapath = 'data\keyboard_data\\' + str(t * 30) + 's_sum'
mkdir(datapath)
datapath2 = datapath + '\\' + str(t * 30) + 's_sum.txt'
file = open(datapath2, 'w')
file.close()

np.savetxt(datapath2, data, fmt='%d', delimiter=' ')
