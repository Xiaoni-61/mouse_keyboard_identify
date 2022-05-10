#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/5/2022 下午 2:43
# @Author: xiaoni
# @File  : pre_keyboard.py

'''
    在维度的层面，去除所有数据没有用到的特征，使得分类结果更为精确
'''

from sklearn import svm
import logging
import numpy as np
import sklearn
import os
import time


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def keyboardEvent_label(s):
    it = {'微博键盘': 0, '淘宝键盘': 1, '游戏键盘': 2, '记事本键盘': 3}
    return it[s]


def mouseEvent_label(s):
    it = {'微博鼠标': 0, '淘宝鼠标': 1, '游戏鼠标': 2, '记事本鼠标': 3}
    return it[s]


def people_label(s):
    # s = s.encode("utf-8").decode("latin1")
    it = {'chenyu': 0, 'dongzhenyu': 1, 'leishengchuan': 2, 'liyajie': 3, 'maoyu': 4, 'niudanyang': 5, 'renqiuchen': 6,
          'shenyanping': 7, 'wangcheng': 8, 'wanghe': 9}
    return it[s]


def load_data(t):
    sum = np.array([])
    # i 指第几个小时的数据
    for i in range(10):
        for j in range(len(peopleName)):
    # for i in range(1):
    #     for j in range(2):

            path = 'D:\大学生活\大四下\计算机毕设\数据2\Keyboard\keyboard\data\\' + str(i + 1) + '小时_keyboard_' + peopleName[
                j] + '_' + str(t * 30) + 's' + '\\' + str(i + 1) + '小时_keyboard_' + peopleName[j] + '_' + str(
                t) + 's.txt'
            # path = 'data/keyboard_data/' + keyboardEventName[i] + '_' + peopleName[j] + '_' + (str)(t) + "0s/" + \
            #        keyboardEventName[i] + '_' + peopleName[j] + '_' + (str)(t) + "0s" + ".txt"
            # path = 'data/keyboard_data/'+keyboardEventName[i]+'_'+peopleName[j]+'_'+"60s/"+"test.txt"
            # data = np.loadtxt(path, delimiter=' ', converters={48510: people_label})
            data = file2array(path)
            if j == 0 and i == 0:
                sum = data
            else:
                sum = np.concatenate((sum, data))
            del data
            logging.info(peopleName[j] + ' 第' + str(i + 1) + '小时')
            print(peopleName[j] + ' 第' + str(i + 1) + '小时')
    return sum


def file2array(path, delimiter=' '):  # delimiter是数据分隔符
    fp = open(path, 'r', encoding='GBK')
    string = fp.read()  # string是一行字符串，该字符串包含文件所有内容
    fp.close()
    row_list = string.splitlines()  # splitlines默认参数是‘\n’
    data_list = [[i for i in row.strip().split(delimiter)] for row in row_list]
    sum = np.array([])
    for i in range(0, len(data_list)):
        data_list[i][-2] = people_label(data_list[i][-2])
    for i in range(0, len(data_list)):
        data_list[i].pop()
        # data_list[i].append(data_list[i][-2] * 100 + data_list[i][-1])
    for i in range(0, len(data_list)):
        data_list[i] = np.array(data_list[i])
        data_list[i] = [(int)(p) for p in data_list[i]]
        if i == 0:
            sum = data_list[i]
        elif i == 1:
            sum = np.concatenate(([sum], [data_list[i]]))
        else:
            sum = np.concatenate((sum, [data_list[i]]))
    # print(type(sum[0][0]))
    del data_list
    return sum

    # def file1array(path, delimiter=' '):  # delimiter是数据分隔符
    #     fp = open(path, 'r', encoding='GBK')
    #     string = fp.read()  # string是一行字符串，该字符串包含文件所有内容
    #     fp.close()
    #     row_list = string.splitlines()  # splitlines默认参数是‘\n’
    #     data_list = [[i for i in row.strip().split(delimiter)] for row in row_list]
    #
    #     for i in range(0, len(data_list) - 1):
    #         data_list[i][-2] = people_label(data_list[i][-2])
    #     for i in range(0, len(data_list) - 1):
    #         data_list[i][-1] = keyboardEvent_label(data_list[i][-1])
    #         data_list[i].append(data_list[i][-2] * 100 + data_list[i][-1])
    #     for i in range(0, len(data_list) - 1):
    #         data_list[i] = np.array(data_list[i])

    # return data_list


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG  # 设置日志输出格式
                        , filename="pre_keyboard.log"  # log日志输出的文件位置和文件名
                        # , filemode="w"  # 文件的写入格式，w为重新写入文件，默认是追加
                        ,
                        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s"
                        # -8表示占位符，让输出左对齐，输出长度都为8位
                        , datefmt="%Y-%m-%d %H:%M:%S"  # 时间输出的格式
                        )

    time_begin = time.time()
    logging.debug("-------------------------------------")

    peopleName = ['chenyu', 'dongzhenyu', 'leishengchuan', 'liyajie', 'maoyu', 'niudanyang', 'renqiuchen',
                  'shenyanping', 'wangcheng', 'wanghe']

    keyboardEventName = ['微博键盘', '淘宝键盘', '游戏键盘', '记事本键盘']
    mouseEventName = ['微博鼠标', '淘宝鼠标', '游戏鼠标', '记事本鼠标']
    # t代表时间间隔  *30s
    number = 0
    for t in range(7, 11):
        logging.debug(str(t * 30) + "s start read data!")
        data = load_data(t)
        logging.debug("read data ok!")
        num1 = 0
        length = len(data[0])
        j = 0
        while j < (len(data[0]) - num1):
            number = number + 1
            a = data[:, j]

            for p in range(len(a)):
                if a[p] == 0:
                    continue
                else:
                    break

            if p == (len(a) - 1):
                data = np.delete(data, j, axis=1)
                print(number)
                if number % 10 == 0:
                    logging.info("remove " + str(t * 30) + "s " + str(number))
                j = j - 1
                num1 = num1 + 1
                # for t1 in range(len(data)):
                #     data[t1].pop(j)
            j = j + 1

        print("ok")
        logging.debug("remove ok!")
        datapath = 'data\keyboard_data\\' + str(t * 30) + 's_sum'
        mkdir(datapath)
        datapath2 = datapath + '\\' + str(t * 30) + 's_sum.txt'
        file = open(datapath2, 'w')
        file.close()
        logging.debug("start save!")
        np.savetxt(datapath2, data, fmt='%d', delimiter=' ')
        logging.debug("save ok!")
        del (data)

    time_end = time.time()
    timespan = time_end - time_begin
    logging.debug("total use time:" + str(timespan))
