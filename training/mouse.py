#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 2/5/2022 下午 11:00
# @Author: xiaoni
# @File  : mouse.py


from sklearn import svm
import numpy as np
import sklearn

import time


def keyboardEvent_label(s):
    it = {'微博键盘': 0, '淘宝键盘': 1, '游戏键盘': 2, '记事本键盘': 3}
    return it[s]


def mouseEvent_label(s):
    it = {'微博鼠标': 0, '淘宝鼠标': 1, '游戏鼠标': 2, '记事本鼠标': 3}
    return it[s]


def people_label(s):
    # s = s.encode("utf-8").decode("latin1")
    it = {'于高远': 0, '任义婷': 1, '何晨曦': 2, '侯宇': 3, '刘思言': 4, '刘晓晴': 5, '吴雨晗': 6, '宋传勇': 7, '尹悦': 8, '庞钦玉': 9, '康笑笑': 10,
          '张博伦': 11, '张岩': 12, '张潇': 13, '张燕丘': 14, '张翰澄': 15, '张翰澄': 16, '彭婷': 17, '慕聪颖': 18, '扈新宇': 19, '李亦铭': 20,
          '李健': 21, '李国坤': 22, '李涵乔': 23, '汪慧灵': 24, '沙嫣茹': 25, '王广飞': 26,
          '王智颖': 27, '王江宁': 28, '王童童': 29, '王靖怡': 30, '田静': 31, '白雪涛': 32, '纪傲': 33, '肖婧琦': 34, '苟熙': 35, '许杨': 36,
          '金戈慧': 37, '陶泽南': 38, '高驰': 39, '黄思宇': 40}
    return it[s]


def load_data(t):
    sum = np.array([])
    for j in range(len(peopleName)):
        for i in range(len(keyboardEventName)):
    # for i in range(1):
    #     for j in range(3):
            # D:\大学生活\大四下\计算机毕设\代码\data\keyboard_data\微博键盘_于高远_60s
            # path = 'data/mouse_data/' + keyboardEventName[i] + '_' + peopleName[j] + '_' + (str)(t) + "0s/" + \
            #       keyboardEventName[i] + '_' + peopleName[j] + '_' + (str)(t) + "0s" + ".txt"
            path = 'data/mouse_data/' + peopleName[j] + '_' + (str)(t) + '0s/' + peopleName[j] + '_' + \
                   mouseEventName[i] + '_' + (str)(t) + '0s' + ".txt"
            # data = np.loadtxt(path, delimiter=' ', converters={48510: people_label})
            data = file2array(path)
            if j == 0 and i == 0:
                sum = data
            else:
                sum = np.concatenate((sum, data))
            del data
            print(peopleName[j] + mouseEventName[i])
    return sum


def file2array(path, delimiter=' '):  # delimiter是数据分隔符
    fp = open(path, 'r', encoding='GBK')
    string = fp.read()  # string是一行字符串，该字符串包含文件所有内容
    fp.close()
    row_list = string.splitlines()  # splitlines默认参数是‘\n’
    data_list = [[i for i in row.strip().split(delimiter)] for row in row_list]
    sum = np.array([])
    for i in range(0, len(data_list) - 1):
        data_list[i][-2] = people_label(data_list[i][-2])
    for i in range(0, len(data_list) - 1):
        data_list[i][-1] = mouseEvent_label(data_list[i][-1])
        data_list[i].append(data_list[i][-2] * 100 + data_list[i][-1])
    for i in range(0, len(data_list) - 1):
        data_list[i] = np.array(data_list[i])
        data_list[i] = [np.float32(p) for p in data_list[i]]
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

    time_begin = time.time()

    peopleName = ['于高远', '任义婷', '何晨曦', '侯宇', '刘思言', '刘晓晴', '吴雨晗', '宋传勇', '尹悦', '庞钦玉', '康笑笑',
                  '张博伦', '张岩', '张潇', '张燕丘', '张翰澄', '张翰澄', '彭婷', '慕聪颖', '扈新宇', '李亦铭', '李健',
                  '李国坤', '李涵乔', '汪慧灵', '沙嫣茹', '王广飞', '王智颖', '王江宁', '王童童', '王靖怡', '田静', '白雪涛', '纪傲', '肖婧琦', '苟熙', '许杨',
                  '金戈慧', '陶泽南', '高驰', '黄思宇']

    keyboardEventName = ['微博键盘', '淘宝键盘', '游戏键盘', '记事本键盘']
    mouseEventName = ['微博鼠标', '淘宝鼠标', '游戏鼠标', '记事本鼠标']
    for t in range(1, 6):
        data = load_data(t)
        x, y, z = np.split(data, indices_or_sections=(43, 45), axis=1)  # x为数据，y为标签,axis是分割的方向，1表示横向，0表示纵向，默认为0
        x = x[:, 0:43]  # 为便于后边画图显示，只选取前两维度。若不用画图，可选取前四列x[:,0:4]
        train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(x,
                                                                                                  z,
                                                                                                  random_state=1,
                                                                                                  # 作用是通过随机数来随机取得一定量得样本作为训练样本和测试样本
                                                                                                  train_size=0.7,
                                                                                                  test_size=0.3)
        # train_data:训练样本，test_data：测试样本，train_label：训练样本标签，test_label：测试样本标签
        classifier = svm.SVC(C=1, kernel='rbf', gamma=30, decision_function_shape='ovo')  # ovr:一对多策略 高斯核
        # classifier = svm.SVC(C=1, kernel='sigmoid')  # ovr:一对多策略 高斯核
        print("开始训练！")
        classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时默认是行序优先

        # 4.计算svc分类器的准确率
        print(str(t) + "0s_训练集得分：", classifier.score(train_data, train_label))
        print(str(t) + "0s_测试集得分：", classifier.score(test_data, test_label))
        break

    print("over!")

    time_end = time.time()
    time = time_end - time_begin
    print("time:" + (str)(time))
