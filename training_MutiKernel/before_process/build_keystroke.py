#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 6/5/2022 上午 10:09
# @Author: xiaoni
# @File  : build_keystroke.py
import os

if __name__ == '__main__':
    hours = ['1小时', '2小时', '3小时', '4小时', '5小时', '6小时', '7小时', '8小时', '9小时', '10小时']
    names = ['chenyu', 'dongzhenyu', 'leishengchuan', 'liyajie', 'maoyu', 'niudanyang', 'renqiuchen', 'shenyanping',
             'wangcheng', 'wanghe']
    fp = open('./keystroke.txt', 'w')
    path = "D:\大学生活\大四下\计算机毕设\数据2\\tenPeople"

    for j in range(len(names)):
        for i in range(len(hours)):
            patha = path + '\\' + names[j] + '\\' + hours[i] + '_keyboard'
            fp.write(patha+'\n')
            f = open(patha + '\\1.txt', 'r')
            a = f.readline()
            a = a.split('\n')
            a = a[0].split(',')
            fp.write(a[0]+'\n')

    fp.close()
