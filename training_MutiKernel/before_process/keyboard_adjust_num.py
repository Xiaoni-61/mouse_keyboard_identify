#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 5/5/2022 下午 7:58
# @Author: xiaoni
# @File  : keyboard_adjust_num.py

import os


def copy_dirs(src_path, target_path):
    file_count = 0
    source_path = os.path.abspath(src_path)
    target_path = os.path.abspath(target_path)
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if os.path.exists(source_path):
        for root, dirs, files in os.walk(source_path):
            for file in files:
                src_file = os.path.join(root, file)
                # shutil.copy(src_file, target_path)
                file_count += 1
                print(src_file)
    return int(file_count)


if __name__ == '__main__':
    hours = ['1小时', '2小时', '3小时', '4小时', '5小时', '6小时', '7小时', '8小时', '9小时', '10小时']
    names = ['chenyu', 'dongzhenyu', 'leishengchuan', 'liyajie', 'maoyu', 'niudanyang', 'renqiuchen', 'shenyanping',
             'wangcheng', 'wanghe']

    path = "D:\大学生活\大四下\计算机毕设\数据2\\tenPeople"
    for i in range(len(hours)):
        for j in range(len(names)):
            patha = path + '\\' + names[j] + '\\' + hours[i] + '_keyboard'
            if os.path.exists(patha):
                for root, dirs, files in os.walk(patha):
                    for ii in range(len(files)):
                        old = patha + '\\' + files[ii]
                        new = patha + '\\' + str(ii + 1)+'.txt'
                        os.rename(old, new)
