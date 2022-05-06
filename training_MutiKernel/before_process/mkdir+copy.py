# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import os
import shutil
import sys


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
                shutil.copy(src_file, target_path)
                file_count += 1
                print(src_file)
    return int(file_count)


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")
        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")

    # 调用函数


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    hours = ['1小时', '2小时', '3小时', '4小时', '5小时', '6小时', '7小时', '8小时', '9小时', '10小时']
    names = ['chenyu', 'dongzhenyu', 'leishengchuan', 'liyajie', 'maoyu', 'niudanyang', 'renqiuchen', 'shenyanping',
             'wangcheng', 'wanghe']
    pathA = "D:\大学生活\大四下\计算机毕设\数据2\\ten+hours"
    pathB = "D:\大学生活\大四下\计算机毕设\数据2\\tenPeople"
    print_hi('PyCharm')
    # D:\大学生活\大四下\计算机毕设\数据2\10+hours\1小时\chenyu
    # for i in range(0, len(hours) - 1):
    #     for j in range(0, len(names) - 1):
    #         path1 = pathA + '\\' + hours[i] + '\\' + names[j] + '\\' + 'keyboard'
    #         mkdir(path1)
    # for i in range(0, len(hours)):
    #     for j in range(0, len(names)):
    #         path1 = path + '\\' + names[j] + '\\' + hours[i] + '_' + 'keyboard'
    #         mkdir(path1)
    #         path2 = path + '\\' + names[j] + '\\' + hours[i] + '_' + 'mousemovements'
    #         mkdir(path2)
    for i in range(0, len(hours)):
        for j in range(0, len(names)):
            path_origin_keyboard = pathA + '\\' + hours[i] + '\\' + names[j] + '\\' + 'keyboard'
            path_origin_mouse = pathA + '\\' + hours[i] + '\\' + names[j] + '\\' + 'mousemovements'
            path2_keyboard = pathB + '\\' + names[j] + '\\' + hours[i] + '_' + 'keyboard'
            path2_mousemovements = pathB + '\\' + names[j] + '\\' + hours[i] + '_' + 'mousemovements'

            folder1 = os.path.exists(path_origin_keyboard)
            folder2 = os.path.exists(path_origin_mouse)

            folder3 = os.path.exists(path2_keyboard)
            folder4 = os.path.exists(path2_mousemovements)
            # mkdir(path2_mousemovements)
            # os.system('copy' + ' ' + path_origin_keyboard + ' ' + path2_keyboard)
            # os.system('copy' + ' ' + path_origin_mouse + ' ' + path2_mousemovements)
            a = copy_dirs(path_origin_keyboard, path2_keyboard)
            print("a="+str(a))
            b = copy_dirs(path_origin_mouse, path2_mousemovements)
            print("b="+str(b))
            print(hours[i] + '\\' + names[j])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/

# mkdir('data')
# mkdir('data1')
# path1 = 'D:\大学生活\大四下\计算机毕设\数据2\建立keyboard\data'
# path2 = 'D:\大学生活\大四下\计算机毕设\数据2\建立keyboard\data1'
# # os.system('copy'+' '+path1+' '+path2)
# # os.system('copy D:\大学生活\大四下\计算机毕设\数据2\建立keyboard\data\* D:\大学生活\大四下\计算机毕设\数据2\建立keyboard\data1\\')
# a = copy_dirs(path1, path2)
# print(a)

# try:
#     shutil.copy('D:\大学生活\大四下\计算机毕设\数据2\\ten+hours\\1小时\chenyu\keyboard\*',
#                 'D:\大学生活\大四下\计算机毕设\数据2\\tenPeople\\chenyu\\1小时_keyboard\\')
# except IOError as e:
#     print("Unable to copy file. %s" % e)
# except:
#     print("Unexpected error:", sys.exc_info())
# # os.system('copy D:\大学生活\大四下\计算机毕设\数据2\\ten+hours\\1小时\chenyu\keyboard\* D:\大学生活\大四下\计算机毕设\数据2\\tenPeople\\chenyu\\1小时_keyboard\\')
