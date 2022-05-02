import os
import math
import copy

'''
filepath = input("input the path of out file:\n")
outpath = filepath + "\\out.txt"
# filepath = 'D:\project\鼠标键盘特征提取\白雪涛\记事本鼠标\out.txt'
'''

# 不同移动距离下，事件速度的平均值
def cal_MSD(data):
    all_speed = [[] for i in range(8)]
    for t in data:
        if int(t[2]) == 0:
            continue
        index = int(t[2]) // 100
        if index == 8:
            index -= 1
        all_speed[index].append(float(t[4]))

    avg_speed = []
    for t in all_speed:
        if not len(t):
            avg_speed.append(0)
        else:
            avg_speed.append(sum(t) / len(t))

    return avg_speed

#speed
def cal_MDA(data):
    all_speed = [[] for i in range(8)]
    for t in data:
        if int(t[5]) == 0:
            continue
        index = int(t[5]) - 1
        all_speed[index].append(float(t[4]))

    avg_speed = []
    for t in all_speed:
        if not len(t):
            avg_speed.append(0)
        else:
            avg_speed.append(sum(t) / len(t))

    return avg_speed

# 归一化（各方向比例）
def cal_MDH(data):
    all_op = [0 for i in range(8)]
    for t in data:
        if int(t[5]) == 0:
            continue
        index = int(t[5]) - 1
        all_op[index] += 1

    avg_op = []
    for i in all_op:
        if sum(all_op) == 0:
            avg_op.append(0)
        else:
            avg_op.append(i / sum(all_op))

    return avg_op

# 每种事件的 平均速度
def cal_ATA(data):
    move = [1]
    drag = [2]
    click = [3, 4, 5]

    all_speed = [[] for i in range(3)]
    for t in data:
        tmp = int(t[1])
        index = 0
        if tmp in move:
            index = 0
        elif tmp in drag:
            index = 1
        elif tmp in click:
            index = 2

        all_speed[index].append(float(t[4]))

    avg_speed = []
    for t in all_speed:
        if not len(t):
            avg_speed.append(0)
        else:
            avg_speed.append(sum(t) / len(t))

    return avg_speed

# 每种事件在这个时间片内所占的比例（归一化）
def cal_ATH(data):
    move = [1]
    drag = [2]
    click = [3, 4, 5]

    all_op = [0 for i in range(3)]
    for t in data:
        tmp = int(t[1])
        index = 0
        if tmp in move:
            index = 0
        elif tmp in drag:
            index = 1
        elif tmp in click:
            index = 2

        all_op[index] += 1

    avg_op = []
    for i in all_op:
        if sum(all_op) == 0:
            avg_op.append(0)
        else:
            avg_op.append(i / sum(all_op))

    return avg_op

# 移动distance出现次数（从100到800）占总次数的比例
def cal_TDH(data):
    all_op = [0 for i in range(8)]
    for t in data:
        if int(t[2]) == 0:
            continue
        index = int(t[2]) // 100
        all_op[index] += 1

    avg_op = []
    for i in all_op:
        if sum(all_op) == 0:
            avg_op.append(0)
        else:
            avg_op.append(i / sum(all_op))

    return avg_op

# 一次事件的所用时间的次数 占总事件数量的比例
def cal_MTH(data):
    all_op = [0 for i in range(5)]
    for t in data:
        tt = int(t[7]) - int(t[6])
        if tt >= 1000 or tt < 0:
            continue
        index = tt // 200
        if index == 5:
            index -= 1
        all_op[index] += 1

    avg_op = []
    for i in all_op:
        if sum(all_op) == 0:
            avg_op.append(0)
        else:
            avg_op.append(i / sum(all_op))

    return avg_op


def cal_all(data):
    all_list = []
    all_list.append(cal_MSD(data))
    all_list.append(cal_MDA(data))
    all_list.append(cal_MDH(data))
    all_list.append(cal_ATA(data))
    all_list.append(cal_ATH(data))
    all_list.append(cal_TDH(data))
    all_list.append(cal_MTH(data))
    res_list = []
    for t in all_list:
        for j in t:
            res_list.append(copy.deepcopy(j))
    return res_list


Dirs = ['\\记事本鼠标', '\\微博鼠标', '\\淘宝鼠标', '\\游戏鼠标']
# Allpath = input("input the file path:\n")

all_list = []
res = []
# classified_list = {}
flist = open('..\mouse.txt', 'rb')
# flist = open('D:\\硬盘\\学习\\学习\\Keyboard\\mouse.txt', 'rb')
all_file = flist.readlines()
for Allpath in all_file:
    for Dir in Dirs:
        if not isinstance(Allpath, str):            # 判断该对象类型
            Allpath = Allpath.decode().strip('\n')  # 解码
        else:
            Allpath = Allpath.strip('\n')
        Allpath = Allpath.strip('\r')
        filepath = Allpath + Dir
        out_path = filepath + '\\out.txt'
        fo = open(out_path, 'rb')
        # interval = 1

        # interval = int(input("input interval:"))  # Unit: seconds
        # interval *= 1000
        # all_interval = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        # all_overlapping = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
        # all_overlapping = [0, 0.20, 0.40, 0.60, 0.80]
        all_overlapping = [0]
        all_line = fo.readlines()
        # all_list = []
        tmp_list = []
        for line in all_line:
            # print(line)
            tmp = str(line).split(',')[:-2]             # 去掉最后两列数据
            # all_list.append(tmp)
            tmp_list.append(tmp)

        all_list = [x for x in tmp_list if int(x[6]) != 0 and int(x[2]) <= 800]
        all_list = sorted(all_list, key=lambda x: int(x[6]))
        start_time = int(all_list[0][6])
        # end_time = int(all_list[-1][7])

        all_name = filepath.split('\\')

        for interval in range(10, 61, 10):
            interval *= 1000
            for overlapping in all_overlapping:
                # slice_num = 1 // round((1 - overlapping), 2)  # 获得一个时间片内可被交叠剩余长度切割的完整切片数量（向下取整）
                # slice_length = interval * round((1 - overlapping), 2)  # 切片长度
                res = []
                classified_list = {}
                num = 0
                if 3600000 % interval:
                    num = (3600000 + interval) // interval
                else:
                    num = 3600000 // interval

                # 交叠
                # num = math.ceil((num - 1) / round((1 - overlapping), 2)) + 1
                # 结束时间
                # end_time = (num - 1) * slice_length + interval + start_time
                for i in range(num):
                    classified_list[i] = []

                cnt = 0
                for line in all_list:
                    if int(line[2]) >= 800:
                        continue
                    if int(line[6]) - start_time > 3600000:
                        break
                    if int(line[6]) == 0:
                        continue
                    # index = (int(line[6]) - start_time + interval - 1) // interval
                    # 一条数据位于多个时间片内
                    '''
                    index = []
                    if int(line[6]) <= math.ceil(start_time + slice_length * slice_num):  # 第一个时间片内特殊处理
                        slice_index = (int(line[6]) - start_time) // slice_length  # 初始为哪个切片内
                        for i in range(int(slice_index + 1)):
                            # index.append(i)
                            classified_list[i].append(line)
                    elif int(line[6]) >= math.ceil(end_time - slice_length * slice_num):  # 最后一个时间片内特殊处理
                        slice_index = (end_time - int(line[6])) // slice_length
                        for i in range(int(slice_index + 1)):
                            # index.append(num - i - 1)
                            classified_list[num - i - 1].append(line)
                    else:  # 中间使用通用方法
                        slice_index = (int(line[6]) - start_time) // slice_length  # 获得前面有几个切片长度，即为现在最后的下标
                        if (int(line[6]) - start_time) % slice_length <= interval - slice_num * slice_length:
                            for i in range(int(slice_index - slice_num), int(slice_index + 1)):
                                # index.append(i)
                                classified_list[i].append(line)
                        else:
                            for i in range(int(slice_index - slice_num + 1), int(slice_index + 1)):
                                # index.append(i)
                                classified_list[i].append(line)
                                '''
                    index = (int(line[6]) - start_time) // interval
                    if index == num:
                        index -= 1

                    classified_list[index].append(line)

                for i in classified_list:
                    res.append(copy.deepcopy(cal_all(classified_list[i])))

                if not os.path.exists('./' + 'generate_data/' + all_name[6] + '_' + str(int(interval / 1000)) + 's'):
                    os.makedirs('./' + all_name[6] + '_' + str(int(interval / 1000)) + 's')
                file_name = './' + 'generate_data/' + all_name[6] + '_' + str(int(interval / 1000)) + 's/' + \
                            all_name[6] + '_' + all_name[7] + '_' + str(int(interval / 1000)) + 's.txt'
                print(res)
                file = open(file_name, 'w')
                for i in range(len(res)):
                    for j in range(len(res[0])):
                        file.write(str(res[i][j]))
                        file.write(' ')
                    file.write(all_name[6] + '\n')
                file.close()
        print("\n"+"!!"+all_name[6])
        