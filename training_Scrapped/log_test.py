#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 3/5/2022 下午 12:44
#@Author: xiaoni
#@File  : log_test.py

import math
import logging

#默认的warning级别，只输出warning以上的
#使用basicConfig()来指定日志级别和相关信息

logging.basicConfig(level=logging.DEBUG #设置日志输出格式
                    ,filename="demo.log" #log日志输出的文件位置和文件名
                    ,filemode="w" #文件的写入格式，w为重新写入文件，默认是追加
                    ,format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s" #日志输出的格式
                    # -8表示占位符，让输出左对齐，输出长度都为8位
                    ,datefmt="%Y-%m-%d %H:%M:%S" #时间输出的格式
                    )

logging.debug("This is  DEBUG !!")
logging.info("This is  INFO !!")
logging.warning("This is  WARNING !!")
logging.error("This is  ERROR !!")
logging.critical("This is  CRITICAL !!")

#在实际项目中，捕获异常的时候，如果使用logging.error(e)，只提示指定的logging信息，不会出现
#为什么会错的信息，所以要使用logging.exception(e)去记录。

try:
    3/0
except Exception as e:
    # logging.error(e)
    logging.exception(e)