# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/10/29 下午9:56
# @Disc    :


import os

def check_create_dir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)