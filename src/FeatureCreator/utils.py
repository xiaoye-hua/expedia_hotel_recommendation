# -*- coding: utf-8 -*-
# @File    : utils.py
# @Author  : Hua Guo
# @Disc    :


def get_label(row):
    if row['booking_bool'] == 1:
        return 5
    elif row['click_bool'] == 1:
        return 2
    else:
        return 0