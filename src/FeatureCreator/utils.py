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


def map_categorical(df):
    def map_position(row):
        position = row['position']
        if position > 9:
            return 10
        return position
    df['original_position'] = df['position']
    df['position'] = df.apply(lambda row: map_position(row), axis=1)
    return df