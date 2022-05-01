# -*- coding: utf-8 -*-
# @File    : config.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 上午9:01
# @Disc    : configuration of this project


raw_data_usecols = ['row_id', 'is_clicked', 'timestamp', 'channel', 'site_id',
       'site_category', 'app_id', 'app_category', 'device_id', 'device_ip',
       'device_model', 'device_type', 'device_conn_type', 'P1', 'P2', 'P3',
       'P4']

# feature related
item_col = 'site_id'
user_col = 'device_model'
item_feature_prefix = 'item_'
user_feature_prefix = 'user_'
user_most_click_col_prex = user_feature_prefix + 'most_clicked_'
user_feature_file = 'user_features.csv'
item_feature_file = 'item_features.csv'
channel_feature_file = 'channel_features.csv'

cate_encode_cols = ['P1', 'P2', "P3", 'site_id', 'site_category', 'app_id', 'app_category'
                                 , 'device_model',
                                 "device_id"
                                 ]
