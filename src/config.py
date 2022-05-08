# -*- coding: utf-8 -*-
# @File    : config.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 上午9:01
# @Disc    : configuration of this project

import os


raw_data_usecols = ['row_id', 'is_clicked', 'timestamp', 'channel', 'site_id',
       'site_category', 'app_id', 'app_category', 'device_id', 'device_ip',
       'device_model', 'device_type', 'device_conn_type', 'P1', 'P2', 'P3',
       'P4']

# feature related
item_col = 'site_id'
user_col = 'device_model'
user_feature_prefix = 'user_'
user_most_click_col_prex = user_feature_prefix + 'most_clicked_'
user_feature_file = 'user_features.csv'
item_feature_file = 'item_features.csv'
channel_feature_file = 'channel_features.csv'





# Expedia Project
regression_label = 'label'
offline_feature_path = 'model_finished/offline_features/'
position_feature_path = os.path.join(offline_feature_path, 'position_features.csv')
item_feature_prefix = 'prop_id_'
dest_feature_prefix = 'srch_dest_id_'
listwise_feature_prefix = 'listwise_'
item_feature_file = 'prop_id_features.csv'
dest_feature_file = 'srch_dest_id_features.csv'
listwise_feature_file = 'listwise_feature.csv'

submission_cols_origin = ['srch_id', 'prop_id']

submission_cols =  ['SearchId','PropertyId']
search_id = 'srch_id'
prop_id = 'prop_id'
dest_id = 'srch_destination_id'

original_dense_features = ['visitor_hist_starrating'
            , 'visitor_hist_adr_usd'
            , 'prop_starrating'
            , 'prop_review_score'
            , 'prop_location_score1'
            , 'prop_location_score2'
            , 'prop_log_historical_price'
            , 'price_usd'
            , 'srch_length_of_stay'
            , 'srch_booking_window'
            , 'srch_adults_count'
            , 'srch_children_count'
            , 'srch_room_count'
            , 'srch_query_affinity_score'
            , 'orig_destination_distance'
            , 'comp1_rate_percent_diff'
            , 'comp2_rate_percent_diff'
            , 'comp3_rate_percent_diff'
            , 'comp4_rate_percent_diff'
            , 'comp5_rate_percent_diff'
            , 'comp6_rate_percent_diff'
            , 'comp7_rate_percent_diff'
            , 'comp8_rate_percent_diff']