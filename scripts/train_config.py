# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :

from src.FeatureCreator.FeatureCreator import FeatureCreator
from src.FeatureCreator.XGBFeatureCreator import XGBFeatureCreator
from src.Pipeline.XGBRegressionPipeline import XGBRegressionPipeline
from src.Pipeline.LGBRegPipeline import LGBRegPipeline


debug = False
no_test = True

if debug:
    raw_data_path = 'data/debug_data'
    model_dir = 'model_training/debug'
else:
    raw_data_path = 'data/raw_data'
    model_dir = 'model_training/'
debug_num = 10000
dir_mark = '0429_xgb_v1'
# dir_mark = '0430_lightgbm_v1'

search_id = 'srch_id'
prop_id = 'prop_id'

# base_dir = '../'
data_dir = 'raw_data/'
result_dir = 'result/'

train_config_detail = {
    "0430_lightgbm_v1": {
        "pipeline_class": LGBRegPipeline
        , 'feature_creator': FeatureCreator
        , 'train_valid': True
        , 'model_params': {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.1,
            'is_enable_sparse': True,
            'verbose': 0,
            'num_iterations':200,
            'metric_freq': 1,
            'is_training_metric': True,
            'tree_learner': 'serial',
            'bagging_freq': 5,
            'min_sum_hessian_in_leaf': 5,
            'use_two_round_loading': False,
            'num_machines': 1,
            'subsample_for_bin': 200000,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'min_split_gain': 0.0,
            'colsample_bytree': 1.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0
        }
        , 'sparse_features': [
            'site_id'
            , 'visitor_location_country_id'
            , 'prop_country_id'
            , 'srch_destination_id'
            , 'prop_brand_bool'
            , 'promotion_flag'
            , 'srch_saturday_night_bool'
            , 'random_bool'
            , 'comp1_rate'
            , 'comp1_inv'
            , 'comp2_rate'
            , 'comp2_inv'
            , 'comp3_rate'
            , 'comp3_inv'
            , 'comp4_rate'
            , 'comp4_inv'
            , 'comp5_rate'
            , 'comp5_inv'
            , 'comp6_rate'
            , 'comp6_inv'
            , 'comp7_rate'
            , 'comp7_inv'
            , 'comp8_rate'
            , 'comp8_inv'
            # new feature
            , 'day'
            , 'hour'
            , 'dayofweek'
            , 'position'
        ]
        , 'dense_features': [
            'visitor_hist_starrating'
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
            , 'comp8_rate_percent_diff'
            # new feature
               # position feature
            , 'position_ctr', 'position_ctcvr', 'position_reg_label','position_cnt'
        ]
        # , 'feature_clean_func': clean_feature
        # , 'target_col': 'whether_min'
        # , 'data_dir_mark': 'v1_0422_xgb_clareg'
    },
    "0429_xgb_v1": {
        "pipeline_class": XGBRegressionPipeline
        , 'feature_creator': XGBFeatureCreator  #
        # , 'feature_creator': FeatureCreator
        , 'train_valid': True
        , 'sparse_features': [
            'site_id'
            , 'visitor_location_country_id'
            , 'prop_country_id'
            , 'srch_destination_id'
            , 'prop_brand_bool'
            , 'promotion_flag'
            , 'srch_saturday_night_bool'
            , 'random_bool'
            , 'comp1_rate'
            , 'comp1_inv'
            , 'comp2_rate'
            , 'comp2_inv'
            , 'comp3_rate'
            , 'comp3_inv'
            , 'comp4_rate'
            , 'comp4_inv'
            , 'comp5_rate'
            , 'comp5_inv'
            , 'comp6_rate'
            , 'comp6_inv'
            , 'comp7_rate'
            , 'comp7_inv'
            , 'comp8_rate'
            , 'comp8_inv'
            # new feature
            # , 'day'
            # , 'hour'
            # , 'dayofweek'
            # , 'position'
        ]
        , 'dense_features': [
            'visitor_hist_starrating'
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
            , 'comp8_rate_percent_diff'
            # new feature
            # position feature
            # , 'position_ctr', 'position_ctcvr', 'position_reg_label', 'position_cnt'
        ]
        # , 'feature_clean_func': clean_feature
        # , 'target_col': 'whether_min'
        # , 'data_dir_mark': 'v1_0422_xgb_clareg'
    },
}
#
# raw_data_path = 'data/raw_data/case_dataset.csv'
# debug_data_path = 'data/debug_data/case_dataset.csv'