# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :

from src.FeatureCreator import FeatureCreator
from src.Pipeline.XGBoostPipeline import XGBoostPipeline

debug = True
debug_num = 1000
dir_mark = '0430_lightgbm_v1'

submission_cols =  ['SearchId','PropertyId']
search_id = 'srch_id'
prop_id = 'prop_id'
regression_label = 'label'

# base_dir = '../'
data_dir = 'raw_data/'
result_dir = 'result/'

position_feature_path = 'model_finished/offline_features/position_features.csv'

train_config_detail = {
    "0430_lightgbm_v1": {
        "pipeline_class": XGBoostPipeline
        , 'feature_creator': FeatureCreator
        , 'train_valid': True
        , 'sparse_features': [

            # 'has_free_cancellation',
            #     'can_pay_later',  -> error TypeError: ufunc 'isnan' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''

            # 'month',
            # 'day_name',
            'checkin_month',  # try XGB category
            'checkin_day_name',
            'domestic',
            'search_device_type',  # remove null value
            # 'country_code',
            # 'search_country_code',
            'property_type',  # 32 unique value -> can we reduce the # of categories
            'parent_chain_name',  # 58 unique value
            'hotel_country_code_whether_us',
            'search_country_code_whether_us',
            'hotel_country_code_country_group'
            , 'search_country_code_country_group'
        ]
        , 'dense_features': [
            'total_rate_usd',
            # 'random_data',
            # 'random_data_2',
            'day_distance',
            'length_of_stay',
            'base_rate_usd',
            'daily_rate_usd',
            'top3_min_diff',
            'top3_max_diff',
            'bks_diff',
            'exp_diff',
            'top3_min_frac',
            'top3_max_frac',
            'bks_diff_frac',
            'exp_diff_frac',
            # 'cte_diff'
            # 'eps_diff'
        ]
        , 'feature_clean_func': clean_feature
        , 'target_col': 'whether_min'
        , 'data_dir_mark': 'v1_0422_xgb_clareg'
    },
}
#
# raw_data_path = 'data/raw_data/case_dataset.csv'
# debug_data_path = 'data/debug_data/case_dataset.csv'