# -*- coding: utf-8 -*-
# @File    : train_config.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :

from src.FeatureCreator.FeatureCreator import FeatureCreator
from src.FeatureCreator.FeatureCreatorV2 import FeatureCreatorV2
from src.FeatureCreator.FeatureCreatorV3 import FeatureCreatorV3
from src.FeatureCreator.FeatureCreatorV4 import FeatureCreatorV4
from src.FeatureCreator.FeatureCreatorV5 import FeatureCreatorV5

from src.FeatureCreator.ItemFeatureCreator import ItemFeatureCreator
from src.FeatureCreator.ItemFeatureCreatorV2 import ItemFeatureCreatorV2
from src.Pipeline.XGBRegressionPipeline import XGBRegressionPipeline
from src.Pipeline.LGBRegPipeline import LGBRegPipeline
from src.Pipeline.LGBMRankerPipeline import LGBMRankerPipeline

dir_mark = '0507_lgbmranker_v2'
# dir_mark = '0429_xgb_v1'
debug = False
big_data = True
debug_num = 100000
no_test = False

if debug:
    raw_data_path = 'data/debug_data'
    model_dir = 'model_training/debug'
else:
    raw_data_path = 'data/raw_data'
    model_dir = 'model_training/'

# base_dir = '../'
big_data_dir = 'data/raw_data/big_data'
small_data_dir = 'data/raw_data/small_data'
if big_data:
    data_dir = big_data_dir
else:
    data_dir = small_data_dir
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
        , 'feature_creator': FeatureCreatorV2  #
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
    "0502_lightgbm_v1": { # excpet pipeline -> everything is the same
        "pipeline_class": LGBRegPipeline
        , 'feature_creator': FeatureCreatorV2  #
        , 'train_valid': True
        , 'model_params': {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': 0.1,
            # 'is_enable_sparse': True,
            'verbose': 0,
            'num_iterations': 2000,
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
    },
    "0503_lgbmranker_v1": {  # excpet pipeline -> everything is the same
        "pipeline_class": LGBMRankerPipeline
        , 'feature_creator': FeatureCreatorV2  #
        , 'train_valid': True
        , 'model_params':{
            "task": "train",
            # "num_leaves": 255,
            # "min_data_in_leaf": 1,
            # "min_sum_hessian_in_leaf": 100,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [
                38
                             ],
            "learning_rate": 0.05,
            'num_iterations': 2000
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
    },
    "0503_lgbmranker_v2": {  # excpet pipeline -> everything is the same
        "pipeline_class": LGBMRankerPipeline
        , 'feature_creator': FeatureCreatorV3 #
        , 'item_feature_creator': ItemFeatureCreator
        , 'train_valid': True
        , 'model_params': {
            "task": "train",
            # "num_leaves": 255,
            # "min_data_in_leaf": 1,
            # "min_sum_hessian_in_leaf": 100,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [
                38
            ],
            "learning_rate": 0.05,
            'num_iterations': 5000
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
            , 'month'
            , 'hour'
            , 'dayofweek'
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
            # item features
            , 'prop_id_size', 'prop_id_mean_visitor_hist_starrating',
            'prop_id_std_visitor_hist_starrating',
            'prop_id_median_visitor_hist_starrating',
            'prop_id_mean_visitor_hist_adr_usd', 'prop_id_std_visitor_hist_adr_usd',
            'prop_id_median_visitor_hist_adr_usd', 'prop_id_mean_prop_starrating',
            'prop_id_std_prop_starrating', 'prop_id_median_prop_starrating',
            'prop_id_mean_prop_review_score', 'prop_id_std_prop_review_score',
            'prop_id_median_prop_review_score', 'prop_id_mean_prop_location_score1',
            'prop_id_std_prop_location_score1',
            'prop_id_median_prop_location_score1',
            'prop_id_mean_prop_location_score2', 'prop_id_std_prop_location_score2',
            'prop_id_median_prop_location_score2',
            'prop_id_mean_prop_log_historical_price',
            'prop_id_std_prop_log_historical_price',
            'prop_id_median_prop_log_historical_price', 'prop_id_mean_price_usd',
            'prop_id_std_price_usd', 'prop_id_median_price_usd',
            'prop_id_mean_srch_length_of_stay', 'prop_id_std_srch_length_of_stay',
            'prop_id_median_srch_length_of_stay',
            'prop_id_mean_srch_booking_window', 'prop_id_std_srch_booking_window',
            'prop_id_median_srch_booking_window', 'prop_id_mean_srch_adults_count',
            'prop_id_std_srch_adults_count', 'prop_id_median_srch_adults_count',
            'prop_id_mean_srch_children_count', 'prop_id_std_srch_children_count',
            'prop_id_median_srch_children_count', 'prop_id_mean_srch_room_count',
            'prop_id_std_srch_room_count', 'prop_id_median_srch_room_count',
            'prop_id_mean_srch_query_affinity_score',
            'prop_id_std_srch_query_affinity_score',
            'prop_id_median_srch_query_affinity_score',
            'prop_id_mean_orig_destination_distance',
            'prop_id_std_orig_destination_distance',
            'prop_id_median_orig_destination_distance',
            'prop_id_mean_comp1_rate_percent_diff',
            'prop_id_std_comp1_rate_percent_diff',
            'prop_id_median_comp1_rate_percent_diff',
            'prop_id_mean_comp2_rate_percent_diff',
            'prop_id_std_comp2_rate_percent_diff',
            'prop_id_median_comp2_rate_percent_diff',
            'prop_id_mean_comp3_rate_percent_diff',
            'prop_id_std_comp3_rate_percent_diff',
            'prop_id_median_comp3_rate_percent_diff',
            'prop_id_mean_comp4_rate_percent_diff',
            'prop_id_std_comp4_rate_percent_diff',
            'prop_id_median_comp4_rate_percent_diff',
            'prop_id_mean_comp5_rate_percent_diff',
            'prop_id_std_comp5_rate_percent_diff',
            'prop_id_median_comp5_rate_percent_diff',
            'prop_id_mean_comp6_rate_percent_diff',
            'prop_id_std_comp6_rate_percent_diff',
            'prop_id_median_comp6_rate_percent_diff',
            'prop_id_mean_comp7_rate_percent_diff',
            'prop_id_std_comp7_rate_percent_diff',
            'prop_id_median_comp7_rate_percent_diff',
            'prop_id_mean_comp8_rate_percent_diff',
            'prop_id_std_comp8_rate_percent_diff',
            'prop_id_median_comp8_rate_percent_diff'
            , 'price_usd_percentile'
            , 'price_usd_rank_percentile'
            , 'prop_starrating_percentile'
            , 'prop_starrating_rank_percentile'
            , 'prop_review_score_percentile'
            , 'prop_review_score_rank_percentile'
            # position feature
            # , 'position_ctr', 'position_ctcvr', 'position_reg_label', 'position_cnt'
        ]
        # , 'feature_clean_func': clean_feature
        # , 'target_col': 'whether_min'
        # , 'data_dir_mark': '0429_xgb_v1'
    },
    "0506_lgbmranker_v1": {  # excpet pipeline -> everything is the same
        "pipeline_class": LGBMRankerPipeline
        , 'feature_creator': FeatureCreatorV4  #
        , 'item_feature_creator': ItemFeatureCreator
        , 'position_feature_included': True
        , 'train_valid': True
        , 'model_params': {
            "task": "train",
            # "num_leaves": 255,
            # "min_data_in_leaf": 1,
            # "min_sum_hessian_in_leaf": 100,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [
                38
            ],
            "learning_rate": 0.05,
            'num_iterations': 500
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
            , 'month'
            , 'hour'
            , 'dayofweek'
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
            # item features
            , 'prop_id_size', 'prop_id_mean_visitor_hist_starrating',
           'prop_id_std_visitor_hist_starrating',
           'prop_id_median_visitor_hist_starrating',
           'prop_id_mean_visitor_hist_adr_usd', 'prop_id_std_visitor_hist_adr_usd',
           'prop_id_median_visitor_hist_adr_usd', 'prop_id_mean_prop_starrating',
           'prop_id_std_prop_starrating', 'prop_id_median_prop_starrating',
           'prop_id_mean_prop_review_score', 'prop_id_std_prop_review_score',
           'prop_id_median_prop_review_score', 'prop_id_mean_prop_location_score1',
           'prop_id_std_prop_location_score1',
           'prop_id_median_prop_location_score1',
           'prop_id_mean_prop_location_score2', 'prop_id_std_prop_location_score2',
           'prop_id_median_prop_location_score2',
           'prop_id_mean_prop_log_historical_price',
           'prop_id_std_prop_log_historical_price',
           'prop_id_median_prop_log_historical_price', 'prop_id_mean_price_usd',
           'prop_id_std_price_usd', 'prop_id_median_price_usd',
           'prop_id_mean_srch_length_of_stay', 'prop_id_std_srch_length_of_stay',
           'prop_id_median_srch_length_of_stay',
           'prop_id_mean_srch_booking_window', 'prop_id_std_srch_booking_window',
           'prop_id_median_srch_booking_window', 'prop_id_mean_srch_adults_count',
           'prop_id_std_srch_adults_count', 'prop_id_median_srch_adults_count',
           'prop_id_mean_srch_children_count', 'prop_id_std_srch_children_count',
           'prop_id_median_srch_children_count', 'prop_id_mean_srch_room_count',
           'prop_id_std_srch_room_count', 'prop_id_median_srch_room_count',
           'prop_id_mean_srch_query_affinity_score',
           'prop_id_std_srch_query_affinity_score',
           'prop_id_median_srch_query_affinity_score',
           'prop_id_mean_orig_destination_distance',
           'prop_id_std_orig_destination_distance',
           'prop_id_median_orig_destination_distance',
           'prop_id_mean_comp1_rate_percent_diff',
           'prop_id_std_comp1_rate_percent_diff',
           'prop_id_median_comp1_rate_percent_diff',
           'prop_id_mean_comp2_rate_percent_diff',
           'prop_id_std_comp2_rate_percent_diff',
           'prop_id_median_comp2_rate_percent_diff',
           'prop_id_mean_comp3_rate_percent_diff',
           'prop_id_std_comp3_rate_percent_diff',
           'prop_id_median_comp3_rate_percent_diff',
           'prop_id_mean_comp4_rate_percent_diff',
           'prop_id_std_comp4_rate_percent_diff',
           'prop_id_median_comp4_rate_percent_diff',
           'prop_id_mean_comp5_rate_percent_diff',
           'prop_id_std_comp5_rate_percent_diff',
           'prop_id_median_comp5_rate_percent_diff',
           'prop_id_mean_comp6_rate_percent_diff',
           'prop_id_std_comp6_rate_percent_diff',
           'prop_id_median_comp6_rate_percent_diff',
           'prop_id_mean_comp7_rate_percent_diff',
           'prop_id_std_comp7_rate_percent_diff',
           'prop_id_median_comp7_rate_percent_diff',
           'prop_id_mean_comp8_rate_percent_diff',
           'prop_id_std_comp8_rate_percent_diff',
           'prop_id_median_comp8_rate_percent_diff'
            , 'price_usd_percentile'
            , 'price_usd_rank_percentile'
            # position feature
            , 'position_ctr', 'position_ctcvr', 'position_reg_label', 'position_cnt'
        ]
        , "position_feature_cols" : ['position_ctr', 'position_ctcvr', 'position_reg_label', 'position_cnt']
        # , 'feature_clean_func': clean_feature
        # , 'target_col': 'whether_min'
        # , 'data_dir_mark': '0429_xgb_v1'
    },
    "0507_lgbmranker_v1": {  # excpet pipeline -> everything is the same
        "pipeline_class": LGBMRankerPipeline
        , 'feature_creator': FeatureCreatorV5 #
        , 'item_feature_creator': ItemFeatureCreatorV2
        , 'position_feature_included': False
        , 'train_valid': True
        , 'model_params': {
            "task": "train",
            # "num_leaves": 255,
            # "min_data_in_leaf": 1,
            # "min_sum_hessian_in_leaf": 100,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [
                38
            ],
            "learning_rate": 0.02,
            'num_iterations': 50000
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
            , 'month'
            , 'hour'
            , 'dayofweek'
            # , 'position'
            , 'whether_domestic'
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
            # item features
, 'prop_id_size', 'prop_id_mean_visitor_hist_starrating',
       'prop_id_std_visitor_hist_starrating',
       'prop_id_median_visitor_hist_starrating',
       'prop_id_mean_visitor_hist_adr_usd', 'prop_id_std_visitor_hist_adr_usd',
       'prop_id_median_visitor_hist_adr_usd', 'prop_id_mean_prop_starrating',
       'prop_id_std_prop_starrating', 'prop_id_median_prop_starrating',
       'prop_id_mean_prop_review_score', 'prop_id_std_prop_review_score',
       'prop_id_median_prop_review_score', 'prop_id_mean_prop_location_score1',
       'prop_id_std_prop_location_score1',
       'prop_id_median_prop_location_score1',
       'prop_id_mean_prop_location_score2', 'prop_id_std_prop_location_score2',
       'prop_id_median_prop_location_score2',
       'prop_id_mean_prop_log_historical_price',
       'prop_id_std_prop_log_historical_price',
       'prop_id_median_prop_log_historical_price', 'prop_id_mean_price_usd',
       'prop_id_std_price_usd', 'prop_id_median_price_usd',
       'prop_id_mean_srch_length_of_stay', 'prop_id_std_srch_length_of_stay',
       'prop_id_median_srch_length_of_stay',
       'prop_id_mean_srch_booking_window', 'prop_id_std_srch_booking_window',
       'prop_id_median_srch_booking_window', 'prop_id_mean_srch_adults_count',
       'prop_id_std_srch_adults_count', 'prop_id_median_srch_adults_count',
       'prop_id_mean_srch_children_count', 'prop_id_std_srch_children_count',
       'prop_id_median_srch_children_count', 'prop_id_mean_srch_room_count',
       'prop_id_std_srch_room_count', 'prop_id_median_srch_room_count',
       'prop_id_mean_srch_query_affinity_score',
       'prop_id_std_srch_query_affinity_score',
       'prop_id_median_srch_query_affinity_score',
       'prop_id_mean_orig_destination_distance',
       'prop_id_std_orig_destination_distance',
       'prop_id_median_orig_destination_distance',
       'prop_id_mean_comp1_rate_percent_diff',
       'prop_id_std_comp1_rate_percent_diff',
       'prop_id_median_comp1_rate_percent_diff',
       'prop_id_mean_comp2_rate_percent_diff',
       'prop_id_std_comp2_rate_percent_diff',
       'prop_id_median_comp2_rate_percent_diff',
       'prop_id_mean_comp3_rate_percent_diff',
       'prop_id_std_comp3_rate_percent_diff',
       'prop_id_median_comp3_rate_percent_diff',
       'prop_id_mean_comp4_rate_percent_diff',
       'prop_id_std_comp4_rate_percent_diff',
       'prop_id_median_comp4_rate_percent_diff',
       'prop_id_mean_comp5_rate_percent_diff',
       'prop_id_std_comp5_rate_percent_diff',
       'prop_id_median_comp5_rate_percent_diff',
       'prop_id_mean_comp6_rate_percent_diff',
       'prop_id_std_comp6_rate_percent_diff',
       'prop_id_median_comp6_rate_percent_diff',
       'prop_id_mean_comp7_rate_percent_diff',
       'prop_id_std_comp7_rate_percent_diff',
       'prop_id_median_comp7_rate_percent_diff',
       'prop_id_mean_comp8_rate_percent_diff',
       'prop_id_std_comp8_rate_percent_diff',
       'prop_id_median_comp8_rate_percent_diff'
            , 'price_usd_percentile'
            , 'price_usd_rank_percentile'
            # position feature
            # , 'position_ctr', 'position_ctcvr', 'position_reg_label', 'position_cnt'
            # position ctr, cvr, click_counts
            , 'prop_id_ctr_excluded_self', 'prop_id_ctcvr_excluded_self'
            , 'srch_dest_id_size',
           'srch_dest_id_mean_visitor_hist_starrating',
           'srch_dest_id_std_visitor_hist_starrating',
           'srch_dest_id_median_visitor_hist_starrating',
           'srch_dest_id_mean_visitor_hist_adr_usd',
           'srch_dest_id_std_visitor_hist_adr_usd',
           'srch_dest_id_median_visitor_hist_adr_usd',
           'srch_dest_id_mean_prop_starrating', 'srch_dest_id_std_prop_starrating',
           'srch_dest_id_median_prop_starrating',
           'srch_dest_id_mean_prop_review_score',
           'srch_dest_id_std_prop_review_score',
           'srch_dest_id_median_prop_review_score',
           'srch_dest_id_mean_prop_location_score1',
           'srch_dest_id_std_prop_location_score1',
           'srch_dest_id_median_prop_location_score1',
           'srch_dest_id_mean_prop_location_score2',
           'srch_dest_id_std_prop_location_score2',
           'srch_dest_id_median_prop_location_score2',
           'srch_dest_id_mean_prop_log_historical_price',
           'srch_dest_id_std_prop_log_historical_price',
           'srch_dest_id_median_prop_log_historical_price',
           'srch_dest_id_mean_price_usd', 'srch_dest_id_std_price_usd',
           'srch_dest_id_median_price_usd',
           'srch_dest_id_mean_srch_length_of_stay',
           'srch_dest_id_std_srch_length_of_stay',
           'srch_dest_id_median_srch_length_of_stay',
           'srch_dest_id_mean_srch_booking_window',
           'srch_dest_id_std_srch_booking_window',
           'srch_dest_id_median_srch_booking_window',
           'srch_dest_id_mean_srch_adults_count',
           'srch_dest_id_std_srch_adults_count',
           'srch_dest_id_median_srch_adults_count',
           'srch_dest_id_mean_srch_children_count',
           'srch_dest_id_std_srch_children_count',
           'srch_dest_id_median_srch_children_count',
           'srch_dest_id_mean_srch_room_count', 'srch_dest_id_std_srch_room_count',
           'srch_dest_id_median_srch_room_count',
           'srch_dest_id_mean_srch_query_affinity_score',
           'srch_dest_id_std_srch_query_affinity_score',
           'srch_dest_id_median_srch_query_affinity_score',
           'srch_dest_id_mean_orig_destination_distance',
           'srch_dest_id_std_orig_destination_distance',
           'srch_dest_id_median_orig_destination_distance',
           'srch_dest_id_mean_comp1_rate_percent_diff',
           'srch_dest_id_std_comp1_rate_percent_diff',
           'srch_dest_id_median_comp1_rate_percent_diff',
           'srch_dest_id_mean_comp2_rate_percent_diff',
           'srch_dest_id_std_comp2_rate_percent_diff',
           'srch_dest_id_median_comp2_rate_percent_diff',
           'srch_dest_id_mean_comp3_rate_percent_diff',
           'srch_dest_id_std_comp3_rate_percent_diff',
           'srch_dest_id_median_comp3_rate_percent_diff',
           'srch_dest_id_mean_comp4_rate_percent_diff',
           'srch_dest_id_std_comp4_rate_percent_diff',
           'srch_dest_id_median_comp4_rate_percent_diff',
           'srch_dest_id_mean_comp5_rate_percent_diff',
           'srch_dest_id_std_comp5_rate_percent_diff',
           'srch_dest_id_median_comp5_rate_percent_diff',
           'srch_dest_id_mean_comp6_rate_percent_diff',
           'srch_dest_id_std_comp6_rate_percent_diff',
           'srch_dest_id_median_comp6_rate_percent_diff',
           'srch_dest_id_mean_comp7_rate_percent_diff',
           'srch_dest_id_std_comp7_rate_percent_diff',
           'srch_dest_id_median_comp7_rate_percent_diff',
           'srch_dest_id_mean_comp8_rate_percent_diff',
           'srch_dest_id_std_comp8_rate_percent_diff',
           'srch_dest_id_median_comp8_rate_percent_diff',
            # 'srch_dest_id_click_cnt',
           'srch_dest_id_ctr_excluded_self', 'srch_dest_id_ctcvr_excluded_self'
        ]
        # , "position_feature_cols": ['position_ctr', 'position_ctcvr', 'position_reg_label', 'position_cnt']
        # , 'feature_clean_func': clean_feature
        # , 'target_col': 'whether_min'
        # , 'data_dir_mark': '0429_xgb_v1'
    },
    "0507_lgbmranker_v2": {  # copy from 0503_lgbmranker_v2
        "pipeline_class": LGBMRankerPipeline
        , 'feature_creator': FeatureCreatorV3  #
        , 'item_feature_creator': ItemFeatureCreator
        , 'train_valid': True
        , 'model_params': {
            "task": "train",
            # "num_leaves": 255,
            # "min_data_in_leaf": 1,
            # "min_sum_hessian_in_leaf": 100,
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [
                38
            ],
            "learning_rate": 0.05,
            'num_iterations': 5000
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
            , 'month'
            , 'hour'
            , 'dayofweek'
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
            # item features
            , 'prop_id_size', 'prop_id_mean_visitor_hist_starrating',
            'prop_id_std_visitor_hist_starrating',
            'prop_id_median_visitor_hist_starrating',
            'prop_id_mean_visitor_hist_adr_usd', 'prop_id_std_visitor_hist_adr_usd',
            'prop_id_median_visitor_hist_adr_usd', 'prop_id_mean_prop_starrating',
            'prop_id_std_prop_starrating', 'prop_id_median_prop_starrating',
            'prop_id_mean_prop_review_score', 'prop_id_std_prop_review_score',
            'prop_id_median_prop_review_score', 'prop_id_mean_prop_location_score1',
            'prop_id_std_prop_location_score1',
            'prop_id_median_prop_location_score1',
            'prop_id_mean_prop_location_score2', 'prop_id_std_prop_location_score2',
            'prop_id_median_prop_location_score2',
            'prop_id_mean_prop_log_historical_price',
            'prop_id_std_prop_log_historical_price',
            'prop_id_median_prop_log_historical_price', 'prop_id_mean_price_usd',
            'prop_id_std_price_usd', 'prop_id_median_price_usd',
            'prop_id_mean_srch_length_of_stay', 'prop_id_std_srch_length_of_stay',
            'prop_id_median_srch_length_of_stay',
            'prop_id_mean_srch_booking_window', 'prop_id_std_srch_booking_window',
            'prop_id_median_srch_booking_window', 'prop_id_mean_srch_adults_count',
            'prop_id_std_srch_adults_count', 'prop_id_median_srch_adults_count',
            'prop_id_mean_srch_children_count', 'prop_id_std_srch_children_count',
            'prop_id_median_srch_children_count', 'prop_id_mean_srch_room_count',
            'prop_id_std_srch_room_count', 'prop_id_median_srch_room_count',
            'prop_id_mean_srch_query_affinity_score',
            'prop_id_std_srch_query_affinity_score',
            'prop_id_median_srch_query_affinity_score',
            'prop_id_mean_orig_destination_distance',
            'prop_id_std_orig_destination_distance',
            'prop_id_median_orig_destination_distance',
            'prop_id_mean_comp1_rate_percent_diff',
            'prop_id_std_comp1_rate_percent_diff',
            'prop_id_median_comp1_rate_percent_diff',
            'prop_id_mean_comp2_rate_percent_diff',
            'prop_id_std_comp2_rate_percent_diff',
            'prop_id_median_comp2_rate_percent_diff',
            'prop_id_mean_comp3_rate_percent_diff',
            'prop_id_std_comp3_rate_percent_diff',
            'prop_id_median_comp3_rate_percent_diff',
            'prop_id_mean_comp4_rate_percent_diff',
            'prop_id_std_comp4_rate_percent_diff',
            'prop_id_median_comp4_rate_percent_diff',
            'prop_id_mean_comp5_rate_percent_diff',
            'prop_id_std_comp5_rate_percent_diff',
            'prop_id_median_comp5_rate_percent_diff',
            'prop_id_mean_comp6_rate_percent_diff',
            'prop_id_std_comp6_rate_percent_diff',
            'prop_id_median_comp6_rate_percent_diff',
            'prop_id_mean_comp7_rate_percent_diff',
            'prop_id_std_comp7_rate_percent_diff',
            'prop_id_median_comp7_rate_percent_diff',
            'prop_id_mean_comp8_rate_percent_diff',
            'prop_id_std_comp8_rate_percent_diff',
            'prop_id_median_comp8_rate_percent_diff'
            , 'price_usd_percentile'
            , 'price_usd_rank_percentile'
            , 'prop_starrating_percentile'
            , 'prop_starrating_rank_percentile'
            , 'prop_review_score_percentile'
            , 'prop_review_score_rank_percentile'
            # position feature
            # , 'position_ctr', 'position_ctcvr', 'position_reg_label', 'position_cnt'
        ]
        # , 'feature_clean_func': clean_feature
        # , 'target_col': 'whether_min'
        # , 'data_dir_mark': '0429_xgb_v1'
    },

}

#
# raw_data_path = 'data/raw_data/case_dataset.csv'
# debug_data_path = 'data/debug_data/case_dataset.csv'