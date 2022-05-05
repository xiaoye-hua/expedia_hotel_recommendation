# # -*- coding: utf-8 -*-
# # @File    : model_train.py
# # @Author  : Hua Guo
# # @Time    : 2021/10/30 下午3:47
# # @Disc    :
# import pandas as pd
# import os
# import logging
#
# from src.Evaluation import get_ndcg
# from scripts.train_config import train_config_detail, dir_mark, data_dir, model_dir, raw_data_path
#
# # =============== Config ============
# logging.basicConfig(level='INFO',
#                     format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
#                     datefmt='%a, %d %b %Y %H:%M:%S',)
#
#
# # target_col = train_config_detail[dir_mark]['target_col']
# pipeline_class = train_config_detail[dir_mark]['pipeline_class']
# feature_creator_class = train_config_detail[dir_mark]['feature_creator']
# model_params = train_config_detail[dir_mark].get('model_params', {})
# # grid_search_dict = train_config_detail[dir_mark].get('grid_search_dict', None)
# # model_params = train_config_detail[dir_mark].get('model_params', {})
# train_valid = train_config_detail[dir_mark].get('train_valid', False)
# dense_features = train_config_detail[dir_mark].get('dense_features', None)
# sparse_features = train_config_detail[dir_mark].get('sparse_features', None)
# feature_cols = dense_features + sparse_features
# feature_clean_func = train_config_detail[dir_mark].get('feature_clean_func', None)
#
# additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})
#
# model_path = os.path.join(model_dir, dir_mark)
#
# if not train_config_detail[dir_mark].get('data_dir_mark', False):
#     target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
# else:
#     target_raw_data_dir = os.path.join(raw_data_path, train_config_detail[dir_mark].get('data_dir_mark', False))
#
# additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})
#
# logging.info(f"Reading data from {data_dir}")
#
# train_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'train_df.pkl'))
# eval_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'eval_df.pkl'))
# test_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'test_df.pkl'))
#
# logging.info(f"Model predicting...")
# pipeline = pipeline_class(model_path=model_path, model_training=False, model_params=model_params)
#
# print(feature_cols)
# eval_df['predicted'] = pipeline.predict(eval_df[feature_cols])
# train_df['predicted'] = pipeline.predict(train_df[feature_cols])
# test_df['predicted'] = pipeline.predict(test_df[feature_cols])
#
# train_ndcg = get_ndcg(train_df)
# eval_ndcg = get_ndcg(eval_df)
# test_ndcg = get_ndcg(test_df)
# print(f"{train_df.shape}; {eval_df.shape}; {test_df.shape}")
# print(f"{train_ndcg}; {eval_ndcg}; {test_ndcg}")
#
#
#
#
#
