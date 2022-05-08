# -*- coding: utf-8 -*-
# @File    : position_benchmark.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from scripts.train_config import train_config_detail, dir_mark, data_dir, debug, debug_num, model_dir, raw_data_path
from scripts.train_config import big_data_dir
from src.config import regression_label, submission_cols #, position_feature_path
from scripts.train_config import no_test
from src.Evaluation import get_ndcg
from src.FeatureCreator.utils import get_label
from src.save_submission import save_submission


# =============== Config ============
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)


# target_col = train_config_detail[dir_mark]['target_col']
pipeline_class = train_config_detail[dir_mark]['pipeline_class']
feature_creator_class = train_config_detail[dir_mark]['feature_creator']
model_params = train_config_detail[dir_mark].get('model_params', {})
# grid_search_dict = train_config_detail[dir_mark].get('grid_search_dict', None)
# model_params = train_config_detail[dir_mark].get('model_params', {})
train_valid = train_config_detail[dir_mark].get('train_valid', False)
dense_features = train_config_detail[dir_mark].get('dense_features', None)
sparse_features = train_config_detail[dir_mark].get('sparse_features', None)
feature_cols = dense_features + sparse_features
feature_clean_func = train_config_detail[dir_mark].get('feature_clean_func', None)
position_feature_included = train_config_detail[dir_mark].get('position_feature_included', False)
position_feature_cols = train_config_detail[dir_mark].get('position_feature_cols', None)
# if position_feature_included:
#     assert position_feature_cols is not None
#     position_df = pd.read_csv(position_feature_path)
additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

model_path = os.path.join(model_dir, dir_mark)



additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

# target_col = train_config_detail[dir_mark].get('target_col', reg_target_col)
# feature_used = dense_features + sparse_features
# # assert feature_used is not None
if not train_config_detail[dir_mark].get('data_dir_mark', False):
    target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
else:
    target_raw_data_dir = os.path.join(raw_data_path, train_config_detail[dir_mark].get('data_dir_mark', False))


logging.info(f"Reading data from {data_dir}")
train_df = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
logging.info(f"train shape: {train_df.shape}")


train_df['label'] = train_df.apply(lambda row: get_label(row), axis=1)
train_df['predicted'] = train_df['position']
logging.info(f"Calculating NDCG (position as rank)")
for k in [5, 10, 38]:
    print(f"{k}: {get_ndcg(train_df, k)}")

df = train_df.groupby('prop_id')['position'].mean().reset_index().rename(columns={
    'position': 'predicted'
})
# df['mean_predicted'] = train_df['position'].mean()

logging.info(f"Reading data from {data_dir}")

test_df = pd.read_pickle(os.path.join(big_data_dir, 'test.pkl'))

test_df = test_df.merge(df, how='left', on='prop_id')

test_df['predicted'].fillna(train_df['position'].mean(), inplace=True)

file_name = 'benchmark.csv'
logging.info(f"saving to {file_name}")
save_submission(rec_df=test_df, file_name=file_name)
