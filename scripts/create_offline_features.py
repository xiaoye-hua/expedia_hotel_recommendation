# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import os
import logging

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scripts.train_config import raw_data_path, big_data_dir
from scripts.train_config import train_config_detail, dir_mark, data_dir, debug, debug_num, model_dir
from src.config import regression_label, submission_cols, offline_feature_path
from src.utils.memory_utils import reduce_mem_usage
from src.utils import check_create_dir
from src.FeatureCreator.ItemFeatureCreator import ItemFeatureCreator

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
feature_clean_func = train_config_detail[dir_mark].get('feature_clean_func', None)

additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

model_path = os.path.join(model_dir, dir_mark)

target_raw_data_dir = os.path.join(raw_data_path, dir_mark)

check_create_dir(target_raw_data_dir)
additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

logging.info(f"Reading data from {data_dir}")
train_df = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
test_df = pd.read_pickle(os.path.join(big_data_dir, 'test.pkl'))

if debug:
    train_df = train_df.sample(debug_num)
    test_df = test_df.sample(debug_num)

logging.info(f"train_df: {train_df.shape}; test_df: {test_df.shape}")
logging.info(f"Creating features")


fc = ItemFeatureCreator(train_df=train_df, test_df=test_df, feature_path=offline_feature_path)

item_features, dest_features = fc.get_features()
logging.info(f"Item features: {item_features.columns}")
logging.info(f"dest features: {dest_features.columns}")

logging.info(item_features.head())
logging.info(dest_features.head())
fc.save_features()