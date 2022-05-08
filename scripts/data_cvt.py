# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :
import pandas as pd
import os
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scripts.train_config import raw_data_path
from scripts.train_config import train_config_detail, dir_mark, data_dir, debug, debug_num, model_dir
from scripts.train_config import big_data_dir
from src.config import regression_label, submission_cols
from src.utils.memory_utils import reduce_mem_usage
from src.utils import check_create_dir

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
item_feature_creator = train_config_detail[dir_mark].get('item_feature_creator', None)

feature_cols = dense_features + sparse_features

check_create_dir(target_raw_data_dir)
additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

logging.info(f"Reading data from {data_dir}")
all_df = pd.read_pickle(os.path.join(data_dir, 'train.pkl'))
test_df = pd.read_pickle(os.path.join(big_data_dir, 'test.pkl'))

if debug:
    all_df = all_df.sample(debug_num)


logging.info(f"Creating item features")
fc = item_feature_creator(train_df=all_df, test_df=test_df, feature_path=target_raw_data_dir)
item_features, dest_features = fc.get_features()
logging.info(f"Item features: {item_features.columns}")

logging.info(item_features.head())
fc.save_features()

logging.info(f"Data shape: {all_df.shape}; Creating all Features...")
fc = feature_creator_class(feature_cols=dense_features+sparse_features,
                           item_feature_class=item_feature_creator)

train_eval, _ = fc.get_features(df=all_df, task='train_eval')


train_eval[feature_cols]

assert 'srch_id' in train_eval.columns
assert train_eval.shape[0] == all_df.shape[0]

search_id_df = pd.DataFrame({'srch_id': pd.unique(train_eval['srch_id'])})
train_eval_srch_id, test_srch_id = train_test_split(search_id_df, test_size=0.2)
train_srch_id, eval_srch_id = train_test_split(train_eval_srch_id, test_size=0.2)

assert len(train_srch_id)+len(eval_srch_id)+len(test_srch_id) == len(search_id_df)

srch_id_dfs = [train_srch_id, eval_srch_id, test_srch_id]
file_names = ['train_df', 'eval_df', 'test_df']
logging.info(f"Saving file to {target_raw_data_dir}")
for srch_id_df, file_name in tqdm(zip(srch_id_dfs, file_names)):
    df = srch_id_df.merge(train_eval, how='left', on='srch_id')
    print(df.shape)
    print(df.sample(5))
    df.to_pickle(os.path.join(target_raw_data_dir, file_name+'.pkl'))
