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
if debug:
    all_df = all_df.sample(debug_num)

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

#
# del train_eval_df
#
# logging.info(f"train evla dim：{train_eval.shape};")
#
#
# train, eval = train_test_split(train_eval, test_size=0.1)
# train_params = {
#     # 'df_for_encode_train': raw_df
#     'train_valid': train_valid
#     , "df_for_encode_train": train_eval[feature_cols].copy()
#     , 'category_features': sparse_features
# }
# del train_eval
#
#
# # Dnn
# # train_params = {
# #     'epoch': 3
# #     , 'batch_size': 512
# #     , 'pca_component_num': pca_component_num
# #     , "df_for_encode_train": raw_df
# #     , 'train_valid': (eval_features[feature_cols], eval_features[['is_clicked']])
# #
# # }
#
#
# logging.info(f"Model training...")
# pipeline = pipeline_class(model_path=model_path, model_training=True, model_params=model_params)
# # logging.info(f"Train data shape : {train_features.shape}")
# pipeline.train(X=train[feature_cols], y=train[regression_label], train_params=train_params)
# # logging.info(f"Test feature creating..")
# # test_features, feature_cols = fc.get_features(df=test_df)
# # test_features = test_features.merge(label, how='left', left_index=True, right_index=True)
# #
# logging.info(f"Model testing...")
# logging.info(f"Test data shape : {eval.shape}")
# pipeline.eval(X=eval[feature_cols], y=eval[regression_label])
# logging.info(f"Model saving to {model_path}..")
# pipeline.save_pipeline()
#
#
