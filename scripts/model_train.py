# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :
import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split
from src.save_submission import save_submission

# from src.FeatureCreator.FeatureCreator import FeatureCreator
# from src.FeatureCreator.UserFeatureCreator import UserFeatureCreator
# from src.FeatureCreator.ItemFeatureCreator import ItemFeatureCreator
# from src.Pipeline.XGBoostPipeline import XGBoostPipeline
# from src.Pipeline.DNNPipeline import DNNPipeline
# from scripts.train_config import raw_data_path, debug, debug_data_path
# from src.config import raw_data_usecols, regression_label
# from src.tmp import data_preprocess
from scripts.train_config import train_config_detail, dir_mark, data_dir, debug, debug_num, model_dir
from src.config import regression_label, submission_cols

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


additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

# target_col = train_config_detail[dir_mark].get('target_col', reg_target_col)
# feature_used = dense_features + sparse_features
# # assert feature_used is not None
# if not train_config_detail[dir_mark].get('data_dir_mark', False):
#     target_raw_data_dir = os.path.join(raw_data_path, dir_mark)
# else:
#     target_raw_data_dir = os.path.join(raw_data_path, train_config_detail[dir_mark].get('data_dir_mark', False))
# logging.info(f"Reading data from {target_raw_data_dir}")
# train_df = pd.read_csv(os.path.join(target_raw_data_dir, 'train.csv'))
# eval_df = pd.read_csv(os.path.join(target_raw_data_dir, 'eval.csv'))
# test_df = pd.read_csv(os.path.join(target_raw_data_dir, 'test.csv'))
logging.info(f"Reading data from {data_dir}")
if debug:
    train_eval_df = pd.read_csv(os.path.join(data_dir, 'train.csv'), nrows=debug_num)
else:
    train_eval_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
# ===================================

#
# raw_df = pd.read_csv(raw_data_path, usecols=raw_data_usecols)

# raw_df = raw_df.set_index('row_id')
# label = raw_df[['is_clicked']]
#
# logging.info(f"Data preprocessing")
# raw_df = data_preprocess(raw_df=raw_df)
# logging.info(f"Train & test Spliting..")
# train_eval_df, test_df = train_test_split(raw_df, test_size=0.2)


# logging.info(f"Creating features...")
# logging.info(f"    User Features...")
# user_fc = UserFeatureCreator(df=train_df, feature_path=model_path)
# user_features, user_feature_cols = user_fc.get_features()
# user_fc.save_features()
# logging.info(f"   Item Features... ")
# item_fc = ItemFeatureCreator(df=train_df, feature_path=model_path)
# item_features, item_feature_cols, channel_features = item_fc.get_features()
# item_fc.save_features()
logging.info(f"    All Features...")
fc = feature_creator_class(feature_cols=dense_features+sparse_features)

train_eval, feature_cols = fc.get_features(df=train_eval_df)

del train_eval_df

logging.info(f"train evla dim：{train_eval.shape};")


train, eval = train_test_split(train_eval, test_size=0.1)
train_params = {
    # 'df_for_encode_train': raw_df
    'train_valid': train_valid
    , "df_for_encode_train": train_eval[feature_cols].copy()
    , 'category_features': sparse_features
}
del train_eval


# Dnn
# train_params = {
#     'epoch': 3
#     , 'batch_size': 512
#     , 'pca_component_num': pca_component_num
#     , "df_for_encode_train": raw_df
#     , 'train_valid': (eval_features[feature_cols], eval_features[['is_clicked']])
#
# }


logging.info(f"Model training...")
pipeline = pipeline_class(model_path=model_path, model_training=True, model_params=model_params)
# logging.info(f"Train data shape : {train_features.shape}")
pipeline.train(X=train[feature_cols], y=train[regression_label], train_params=train_params)
# logging.info(f"Test feature creating..")
# test_features, feature_cols = fc.get_features(df=test_df)
# test_features = test_features.merge(label, how='left', left_index=True, right_index=True)
#
logging.info(f"Model testing...")
logging.info(f"Test data shape : {eval.shape}")
pipeline.eval(X=eval[feature_cols], y=eval[regression_label])
logging.info(f"Model saving to {model_path}..")
pipeline.save_pipeline()


