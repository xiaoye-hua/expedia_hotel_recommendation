# -*- coding: utf-8 -*-
# @File    : model_batch_inference.py
# @Author  : Hua Guo
# @Time    : 2021/11/3 下午11:00
# @Disc    :
import pandas as pd
import os
import logging

from src.FeatureCreator.FeatureCreator import FeatureCreator
from src.FeatureCreator.UserFeatureCreator import UserFeatureCreator
from src.FeatureCreator.ItemFeatureCreator import ItemFeatureCreator
from src.Pipeline.XGBoostPipeline import XGBoostPipeline
from src.Pipeline.DNNPipeline import DNNPipeline
from scripts.train_config import raw_data_path, debug, debug_data_path
from src.config import raw_data_usecols
from src.tmp import data_preprocess

# ============== Config ============
# model_path = 'model_finished/v6_xgb_1104'
# pipeline_class = XGBoostPipeline

# DNN model
model_path = 'model_finished/v7_DNN_1104'
pipeline_class = DNNPipeline

res_path = 'logs/res.csv'
feature_creator_class = FeatureCreator
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',)

# ==================================================================

if debug:
    raw_data_path = debug_data_path
logging.info(f'Loading model from {model_path}...')
pipeline = pipeline_class(model_path=model_path, model_training=False, model_params={})
pipeline.load_model()
logging.info(f'Loading offline user & item features from {model_path}...')
user_fc = UserFeatureCreator(df=None, feature_path=model_path)
user_features, user_feature_cols = user_fc.get_features()
item_fc = ItemFeatureCreator(df=None, feature_path=model_path)
item_features, item_feature_cols, channel_features = item_fc.get_features()
fc = feature_creator_class(user_feature_cols_tuple=(user_features, user_feature_cols)
                           , item_feature_cols_tuple=(item_features, item_feature_cols, channel_features))
logging.info(f"Reading raw data from {raw_data_path}")
raw_df = pd.read_csv(raw_data_path, usecols=raw_data_usecols)
raw_df = raw_df.set_index('row_id')
label = raw_df[['is_clicked']]
raw_df = data_preprocess(raw_df=raw_df)
logging.info(f"Creating features...")
features, feature_cols = fc.get_features(df=raw_df, task='inference')
features = features.merge(label, how='left', left_index=True, right_index=True)
logging.info(f"Model batch inference...")
res = pipeline.predict(X=features[feature_cols])
logging.info(f"Res saving to {res_path}..")
raw_df['predict_prob'] = res
raw_df.to_csv(res_path, index=False)

