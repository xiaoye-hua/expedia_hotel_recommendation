# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :
import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from scripts.train_config import train_config_detail, dir_mark, data_dir, debug, debug_num, model_dir, raw_data_path
from src.config import regression_label, submission_cols, position_feature_path
from scripts.train_config import no_test
from src.Evaluation import get_ndcg

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
if position_feature_included:
    assert position_feature_cols is not None
    position_df = pd.read_csv(position_feature_path)
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
# logging.info(f"Reading data from {target_raw_data_dir}")
# train_df = pd.read_csv(os.path.join(target_raw_data_dir, 'train.csv'))
# eval_df = pd.read_csv(os.path.join(target_raw_data_dir, 'eval.csv'))
# test_df = pd.read_csv(os.path.join(target_raw_data_dir, 'test.csv'))
logging.info(f"Reading data from {target_raw_data_dir}")

train_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'train_df.pkl'))
eval_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'eval_df.pkl'))
test_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'test_df.pkl'))





if no_test:
    train_df = pd.concat([train_df, eval_df], axis=0)
    eval_df = test_df
    del test_df
    df_for_encode_train = pd.concat([train_df, eval_df], axis=0)
else:
    df_for_encode_train = pd.concat([train_df, eval_df, test_df], axis=0)
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
# logging.info(f"    All Features...")
# fc = feature_creator_class(feature_cols=dense_features+sparse_features)

# train, train_feature = fc.get_features(df=train_df)
# eval, eval_feature = fc.get_features(df=eval_df)

# assert train_feature == eval_feature
# if not no_test:
#     test, test_feature = fc.get_features(df=test_df)
#     assert train_feature == test_feature
#     del test_df
#
# del train_df
# del eval_df
if no_test:
    logging.info(f"train dim：{train_df.shape}; evla dim：{eval_df.shape}")
else:
    logging.info(f"train dim：{train_df.shape}; evla dim：{eval_df.shape}; Test data shape : {test_df.shape}")

logging.info('Sort train_df, eval_df with search_id)')
train_df = train_df.sort_values('srch_id')
eval_df = eval_df.sort_values('srch_id')

train_group = train_df.groupby('srch_id')['srch_id'].count().values.tolist()
eval_group = eval_df.groupby('srch_id')['srch_id'].count().values.tolist()

train_params = {
    # 'df_for_encode_train': raw_df
    'train_valid': train_valid
    , "df_for_encode_train": df_for_encode_train[feature_cols].copy()
    , 'category_features': [] #sparse_features
    , 'eval_X': eval_df[feature_cols].copy()
    , 'eval_y': eval_df[regression_label].copy()
    , 'train_group':train_group
    , 'eval_group': eval_group
}


# Dnn
# train_params = {
#     'epoch': 3
#     , 'batch_size': 512
#     , 'pca_component_num': pca_component_num
#     , "df_for_encode_train": raw_df
#     , 'train_valid': (eval_features[feature_cols], eval_features[['is_clicked']])
#
# }

print(feature_cols)
logging.info(f"Model training...")
pipeline = pipeline_class(model_path=model_path, model_training=True, model_params=model_params)
# logging.info(f"Train data shape : {train_features.shape}")
pipeline.train(X=train_df[feature_cols], y=train_df[regression_label], train_params=train_params)
# logging.info(f"Test feature creating..")
# test_features, feature_cols = fc.get_features(df=test_df)
# test_features = test_features.merge(label, how='left', left_index=True, right_index=True)
#
if not no_test:
    logging.info(f"Model testing...")
    pipeline.eval(X=test_df[feature_cols], y=test_df[regression_label])
logging.info(f"Model saving to {model_path}..")
pipeline.save_pipeline()


def change_position_features(df):
    df = df.drop(position_feature_cols, axis=1)
    df.loc[:, 'position'] = 1
    df = df.merge(position_df, how='left', on='position')
    return df

if position_feature_included:
    eval_df = change_position_features(eval_df)
    train_df = change_position_features(train_df)
    test_df = change_position_features(test_df)

    for df in [eval_df, train_df, test_df]:
        print(pd.unique(df['position']))
        assert len(pd.unique(df['position'])) == 1


eval_df['predicted'] = pipeline.predict(eval_df[feature_cols])
train_df['predicted'] = pipeline.predict(train_df[feature_cols])
test_df['predicted'] = pipeline.predict(test_df[feature_cols])

train_ndcg = get_ndcg(train_df)
eval_ndcg = get_ndcg(eval_df)
test_ndcg = get_ndcg(test_df)
print(f"{train_df.shape}; {eval_df.shape}; {test_df.shape}")
print(f"{train_ndcg}; {eval_ndcg}; {test_ndcg}")

