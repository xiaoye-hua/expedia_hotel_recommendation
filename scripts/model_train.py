# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
import logging
from sklearn.model_selection import train_test_split
from scripts.train_config import train_config_detail, dir_mark, data_dir, debug, debug_num, model_dir, raw_data_path
from src.config import regression_label, submission_cols, position_feature_path
from scripts.train_config import no_test
from src.Evaluation import get_ndcg
from src.config import log_dir
from src.DataProfiling import DataProfiling

# =============== Config ============
curDT = datetime.now()
date_time = curDT.strftime("%m%d%H")
current_file = os.path.basename(__file__).split('.')[0]
log_file = '_'.join([dir_mark, current_file, date_time, '.log'])
logging.basicConfig(level='INFO',
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=os.path.join(log_dir, log_file)
                    )
console = logging.StreamHandler()
logging.getLogger().addHandler(console)

data_profiling = True
# target_col = train_config_detail[dir_mark]['target_col']
target_col = regression_label
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
epochs = train_config_detail[dir_mark].get('epochs', None)
batch_size = train_config_detail[dir_mark].get('batch_size', None)
dense_to_sparse = train_config_detail[dir_mark].get('dense_to_sparse', None)
task = train_config_detail[dir_mark].get('task', None) # params for deepFM
fillna = train_config_detail[dir_mark].get('fillna', False)


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
# print(f"Reading data from {target_raw_data_dir}")
# train_df = pd.read_csv(os.path.join(target_raw_data_dir, 'train.csv'))
# eval_df = pd.read_csv(os.path.join(target_raw_data_dir, 'eval.csv'))
# test_df = pd.read_csv(os.path.join(target_raw_data_dir, 'test.csv'))
print(f"Reading data from {target_raw_data_dir}")

train_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'train_df.pkl'))
eval_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'eval_df.pkl'))
test_df = pd.read_pickle(os.path.join(target_raw_data_dir, 'test_df.pkl'))


if feature_clean_func is not None:
    train_df = feature_clean_func(df=train_df)
    eval_df = feature_clean_func(df=eval_df)
    test_df = feature_clean_func(df=test_df)


if no_test:
    train_df = pd.concat([train_df, eval_df], axis=0)
    eval_df = test_df
    del test_df
    df_for_encode_train = pd.concat([train_df, eval_df], axis=0)
else:
    df_for_encode_train = pd.concat([train_df, eval_df, test_df], axis=0)

print(train_df[feature_cols].isna().sum())
if fillna:
    for col in tqdm(feature_cols):
        train_df[col] = train_df[col].fillna(df_for_encode_train[col].max())
        test_df[col] = test_df[col].fillna(df_for_encode_train[col].max())
        eval_df[col] = eval_df[col].fillna(df_for_encode_train[col].max())
print(train_df[feature_cols].isna().sum())
df_for_encode_train = pd.concat([train_df, eval_df, test_df], axis=0)

# ===================================

#
# raw_df = pd.read_csv(raw_data_path, usecols=raw_data_usecols)

# raw_df = raw_df.set_index('row_id')
# label = raw_df[['is_clicked']]
#
# print(f"Data preprocessing")
# raw_df = data_preprocess(raw_df=raw_df)
# print(f"Train & test Spliting..")
# train_eval_df, test_df = train_test_split(raw_df, test_size=0.2)


# print(f"Creating features...")
# print(f"    User Features...")
# user_fc = UserFeatureCreator(df=train_df, feature_path=model_path)
# user_features, user_feature_cols = user_fc.get_features()
# user_fc.save_features()
# print(f"   Item Features... ")
# item_fc = ItemFeatureCreator(df=train_df, feature_path=model_path)
# item_features, item_feature_cols, channel_features = item_fc.get_features()
# item_fc.save_features()
# print(f"    All Features...")
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
    print(f"train dim：{train_df.shape}; evla dim：{eval_df.shape}")
else:
    print(f"train dim：{train_df.shape}; evla dim：{eval_df.shape}; Test data shape : {test_df.shape}")

print('Sort train_df, eval_df with search_id)')
train_df = train_df.sort_values('srch_id')
eval_df = eval_df.sort_values('srch_id')

train_group = train_df.groupby('srch_id')['srch_id'].count().values.tolist()
eval_group = eval_df.groupby('srch_id')['srch_id'].count().values.tolist()

train_params = {
    'epoches': epochs
    , 'batch_size': batch_size
    , 'dense_to_sparse': dense_to_sparse
    , 'train_valid': train_valid
    , "df_for_encode_train": df_for_encode_train[feature_cols].copy()
    , 'category_features': [] #sparse_features
    , 'eval_X': eval_df[feature_cols].copy()
    , 'eval_y': eval_df[regression_label].copy()
    , 'train_group':train_group
    , 'eval_group': eval_group
    , 'sparse_features': sparse_features
    , 'dense_features': dense_features
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


if data_profiling:
    profile_cols = feature_cols + [regression_label]
    profile_tool = DataProfiling(data_dir=os.path.join(model_path, 'profiling'))
    train_profile = profile_tool.profiling_save(df=train_df[profile_cols], file_name='train.html')
    test_profile = profile_tool.profiling_save(df=test_df[profile_cols], file_name='test.html')
    eval_profile = profile_tool.profiling_save(df=eval_df[profile_cols], file_name='eval.html')
    # profile_tool.compare_save(profile1=train_profile, profile2=test_profile, file_name='compare_train_test.html')
    # profile_tool.compare_save(profile1=train_profile, profile2=eval_profile, file_name='compare_train_eval.html')
print(feature_cols)
print(f"Model training...")
pipeline = pipeline_class(model_path=model_path, model_training=True, model_params=model_params, task=task)
# print(f"Train data shape : {train_features.shape}")
pipeline.train(X=train_df[feature_cols], y=train_df[regression_label], train_params=train_params)
# print(f"Test feature creating..")
# test_features, feature_cols = fc.get_features(df=test_df)
# test_features = test_features.merge(label, how='left', left_index=True, right_index=True)
#
if not no_test:
    print(f"Model testing...")
    pipeline.eval(X=test_df[feature_cols], y=test_df[regression_label])
print(f"Model saving to {model_path}..")
pipeline.save_pipeline()

# position_feature_included = Fa
def get_res(train_df, eval_df, test_df, target_position=1):
    def change_position_features(df, target_position=1):
        df = df.drop(position_feature_cols, axis=1)
        df.loc[:, 'position'] = target_position
        # df = df.merge(position_df, how='left', on='position')
        return df
    if target_position is not None:
        logging.info(f"Changing position feature to {target_position}")
        eval_df = change_position_features(eval_df, target_position=target_position)
        train_df = change_position_features(train_df, target_position=target_position)
        test_df = change_position_features(test_df, target_position=target_position)

        for df in [eval_df, train_df, test_df]:
            print(pd.unique(df['position']))
            assert len(pd.unique(df['position'])) == 1
    else:
        logging.info(f"Without changing position feature")
    eval_df['predicted'] = pipeline.predict(eval_df[feature_cols])
    train_df['predicted'] = pipeline.predict(train_df[feature_cols])
    test_df['predicted'] = pipeline.predict(test_df[feature_cols])
    print(f"{train_df['predicted'].mean()}; {eval_df['predicted'].mean()}; {test_df['predicted'].mean()}")

    train_ndcg = get_ndcg(train_df)
    eval_ndcg = get_ndcg(eval_df)
    test_ndcg = get_ndcg(test_df)
    print(f"{train_df[feature_cols].shape}; {eval_df[feature_cols].shape}; {test_df[feature_cols].shape}")
    print(f"{train_ndcg}; {eval_ndcg}; {test_ndcg}")


get_res(train_df=train_df, eval_df=eval_df, test_df=test_df, target_position=None)
if position_feature_included:
    for position in range(1, 3):
        print(f"position: {position}")
        get_res(train_df=train_df, eval_df=eval_df, test_df=test_df, target_position=position)


print('#'*10)
print('search list where previous method performed badly')
print('*'*10)

def get_poor_df(df):
    test_df = df.copy()
    max_label_df = test_df.groupby('srch_id')['label'].max().reset_index().rename(columns={'label': 'max_label'})
    current_label_df = test_df[test_df['position']==1][['srch_id', 'label']].rename(columns={'label': 'current_label'})
    merged_df = current_label_df.merge(max_label_df, how='left', on='srch_id')
    poor_srch_df = merged_df[merged_df['current_label']!=merged_df['max_label']]#.shape
    poor_srch_id = poor_srch_df['srch_id'].unique()
    poor_df = test_df[test_df['srch_id'].isin(poor_srch_id)]
    print(f"Item number: previous: {test_df.shape}; current: {poor_df.shape}")
    print(f"search list number: Previous: {len(test_df['srch_id'].unique())}; current: {len(poor_df['srch_id'].unique())}")
    return poor_df

poor_train = get_poor_df(df=train_df)
poor_eval = get_poor_df(df=eval_df)
poor_test = get_poor_df(df=test_df)

get_res(train_df=poor_train, eval_df=poor_eval, test_df=poor_test, target_position=None)
if position_feature_included:
    for position in range(1, 3):
        print(f"position: {position}")
        get_res(train_df=poor_train, eval_df=poor_eval, test_df=poor_test, target_position=position)