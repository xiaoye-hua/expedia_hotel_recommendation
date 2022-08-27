# -*- coding: utf-8 -*-
# @File    : model_train.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 下午3:47
# @Disc    :
import pandas as pd
import os
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from src.save_submission import save_submission
from scripts.train_config import train_config_detail, dir_mark, data_dir, debug, debug_num, model_dir, big_data_dir
from src.config import regression_label, submission_cols, log_dir

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

additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

model_path = os.path.join(model_dir, dir_mark)
item_feature_creator = train_config_detail[dir_mark].get('item_feature_creator', None)

additional_train_params = train_config_detail[dir_mark].get('additional_train_params', {})

logging.info(f"Reading data from {data_dir}")

test_df = pd.read_pickle(os.path.join(big_data_dir, 'test.pkl'))

if debug:
    test_df = test_df.sample(debug_num)

logging.info(f"    All Features...")
fc = feature_creator_class(feature_cols=dense_features+sparse_features, item_feature_class=item_feature_creator)
test, _ = fc.get_features(df=test_df, task='inference')

print(feature_cols)

del test_df

logging.info(f"test dim: {test.shape}")

logging.info(f"Model predicting...")
pipeline = pipeline_class(model_path=model_path, model_training=False, model_params=model_params)

test['predicted'] = pipeline.predict(test[feature_cols])

if debug:
    file_name = 'debug_'+dir_mark+'.csv'
else:
    file_name = dir_mark + '.csv'
logging.info(f"saving to {file_name}")
save_submission(rec_df=test[submission_cols + ['predicted']], file_name=file_name)

