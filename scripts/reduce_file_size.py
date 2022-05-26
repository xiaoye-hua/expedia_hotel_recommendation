# -*- coding: utf-8 -*-
# @File    : reduce_file_size.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import numpy as np
import os
from src.utils import check_create_dir

from src.utils.memory_utils import reduce_mem_usage
from scripts.train_config import big_data_dir, small_data_dir
train_path = 'raw_data/train.csv'
test_path = 'raw_data/test.csv'
print(f"Reading data from {train_path}; {test_path}")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

print("Reducing memory...")
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

check_create_dir(big_data_dir)
check_create_dir(small_data_dir)
print(f"train: {train.shape}; test: {test.shape}")
train.to_pickle(os.path.join(big_data_dir, 'train.pkl'))
test.to_pickle(os.path.join(big_data_dir, 'test.pkl'))

search_id_df = pd.DataFrame({'srch_id': pd.unique(train['srch_id'])})
srch_len = int(search_id_df.shape[0]/10)
search_id_df = search_id_df.sample(srch_len)
new_train = search_id_df.merge(train, how='left', on='srch_id')
print(f"new_train: {new_train.shape}")
new_train.to_pickle(os.path.join(small_data_dir, 'train.pkl'))




