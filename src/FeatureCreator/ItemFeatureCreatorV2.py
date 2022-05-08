# -*- coding: utf-8 -*-
# @File    : ItemFeatureCreatorV2.py
# @Author  : Hua Guo
# @Time    : 2021/11/3 上午11:59
# @Disc    :
import pandas as pd
from typing import Tuple, List, Optional, Union
import numpy as np
import os
from src.config import search_id, prop_id, original_dense_features, dest_id, dest_feature_prefix

from src.FeatureCreator import BaseFeatureCreator
from src.config import item_feature_file, dest_feature_file, item_feature_prefix, listwise_feature_prefix, listwise_feature_file, offline_feature_path


class ItemFeatureCreatorV2(BaseFeatureCreator):
    def __init__(self, train_df=None, test_df=None, feature_path=offline_feature_path) -> None:
        super(ItemFeatureCreatorV2, self).__init__()
        self._check_dir(feature_path)
        self.feature_dir = feature_path
        self.item_feature_path = os.path.join(self.feature_dir, item_feature_file)
        self.dest_feature_path = os.path.join(self.feature_dir, dest_feature_file)

        # self.item_col = item_col
        # self.feature_cols = []
        # self.target_col = 'is_clicked'
        # self.item_feature = None
        self.col_prefix = item_feature_prefix
        self.dest_prefix = dest_feature_prefix

        if train_df is None:
            self.train_test = None
            self.item_feature = None
            self.dest_feature = None

        else:
            self.train_test = pd.concat([train_df[test_df.columns], test_df], axis=0)
            self.train = train_df
            self.test = test_df
            self.item_feature = self.train_test.groupby(prop_id).size().reset_index().rename(columns={0: self.col_prefix + 'size'})
            self.dest_feature = self.train_test.groupby(dest_id).size().reset_index().rename(columns={0: self.dest_prefix + 'size'})
            # self._get_channel_feature()
            # self.df = self.df.merge(self.channel_ctr, how='left', on='channel')

    # def _get_channel_feature(self):
    #     self.channel_ctr = self.df.groupby(['channel'])[self.target_col].mean().reset_index().rename(
    #         columns={self.target_col: 'channel_ctr'})
    #     channel_ctr_mean = self.channel_ctr['channel_ctr'].mean()
    #     self.channel_ctr['channel_ctr'] = self.channel_ctr['channel_ctr'] + channel_ctr_mean

    # def _get_exposure_click_ctr_with_position_bias(self):
    #     # feature_df = self.df.groupby(self.item_col)[[self.item_col, self.target_col]].mean().reset_index()
    #     map_func = [np.sum, np.size,
    #                 np.mean
    #                 ]
    #     map_col_name = ['sum', 'size',
    #                     'mean'
    #                     ]
    #     target_col_name = [self.col_prefix+ 'click_num', self.col_prefix+'exposure_num',
    #                        self.col_prefix+'ctr_with_pb'
    #                        ]
    #     feature_df = self.df.groupby(self.item_col).agg({
    #         self.target_col: map_func
    #     }).reset_index().set_index(self.item_col)[self.target_col].rename(columns=dict(zip(map_col_name, target_col_name))).reset_index()
    #     self.feature_cols.extend(target_col_name)
    #     self._merge_feature_df(feature_df=feature_df)
    #
    # def _get_ctr_without_position_bias(self):
    #     target_col = self.col_prefix+'ctr_without_pb'
    #     self.df['is_clicked_without_pb'] = self.df[self.target_col]/self.df['channel_ctr']
    #     feature_df = self.df.groupby([self.item_col])[self.target_col].mean().reset_index().rename(columns={self.target_col: target_col})
    #     assert feature_df.shape[0] == self.item_feature.shape[0], f"{feature_df.shape}, { self.item_feature.shape}"
    #     self.feature_cols.append(target_col)
    #     self._merge_feature_df(feature_df=feature_df[[self.item_col, target_col]])
    #
    # def _get_item_cross_features(self):
    #     cols = [ 'device_model' 'site_category', 'device_conn_type', 'app_id', 'app_category',
    #              'device_ip', 'device_id', 'device_type', 'P1', "P2", "P3", "P4"]
    #     targer_cols = [self.col_prefix+col+'_nunique' for col in cols]
    #     feature_df = self.df.groupby(self.item_col)[cols].nunique().reset_index().rename(columns=dict(zip(cols, targer_cols)))
    #     assert feature_df.shape[0]  == self.item_feature.shape[0]
    #     self.feature_cols.extend(targer_cols)
    #     self._merge_feature_df(feature_df=feature_df)

    def _get_item_target_features(self):
        item_id = prop_id
        prefix = self.col_prefix
        # rename_map = {
        #     'click_bool': self.col_prefix + 'ctr'
        #     , 'booking_bool': self.col_prefix + 'ctcvr'
        # }
        rename_map2 = {
            "click_bool": prefix + 'click_cnt'
        }
        rename_map3 = {
            "booking_bool": prefix + 'booking_cnt'
        }
        rename_map4 = {
            "booking_bool": prefix + 'total_cnt'
        }
        click_count = self.train.groupby(item_id)['click_bool'].sum().reset_index().rename(columns=rename_map2)
        booking_count = self.train.groupby(item_id)['booking_bool'].sum().reset_index().rename(columns=rename_map3)
        total_count = self.train.groupby(item_id)['booking_bool'].count().reset_index().rename(columns=rename_map4)
        for df in [click_count, booking_count, total_count]:
            self.item_feature = self.item_feature.merge(df,how='left', on=item_id)
        self.item_feature[self.col_prefix + 'all_ctr'] = self.train['click_bool'].mean()
        self.item_feature[self.col_prefix + 'all_ctcvr'] = self.train['booking_bool'].mean()


    def _get_item_statistic_features(self):
        func_lst = [np.mean, np.std, np.median]
        stats_cols = ['mean', 'std', 'median']
        assert len(stats_cols) == len(func_lst)
        for col in original_dense_features:
            target_cols = [self.col_prefix + fun + '_' + col for fun in stats_cols]
            self.sample_df = self.train_test.groupby(prop_id).agg({col: func_lst})[col].reset_index()\
                .rename(columns=dict(zip(stats_cols, target_cols)))
            self.item_feature = self.item_feature.merge(self.sample_df, how='left', left_on=prop_id, right_on=prop_id)

    def _get_dest_target_features(self):
        item_id = dest_id
        prefix = self.dest_prefix

        rename_map2 = {
            "click_bool": prefix + 'click_cnt'
        }
        rename_map3 = {
            "booking_bool": prefix + 'booking_cnt'
        }
        rename_map4 = {
            "booking_bool": prefix + 'total_cnt'
        }
        click_count = self.train.groupby(item_id)['click_bool'].sum().reset_index().rename(columns=rename_map2)
        booking_count = self.train.groupby(item_id)['booking_bool'].sum().reset_index().rename(columns=rename_map3)
        total_count = self.train.groupby(item_id)['booking_bool'].count().reset_index().rename(columns=rename_map4)
        for df in [click_count, booking_count, total_count]:
            self.dest_feature = self.dest_feature.merge(df,how='left', on=item_id)
        self.dest_feature[prefix + 'all_ctr'] = self.train['click_bool'].mean()
        self.dest_feature[prefix + 'all_ctcvr'] = self.train['booking_bool'].mean()

    def _get_dest_statistic_features(self):
        item_id = dest_id
        func_lst = [np.mean, np.std, np.median]
        stats_cols = ['mean', 'std', 'median']
        assert len(stats_cols) == len(func_lst)
        for col in original_dense_features:
            target_cols = [self.dest_prefix +fun + "_" + col for fun in stats_cols]
            self.sample_df = self.train_test.groupby(item_id).agg({col: func_lst})[col].reset_index()\
                .rename(columns=dict(zip(stats_cols, target_cols)))
            self.dest_feature = self.dest_feature.merge(self.sample_df, how='left', left_on=item_id, right_on=item_id)

    def get_features(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if self.train_test is None:
            self.item_feature = pd.read_csv(self.item_feature_path)
            self.dest_feature = pd.read_csv(self.dest_feature_path)
        else:
            funcs = [
                self._get_item_statistic_features
                , self._get_item_target_features
                , self._get_dest_statistic_features
                , self._get_dest_target_features
            ]
            for func in funcs:
                func()
        return self.item_feature, self.dest_feature

    def save_features(self):
        print(f"Item features saving to {self.item_feature_path}")
        self.item_feature.to_csv(self.item_feature_path, index=False)
        print(f"dest feature saving to {self.dest_feature_path}")
        self.dest_feature.to_csv(self.dest_feature_path, index=False)