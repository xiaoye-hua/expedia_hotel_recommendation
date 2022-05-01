# -*- coding: utf-8 -*-
# @File    : ItemFeatureCreator.py
# @Author  : Hua Guo
# @Time    : 2021/11/3 上午11:59
# @Disc    :
import pandas as pd
from typing import Tuple, List, Optional, Union
import numpy as np
import os

from src.FeatureCreator import BaseFeatureCreator
from src.config import item_feature_file, item_feature_prefix, channel_feature_file, item_col


class ItemFeatureCreator(BaseFeatureCreator):
    def __init__(self, df: Union[pd.DataFrame, None], feature_path:str) -> None:
        super(ItemFeatureCreator, self).__init__()
        self._check_dir(feature_path)
        self.feature_dir = feature_path
        self.feature_path = os.path.join(self.feature_dir, item_feature_file)
        self.item_col = item_col
        self.feature_cols = []
        self.target_col = 'is_clicked'
        self.item_feature = None
        self.col_prefix = item_feature_prefix
        if df is None:
            self.df = None
            self.channel_ctr = None
        else:
            self.df = df
            self._get_channel_feature()
            self.df = self.df.merge(self.channel_ctr, how='left', on='channel')

    def _get_channel_feature(self):
        self.channel_ctr = self.df.groupby(['channel'])[self.target_col].mean().reset_index().rename(
            columns={self.target_col: 'channel_ctr'})
        channel_ctr_mean = self.channel_ctr['channel_ctr'].mean()
        self.channel_ctr['channel_ctr'] = self.channel_ctr['channel_ctr'] + channel_ctr_mean

    def _get_exposure_click_ctr_with_position_bias(self):
        # feature_df = self.df.groupby(self.item_col)[[self.item_col, self.target_col]].mean().reset_index()
        map_func = [np.sum, np.size,
                    np.mean
                    ]
        map_col_name = ['sum', 'size',
                        'mean'
                        ]
        target_col_name = [self.col_prefix+ 'click_num', self.col_prefix+'exposure_num',
                           self.col_prefix+'ctr_with_pb'
                           ]
        feature_df = self.df.groupby(self.item_col).agg({
            self.target_col: map_func
        }).reset_index().set_index(self.item_col)[self.target_col].rename(columns=dict(zip(map_col_name, target_col_name))).reset_index()
        self.feature_cols.extend(target_col_name)
        self._merge_feature_df(feature_df=feature_df)

    def _get_ctr_without_position_bias(self):
        target_col = self.col_prefix+'ctr_without_pb'
        self.df['is_clicked_without_pb'] = self.df[self.target_col]/self.df['channel_ctr']
        feature_df = self.df.groupby([self.item_col])[self.target_col].mean().reset_index().rename(columns={self.target_col: target_col})
        assert feature_df.shape[0] == self.item_feature.shape[0], f"{feature_df.shape}, { self.item_feature.shape}"
        self.feature_cols.append(target_col)
        self._merge_feature_df(feature_df=feature_df[[self.item_col, target_col]])

    def _get_item_cross_features(self):
        cols = [ 'device_model' 'site_category', 'device_conn_type', 'app_id', 'app_category',
                 'device_ip', 'device_id', 'device_type', 'P1', "P2", "P3", "P4"]
        targer_cols = [self.col_prefix+col+'_nunique' for col in cols]
        feature_df = self.df.groupby(self.item_col)[cols].nunique().reset_index().rename(columns=dict(zip(cols, targer_cols)))
        assert feature_df.shape[0]  == self.item_feature.shape[0]
        self.feature_cols.extend(targer_cols)
        self._merge_feature_df(feature_df=feature_df)

    def get_features(self, **kwargs) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
        if self.df is None:
            self.item_feature = pd.read_csv(self.feature_path)
            self.channel_ctr = pd.read_csv(os.path.join(self.feature_dir, channel_feature_file))
            self.feature_cols = list(self.item_feature.columns)
            self.feature_cols.remove(self.item_col)
        else:
            self.item_feature = self.df[[self.item_col]].drop_duplicates()
            funcs = [
                self._get_exposure_click_ctr_with_position_bias
                , self._get_ctr_without_position_bias
            ]
            for func in funcs:
                func()
        return self.item_feature, self.feature_cols, self.channel_ctr

    def save_features(self):
        self.channel_ctr.to_csv(os.path.join(self.feature_dir, channel_feature_file), index=False)
        self.item_feature.to_csv(self.feature_path, index=False)

    def _merge_feature_df(self, feature_df: pd.DataFrame) -> None:
        self.item_feature = self.item_feature.merge(feature_df, how='left', on=self.item_col)