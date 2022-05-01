# -*- coding: utf-8 -*-
# @File    : UserFeatureCreator.py
# @Author  : Hua Guo
# @Time    : 2021/11/1 下午5:10
# @Disc    :
import pandas as pd
from typing import Tuple, List, Optional, Union
import numpy as np
import os

from src.FeatureCreator import BaseFeatureCreator
from src.config import user_feature_prefix, user_most_click_col_prex, user_feature_file


class UserFeatureCreator(BaseFeatureCreator):
    def __init__(self, df: Union[pd.DataFrame, None], feature_path) -> None:
        super(UserFeatureCreator, self).__init__()
        self._check_dir(feature_path)
        self.feature_path = os.path.join(feature_path, user_feature_file)
        self.uid_col = 'device_model'
        self.feature_cols = []
        self.target_col = 'is_clicked'
        self.uid_feature = None
        self.col_prefix = user_feature_prefix
        if df is None:
            self.df = None
        else:
            self.df = df
            self.channel_ctr = df.groupby(['channel'])[self.target_col].mean().reset_index().rename(columns={self.target_col: 'channel_ctr'})
            channel_ctr_mean = self.channel_ctr['channel_ctr'].mean()
            self.channel_ctr['channel_ctr'] = self.channel_ctr['channel_ctr'] + channel_ctr_mean
            self.df = self.df.merge(self.channel_ctr, how='left', on='channel')

    # def _merge_assert(self, feature_df: pd.DataFrame) -> None:
    #     self.res = self.res.merge(feature_df, how='left', on=self.uid_col)
    #     assert self.sample_num == self.res.shape[0], f"sample num: {self.sample_num}; res shape: {self.res.shape}"

    def _get_user_ctr_cross_dimension(self):
        large_cates = ["P4", 'P2']
        small_cates = ["P3", 'P1', 'device_conn_type']
        target_cates_cols = small_cates # + large_cates
        for col in target_cates_cols:
            origin_cols = list(pd.unique(self.df[col].values))
            target_cols = ['_'.join([self.col_prefix, col, str(ele), 'ctr']) for ele in origin_cols]
            feature_df = pd.pivot_table(self.df[[self.uid_col, col, self.target_col]], index=self.uid_col, columns=col, aggfunc=np.mean,
                           fill_value=0)[self.target_col].reset_index().rename(columns=dict(zip(origin_cols, target_cols)))[
                target_cols + [self.uid_col]]
            self.uid_feature = self.uid_feature.merge(feature_df, how='left', on=self.uid_col)
            self.feature_cols.extend(target_cols)

    def _get_user_click_exposure_num_feature(self) -> None:
        # return None
        map_func = [np.sum, np.size]
        map_col_name = ['sum', 'size']
        target_col_name = [self.col_prefix+ 'click_num', self.col_prefix+'exposure_num']
        feature_df = self.df.groupby(self.uid_col).agg({
            self.target_col: map_func
        }).reset_index().set_index(self.uid_col)[self.target_col].rename(columns=dict(zip(map_col_name, target_col_name))).reset_index()
        self.feature_cols.extend(target_col_name)
        self._merge_feature_df(feature_df=feature_df)

    def _get_user_ctr_feature(self)-> None:
        target_col = self.col_prefix+'ctr_without_pb'
        self.df['is_clicked_without_pb'] = self.df[self.target_col]/self.df['channel_ctr']
        feature_df = self.df.groupby([self.uid_col])[self.target_col].mean().reset_index().rename(columns={self.target_col: target_col})
        assert feature_df.shape[0] == self.uid_feature.shape[0], f"{feature_df.shape}, { self.uid_feature.shape}"
        self.feature_cols.append(target_col)
        self._merge_feature_df(feature_df=feature_df[[self.uid_col, target_col]])

    def _get_device_conn_related_feature(self) -> None:
        cols = [ 'site_id', 'site_category', 'device_conn_type', 'app_id', 'app_category',
                 'device_ip', 'device_id', 'device_type', 'P1', "P2", "P3", "P4"]
        targer_cols = [self.col_prefix+col+'_nunique' for col in cols]
        feature_df = self.df.groupby(self.uid_col)[cols].nunique().reset_index().rename(columns=dict(zip(cols, targer_cols)))
        assert feature_df.shape[0]  == self.uid_feature.shape[0]
        self.feature_cols.extend(targer_cols)
        self._merge_feature_df(feature_df=feature_df)

    def _get_user_preference_features(self):
        self.max_order_cols = [
            'site_category', 'site_id', 'app_id', 'app_category', 'device_conn_type'
                              , 'P1', 'P2', "P3", "P4"
                               #hour, app_id, app_category or connection type
                               ]
        most_clicked_features_cols = []
        col_prefix = user_most_click_col_prex
        max_order_df = self.df[[self.uid_col]].drop_duplicates()

        is_clicked_df = self.df[self.df[self.target_col]==1]
        for col in self.max_order_cols:
            target_col = col_prefix + col
            most_clicked_features_cols.append(target_col)
            df = pd.DataFrame(is_clicked_df.groupby([self.uid_col, col]).size()).reset_index().sort_values(
                [self.uid_col, 0], ascending=[False, False])
            df = df[df.index.isin(df[[self.uid_col]].drop_duplicates(keep='first').index)][
                [self.uid_col, col]].rename(columns={col: target_col})
            max_order_df = max_order_df.merge(df, how='left', on=self.uid_col)
        # max_order_df = max_order_df.fillna(0)
        self._merge_feature_df(feature_df=max_order_df)
        self.feature_cols.extend(most_clicked_features_cols)

    def get_features(self, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        if self.df is None:
            self.uid_feature = pd.read_csv(self.feature_path)
            self.feature_cols = list(self.uid_feature.columns)
            self.feature_cols.remove(self.uid_col)
            # self.feature_cols = list(set(self.uid_feature) - set([self.uid_col]))
        else:
            self.uid_feature = self.df[[self.uid_col]].drop_duplicates()
            funcs = [
                self._get_user_ctr_cross_dimension
                , self._get_user_click_exposure_num_feature
                , self._get_device_conn_related_feature
                , self._get_user_preference_features
                , self._get_user_ctr_feature
            ]
            for func in funcs:
                func()
        return self.uid_feature, self.feature_cols

    def save_features(self):
        self.uid_feature.to_csv(self.feature_path, index=False)

    def _merge_feature_df(self, feature_df: pd.DataFrame) -> None:
        self.uid_feature = self.uid_feature.merge(feature_df, how='left', on=self.uid_col)
