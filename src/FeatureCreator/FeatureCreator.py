# -*- coding: utf-8 -*-
# @File    : FeatureCreator.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 上午10:36
# @Disc    :
import pandas as pd
from typing import Tuple, List

from src.FeatureCreator import BaseFeatureCreator
# from src.config import user_most_click_col_prex, item_col, user_col, channel_feature_file


class FeatureCreator(BaseFeatureCreator):
    def __init__(self, user_feature_cols_tuple=None, item_feature_cols_tuple=None):
       #  self.cate_features = [
       #          #'channel',
       #           'site_id',
       # 'site_category', 'app_id', 'app_category',
       # 'device_model', 'device_type', 'device_conn_type', 'P1', 'P2',
       #      #     'P3',
       # 'P4']
        self.feature_cols = None
        self.feature_data = None
        # self.positon_ctr = pd.DataFrame({'channel': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 7},
        #  'ctr': {0: 0.16207190688240075,
        #   1: 0.19824186722568696,
        #   2: 0.11909262759924386,
        #   3: 0.0,
        #   4: 0.15436241610738255,
        #   5: 0.08910133843212237,
        #   6: 0.3037383177570093}})
        self.user_feature_cols_tuple = user_feature_cols_tuple
        self.item_feature_cols_tuple = item_feature_cols_tuple
        self.uid_col = user_col
        self.item_col = item_col
        self.positon_col = 'position'

    def _data_process(self) -> None:
        assert self.feature_data is not None
        self.feature_data['date'] = pd.to_datetime(self.feature_data['timestamp'], format='%y%m%d%H')
        self.feature_data['year'] = self.feature_data['date'].dt.year
        self.feature_data['day'] = self.feature_data['date'].dt.day
        self.feature_data['hour'] = self.feature_data['date'].dt.hour

    def _get_position_ctr(self):
        if self.item_feature_cols_tuple is not None:
            channel_ctr = self.item_feature_cols_tuple[2]
            col = 'channel_ctr'
            assert col in list(channel_ctr.columns), channel_ctr.columns
            self.feature_data = self.feature_data.merge(channel_ctr[['channel', col]], how='left', on='channel')
            self.feature_cols.append(col)

    def _get_device_id_feature(self):
        def func1(row):
            if row['device_id'] == 'a99f214a':
                return 1
            else:
                return 0
        def func2(row):
            unique_id = '_'.join([row['device_id'], row['device_model'], str(row['device_type'])])
            if unique_id == "a99f214a_8a4875bd_1":
                return 1
            else:
                return 0
        self.feature_data['whether_target_uid'] = self.feature_data.apply(lambda row: func1(row), axis=1)
        self.feature_data['whether_target_unique_d'] = self.feature_data.apply(lambda row: func2(row), axis=1)
        self.feature_cols.extend(
            ['whether_target_uid', 'whether_target_unique_d']
        )

    def _process_user_item_features(self):
        if self.user_feature_cols_tuple is not None:
            features, cols = self.user_feature_cols_tuple
            self.feature_data = self.feature_data.merge(features, how='left', on=self.uid_col)
            self.feature_cols += cols

            # # processing matching features
            matching_most_clicked_cols = [col for col in self.feature_cols if user_most_click_col_prex in col]
            matching_origin_cols = [col.replace(user_most_click_col_prex, '') for col in matching_most_clicked_cols]
            target_matching_cols = [ele+'_match' for ele in matching_most_clicked_cols]
            for most_clicked, origin, target in zip(matching_most_clicked_cols, matching_origin_cols, target_matching_cols):
                try:
                    self.feature_data[most_clicked] = self.feature_data[most_clicked] == self.feature_data[origin]
                    self.feature_data = self.feature_data.rename(columns={most_clicked: target})
                    col_index = self.feature_cols.index(most_clicked)
                    self.feature_cols[col_index] = target
                except:
                    print()
        if self.item_feature_cols_tuple is not None:
            features, cols = self.item_feature_cols_tuple[:2]
            self.feature_data = self.feature_data.merge(features, how='left', on=self.item_col)
            self.feature_cols += cols

    def get_features(self, df: pd.DataFrame, task='train_eval') -> Tuple[pd.DataFrame, List[str]]:
        """

        :param df:
        :param task: "train_eval" or 'inference'
        :return:
        """
        assert task in ['train_eval', 'inference']
        if task == 'inference':
            df[self.positon_col] = 1

      #   self.feature_data = df.reset_index()
      #   self.feature_cols = [ 'channel',
      #            'site_id',
      #  'site_category', 'app_id', 'app_category',
      #            'device_id',
      #                         # 'device_ip',
      # 'device_model',
      #            'device_type', 'device_conn_type', 'P1', 'P2',
      #            'P3',
      #  'P4', 'hour']
        self._data_process()
        funcs = [
            self._process_user_item_features  # place this function on the first place
            , self._get_position_ctr
            , self._get_device_id_feature
        ]
        for func in funcs:
            func()

        # reset row_id as index for further usage
        self.feature_data = self.feature_data.set_index('row_id')
        self.feature_data = self.feature_data.fillna(0)
        return self.feature_data[self.feature_cols], self.feature_cols