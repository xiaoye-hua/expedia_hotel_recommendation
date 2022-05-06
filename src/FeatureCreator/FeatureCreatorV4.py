# -*- coding: utf-8 -*-
# @File    : FeatureCreator.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
import numpy as np
from typing import Tuple, List

from src.FeatureCreator import BaseFeatureCreator
from src.config import regression_label, position_feature_path
from src.FeatureCreator.utils import get_label
from src.config import submission_cols, submission_cols_origin, prop_id, search_id


class FeatureCreatorV4(BaseFeatureCreator):
    def __init__(self, user_feature_class=None, item_feature_class=None, feature_cols=[]):
        self.feature_cols = set(feature_cols)
        self.user_feature_creator = user_feature_class
        self.item_feature_creator = item_feature_class
        if user_feature_class is not None:
            self.user_feature_creator = user_feature_class()
            self.user_feature_creator.get_features()
        if item_feature_class is not None:
            self.item_feature_creator = item_feature_class()
            self.item_feature_creator.get_features()
        self.position_col = 'position'

    def _get_date_info(self) -> None:
        assert self.feature_data is not None
        self.feature_data['date_time'] = pd.to_datetime(self.feature_data['date_time'])
        # self.feature_data['year'] = self.feature_data['date_time'].dt.year
        self.feature_data['month'] = self.feature_data['date_time'].dt.month
        self.feature_data['hour'] = self.feature_data['date_time'].dt.hour
        self.feature_data['dayofweek'] = self.feature_data['date_time'].dt.dayofweek
        self.feature_cols.update(['month', 'hour', 'dayofweek'])
        # self.feature_data['day_from_0'] = (self.feature_data['date_time'] - pd.to_datetime(min_date)).dt.days

    def _get_position_ctr(self):
        # if self.item_feature_cols_tuple is not None:
        channel_ctr = pd.read_csv(position_feature_path)
        # col = 'channel_ctr'
        position_features = ['position_ctr', 'position_ctcvr', 'position_reg_label','position_cnt']
        # assert col in list(channel_ctr.columns), channel_ctr.columns
        self.feature_data = self.feature_data.merge(channel_ctr[position_features+[self.position_col]], how='left', on=self.position_col)
        self.feature_cols.update(position_features)

    # def _get_device_id_feature(self):
    #     def func1(row):
    #         if row['device_id'] == 'a99f214a':
    #             return 1
    #         else:
    #             return 0
    #     def func2(row):
    #         unique_id = '_'.join([row['device_id'], row['device_model'], str(row['device_type'])])
    #         if unique_id == "a99f214a_8a4875bd_1":
    #             return 1
    #         else:
    #             return 0
    #     self.feature_data['whether_target_uid'] = self.feature_data.apply(lambda row: func1(row), axis=1)
    #     self.feature_data['whether_target_unique_d'] = self.feature_data.apply(lambda row: func2(row), axis=1)
    #     self.feature_cols.update(
    #         ['whether_target_uid', 'whether_target_unique_d']
    #     )

    # def _process_user_item_features(self):
    #     if self.user_feature_cols_tuple is not None:
    #         features, cols = self.user_feature_cols_tuple
    #         self.feature_data = self.feature_data.merge(features, how='left', on=self.uid_col)
    #         self.feature_cols += cols
    #
    #         # # processing matching features
    #         matching_most_clicked_cols = [col for col in self.feature_cols if user_most_click_col_prex in col]
    #         matching_origin_cols = [col.replace(user_most_click_col_prex, '') for col in matching_most_clicked_cols]
    #         target_matching_cols = [ele+'_match' for ele in matching_most_clicked_cols]
    #         for most_clicked, origin, target in zip(matching_most_clicked_cols, matching_origin_cols, target_matching_cols):
    #             try:
    #                 self.feature_data[most_clicked] = self.feature_data[most_clicked] == self.feature_data[origin]
    #                 self.feature_data = self.feature_data.rename(columns={most_clicked: target})
    #                 col_index = self.feature_cols.index(most_clicked)
    #                 self.feature_cols[col_index] = target
    #             except:
    #                 print()
    #     if self.item_feature_cols_tuple is not None:
    #         features, cols = self.item_feature_cols_tuple[:2]
    #         self.feature_data = self.feature_data.merge(features, how='left', on=self.item_col)
    #         self.feature_cols += cols

    def _get_target_variable(self):
        self.feature_data[regression_label] = self.feature_data.apply(lambda row: get_label(row), axis=1)

    # def _map_minus_1(self):
    #     map_dict = {
    #         1: 2
    #         , 0: 1
    #         , -1: 0
    #     }
    #     cols = [
    #         'comp1_rate'
    #         , 'comp2_rate'
    #         , 'comp3_rate'
    #         , 'comp4_rate'
    #         , 'comp5_rate'
    #         , 'comp6_rate'
    #         , 'comp7_rate'
    #         , 'comp8_rate'
    #     ]
    #     for col in cols:
    #         self.feature_data[col] = self.feature_data[col].map(map_dict)

    def _get_item_features(self):
        if self.item_feature_creator is not None:
            self.feature_data = self.feature_data.merge(self.item_feature_creator.item_feature, how='left', on=prop_id)

    def _get_listwise_features(self):
        self.feature_data['price_usd_rank'] = self.feature_data.groupby(search_id)['price_usd'].rank(method='dense', ascending=True)
        self.price_info = self.feature_data.groupby(search_id).agg({'price_usd': [np.max, np.min]})['price_usd'].reset_index().rename(columns={'amin':'price_min', 'amax': 'price_max'})
        self.price_rank_info = self.feature_data.groupby(search_id).agg({'price_usd_rank': [np.max, np.min]})['price_usd_rank'].reset_index().rename(columns={'amin':'price_rank_min', 'amax': 'price_rank_max'})
        self.price_info['price_diff'] = self.price_info['price_max'] -self.price_info['price_min']
        self.price_rank_info['price_rank_diff'] = self.price_rank_info['price_rank_max'] -self.price_rank_info['price_rank_min']
        def change_0(row, col):
            if row[col] == 0:
                return 1
            return row[col]
        self.price_info['price_diff'] = self.price_info.apply(lambda row: change_0(row, 'price_diff'), axis=1)
        self.price_rank_info['price_rank_diff'] = self.price_rank_info.apply(lambda row: change_0(row, 'price_rank_diff'), axis=1)
        self.feature_data = self.feature_data.merge(self.price_info, how='left', on=search_id)
        self.feature_data = self.feature_data.merge(self.price_rank_info, how='left', on=search_id)
        self.feature_data['price_percentile'] = (self.feature_data['price_usd']-self.feature_data['price_min'])/self.feature_data['price_diff']
        self.feature_data['price_rank_percentile'] = (self.feature_data['price_usd_rank']-self.feature_data['price_rank_min'])/self.feature_data['price_rank_diff']

    def get_features(self, df: pd.DataFrame, task='train_eval') -> Tuple[pd.DataFrame, List[str]]:
        """

        :param df:
        :param task: "train_eval" or 'inference'
        :return:
        """
        self.feature_data = df
        del df
        assert task in ['train_eval', 'inference']
        if task == 'inference':
            self.feature_data[self.position_col] = 1
        if task == 'train_eval':
            self._get_target_variable()
        funcs = [

            # self._get_date_info
            self._get_position_ctr
            # self._map_minus_1
            , self._get_item_features
            , self._get_date_info
            , self._get_listwise_features
        ]
        for func in funcs:
            func()
        # # reset row_id as index for further usage
        # self.feature_data = self.feature_data.set_index('row_id')
        # self.feature_data = self.feature_data.fillna(0)
        print(set(self.feature_data)-self.feature_cols)
        print(self.feature_cols-set(self.feature_data))
        if task == 'train_eval':
            final_cols = list(self.feature_cols) + [regression_label, 'srch_id']
        else:
            self.feature_data = self.feature_data.rename(columns=dict(zip(submission_cols_origin, submission_cols)))
            final_cols = list(self.feature_cols) + submission_cols
        return self.feature_data, list(self.feature_cols)