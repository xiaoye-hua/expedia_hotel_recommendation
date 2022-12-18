# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Disc    :
from pandas_profiling import ProfileReport
import pandas as pd
import logging
import os
from src.utils import check_create_dir
logging.getLogger(__name__)


class DataProfiling():
    def __init__(self, data_dir, max_row=100000):
        self.data_dir = data_dir
        check_create_dir(self.data_dir)
        self.max_row = max_row

    def profiling_save(self, df:pd.DataFrame, file_name: str) -> ProfileReport:
        row_num = len(df)
        if row_num > self.max_row:
            logging.info(f'row number is {row_num} (bigger than {self.max_row}) -> sampling')
            df = df.sample(self.max_row)
        title = '_'.join(file_name.split('.')[:-1])
        profile = ProfileReport(df, title=f"{title}", minimal=True)
        file_dir = os.path.join(self.data_dir, file_name)
        logging.info(f'Saving report to {file_dir}')
        profile.to_file(file_dir)
        return profile

    def compare_save(self, profile1: ProfileReport, profile2: ProfileReport, file_name:str) -> None:
        comparison_report = profile1.compare(profile2)
        file_dir = os.path.join(self.data_dir, file_name)
        comparison_report.to_file(file_dir)