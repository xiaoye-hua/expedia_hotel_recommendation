# -*- coding: utf-8 -*-
# @File    : NewOrdinalEncoder.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 上午9:02
# @Disc    :
from sklearn.preprocessing import OrdinalEncoder
from typing import Optional, List


class NewOrdinalEncoder(OrdinalEncoder):
    """
    comparable with null value & numerical input
    """
    def __init__(self, category_cols: List[str]) -> None:
        super(OrdinalEncoder, self).__init__()
        self.ordinal_encoder = OrdinalEncoder(
            # handle_unknown='use_encoded_value', unknown_value='null'
        )
        self.category_cols = category_cols
        self.null_map = {col: 'null' for col in self.category_cols}

    def fit(self, X, y=None):
        X.fillna(self.null_map, inplace=True)
        # X[self.category_cols] = X[self.category_cols].astype('str')
        self.ordinal_encoder.fit(X[self.category_cols])
        return self

    def transform(self, X):
        # X[self.category_cols] = X[self.category_cols].astype('str')
        X.fillna(self.null_map, inplace=True)
        X.loc[:, self.category_cols] = self.ordinal_encoder.transform(X[self.category_cols])
        return X