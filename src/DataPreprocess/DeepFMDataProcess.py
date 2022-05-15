# -*- coding: utf-8 -*-
# @File    : DeepFMDataProcess.py
# @Author  : Hua Guo
# @Disc    :
import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer, OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
import logging
logging.getLogger(__name__)


class DeepFMDataProcess(BaseEstimator, TransformerMixin):
    def __init__(self, dense_feature: List[str], sparse_feature: List[str], dense_to_sparse=False) -> None:
        super(DeepFMDataProcess, self).__init__()
        self.dense_to_sparse = dense_to_sparse
        self.dense_feature = dense_feature
        self.sparse_feature = sparse_feature
        self.pipeline = None

    def fit(self, X, y=None):
        if self.dense_to_sparse:
            numeric_transformer = KBinsDiscretizer(n_bins=20, encode='ordinal')
        else:
            numeric_transformer = MinMaxScaler(feature_range=(0, 1))
        categorical_transformer = OrdinalEncoder(dtype=np.int32)
        self.pipeline = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, self.sparse_feature),
                ("num", numeric_transformer, self.dense_feature),
            ]
        )
        logging.info(self.pipeline)
        self.pipeline.fit(X)
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(self.pipeline.transform(X), columns=X.columns,)
        X[self.sparse_feature] = X[self.sparse_feature].astype("int32")
        X[self.dense_feature] = X[self.dense_feature].astype('float32')
        return X