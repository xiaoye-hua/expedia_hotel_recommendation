# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.metrics import ndcg_score


def get_ndcg(df):
    ndcg_df = df.groupby('srch_id').apply(lambda row: ndcg_score(y_true=[row['label']], y_score=[row['predicted']], k=38))
    ndcg = ndcg_df.mean()
    return round(ndcg, 3)