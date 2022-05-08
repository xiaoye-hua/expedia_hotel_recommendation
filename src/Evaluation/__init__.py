# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.metrics import ndcg_score


def get_ndcg(df, at_k=38):
    ndcg_df = df.groupby('srch_id').apply(lambda row: ndcg_score(y_true=[row['label']], y_score=[row['predicted']],
                                                                 k=at_k))
    ndcg = ndcg_df.mean()
    return round(ndcg, 3)