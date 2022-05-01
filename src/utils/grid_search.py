# -*- coding: utf-8 -*-
# @File    : grid_search.py
# @Author  : Hua Guo
# @Disc    :
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV


def grid_search_parms(X:pd.DataFrame, y: pd.DataFrame) -> None:
    xgb = XGBClassifier()
    parameters = {
        'learning_rate': [
                              .03,
            #                   0.05,
            0.07
        ],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_child_weight': [1],
        # 'silent': [1],
        'subsample': [
            #0.6, 0.8,
                       0.9],
        'colsample_bytree': [
           # 0.6,               0.8,
            0.9],
        'n_estimators': [
            100,
        ]
    }
    xgb_grid = GridSearchCV(xgb,
                            parameters,
                            cv=2,
                            n_jobs=5,
                            verbose=10
                            , scoring= 'roc_auc'
                            #'neg_log_loss'
                            #'
                            #['neg_log_loss', '']
                            )
    xgb_grid.fit(X,
                 y)

    print(xgb_grid.best_score_)
    print(xgb_grid.best_params_)