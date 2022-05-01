# -*- coding: utf-8 -*-
# @File    : XGBoostPipeline.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 上午10:37
# @Disc    :
from xgboost.sklearn import XGBClassifier
from src.NewOrdinalEncoder import NewOrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
import logging

from src.Pipeline import BasePipeline
from src.utils.plot_utils import plot_feature_importances, binary_classification_eval
from src.NewOrdinalEncoder import NewOrdinalEncoder
from src.utils.grid_search import grid_search_parms
from src.config import cate_encode_cols


class XGBoostPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=True, model_params={}) -> None:
        super(XGBoostPipeline, self).__init__(model_training=model_training, model_path=model_path)
        self.model_params = model_params
        self.model_file_name = 'pipeline.pkl'
        self.pipeline = None
        self.eval_result_path = os.path.join(self.model_path, 'eval_result')
        self._check_dir(self.model_path)
        self._check_dir(self.eval_result_path)
        self.cate_encode_cols = cate_encode_cols
        self.pca_component_num = None

    def load_model(self) -> None:
        self.pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.model_file_name)
        )

    def train(self, X: pd.DataFrame, y: pd.DataFrame, train_params: dict) -> None:
        pipeline_lst = []

        df_for_encode_train = train_params['df_for_encode_train']
        train_valid = train_params["train_valid"]
        transformer = NewOrdinalEncoder(category_cols=self.cate_encode_cols)
        transformer.fit(df_for_encode_train)
        X = transformer.transform(X=X)
        pipeline_lst.append(("new_ordinal_transformer", transformer))

        if train_params.get('pca_component_num', False):
            pca_component_num = train_params['pca_component_num']
            self.pca_component_num = pca_component_num
            min_max = MinMaxScaler()
            pca = PCA(n_components=pca_component_num)
            X = min_max.fit_transform(X)
            X = pca.fit_transform(X)
            pipeline_lst.extend(
                [
                    ('min_max', min_max)
                    , ('pca', pca)
                ]
            )
        self.xgb = XGBClassifier(**self.model_params)
        print(f"Model params are {self.xgb.get_params()}")

        # X.to_csv('logs/train_features.csv', index=False)
        grid_search = False
        if grid_search:
            print("Grid searching...")
            grid_search_parms(X, y)
        # X.to_csv('logs/train_features.csv', index=False)
        if train_valid:
            # train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
            eval_X, eval_y = train_valid
            for trans in pipeline_lst:
                eval_X = trans[1].transform(eval_X)
            self.xgb.fit(X=X, y=y, verbose=True, eval_metric='logloss'
                         , eval_set=[[X, y], [eval_X, eval_y]])
            self._plot_eval_result()
        else:
            self.xgb.fit(X=X, y=y, verbose=True, eval_metric='logloss'
                         , eval_set=[[X, y]])
        pipeline_lst.append(('model', self.xgb))
        # print(pipeline_lst)
        self.pipeline = Pipeline(pipeline_lst)
        # self.pipeline.fit(X=X, y=y)

    def predict(self, X) -> pd.DataFrame:
        return self.pipeline.predict_proba(X=X)[:, 1]

    def save_model(self) -> None:
        file_name = joblib.dump(
            value=self.pipeline,
            filename=os.path.join(self.model_path, self.model_file_name)
        )[0]
        logging.info(file_name)

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=True) -> None:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        if importance and self.model_training and not self.pca_component_num:
            plot_feature_importances(model=self.xgb, feature_cols=list(X.columns), show_feature_num=len(X.columns), fig_dir=fig_dir)
        predict_prob = self.pipeline.predict_proba(X=X.copy())[:, 1]
        binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)

    def _plot_eval_result(self):
        # retrieve performance metrics
        results = self.xgb.evals_result()
        # plot learning curves
        plt.plot(results['validation_0']['logloss'], label='train')
        plt.plot(results['validation_1']['logloss'], label='test')
        # show the legend
        plt.legend()
        # show the plot
        plt.savefig(os.path.join(self.eval_result_path, 'xgb_train_eval.png'))
        # plt.show()