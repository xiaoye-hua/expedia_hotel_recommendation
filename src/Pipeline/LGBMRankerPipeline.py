# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Disc    :
# from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRanker
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# from sklearn2pmml import PMMLPipeline as Pipeline
# from sklearn2pmml import sklearn2pmml

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer
import logging
import pandas as pd
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Tuple
import time

from src.BaseClass.BasePipeline import BasePipeline
from src.utils.plot_utils import plot_feature_importances
logging.getLogger(__name__)


class LGBMRankerPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, model_params={}, **kwargs):
        super(LGBMRankerPipeline, self).__init__(model_path=model_path, model_training=model_training)
        if self.model_training:
            self.pipeline = None
        else:
            self.load_pipeline()
        self.model_params = model_params

    def train(self, X, y, train_params) -> None:
        train_valid = train_params.get("train_valid", False)
        category_features = train_params.get('category_features', None)
        assert category_features is not None
        train_group = train_params['train_group']
        eval_group = train_params['eval_group']
        if train_valid:
            if train_params.get('eval_X', None) is not None:
                print('Eval data from train_params: ..')
                train_X, train_y = X.copy(), y.copy()
                test_X, test_y = train_params["eval_X"].copy(), train_params['eval_y'].copy()
            else:
                train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        else:
            train_X, train_y = X.copy(), y.copy()
        pipeline_lst = []
        # df_for_encode_train = train_params['df_for_encode_train']

        # onehot_feature = train_params.get('onehot_feature', None)
        # sparse_features = train_params.get('sparse_features', None)
        # # # assert onehot_feature is not None
        # assert sparse_features is not None
        #
        # data_transfomer = make_column_transformer(
        #     (OneHotEncoder(),  make_column_selector(dtype_include=np.object))
        #     , (OrdinalEncoder(), make_column_selector(dtype_include=np.bool))
        #     , remainder='passthrough'
        # )
        #
        # data_transfomer = ColumnTransformer(
        #     transformers=[
        #         ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=100000), category_features)
        #         # , ('ordianl', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=100), list(set(sparse_feature)-set(onehot_feature)))
        #         , ('passthrough', 'passthrough', list(set(X.columns)-set(category_features)))
        #     ]
        #     ,  remainder='passthrough'
        # )

        # train_X = data_transfomer.fit_transform(train_X.copy())
        # pipeline_lst.append(
        #     ('data_transformer', data_transfomer)
        # )

        # train_X = data_transfomer.transform(train_X)

        logging.info(f"Train data shape after data process: {train_X.shape}")
        grid_search_dict = train_params.get('grid_search_dict', None)
        # if grid_search_dict is not None:
        #     logging.info(f"Grid searching...")
        #     begin = time.time()
        #     self.model_params = self.grid_search_parms(X=train_X, y=train_y, parameters=grid_search_dict)
        #     end = time.time()
        #     logging.info(f"Time consumed: {round((end-begin)/60, 3)}mins")
        self.model_params['importance_type'] = 'gain'
        self.xgb = LGBMRanker(**self.model_params)
        if train_valid:
            eval_X = test_X.copy()
            eval_y = test_y.copy()
            # eval_X = data_transfomer.transform(eval_X)
            logging.info(f"Eval data shape after data process: {eval_X.shape}")
            self.xgb.fit(X=train_X, y=train_y, group=train_group,  verbose=True  #, eval_metric=['mae']
                         , eval_set=[
                   [train_X, train_y],
                                     [eval_X, eval_y]]
                         , eval_group=[ train_group,
                                       eval_group]
                         , eval_at=[38]
                         , early_stopping_rounds=30
                         , categorical_feature=category_features
                         )

            print(f"Model params are {self.xgb.get_params()}")
            # self._plot_eval_result()
        else:
            self.xgb.fit(X=train_X, y=train_y, train_group=train_group, verbose=True #, eval_metric=['mae']
                         , eval_set=[[train_X, train_y]], eval_group=[train_group]
                         , eval_at=[38]
                        , early_stopping_rounds=30
                         , categorical_feature=category_features)
        pipeline_lst.append(('model', self.xgb))
        self.pipeline = Pipeline(pipeline_lst)
        if train_valid:
            self.eval(X=eval_X, y=eval_y, default_fig_dir=os.path.join(self.eval_result_path, 'eval_data'))

    def predict(self, X) -> pd.DataFrame:
        res = self.pipeline.predict(X.copy())
        return res

    def load_pipeline(self, **kwargs) -> None:
        self.pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.model_file_name)
        )

    def save_pipeline(self) -> None:
        file_name = joblib.dump(
            value=self.pipeline,
            filename=os.path.join(self.model_path, self.model_file_name)
        )[0]
        # sklearn2pmml(self.pipeline, os.path.join(self.model_path, 'pipeline.pmml'))
        # logging.info(f"Saving pipeline to {file_name}")

    def _plot_eval_result(self, metric='mae'):
        # retrieve performance metrics
        results = self.xgb.evals_result()
        # plot learning curves
        plt.plot(results['validation_0'][metric], label='train')
        plt.plot(results['validation_1'][metric], label='test')
        # show the legend
        plt.legend()
        # show the plot
        plt.savefig(os.path.join(self.eval_result_path, 'xgb_train_eval.png'))
        # plt.show()

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=True, performance_result_file='performance.txt', **kwargs) -> Tuple[float, float]:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        logging.info(f"Saving eval result to {fig_dir}")
        if importance:
            show_feature_num = min(30, len(X.columns))
            plot_feature_importances(model=self.pipeline['model'], feature_cols=list(X.columns), show_feature_num=show_feature_num,
                                     fig_dir=fig_dir)
        # predicted = self.predict(X=X.copy())

        # mae = mean_absolute_error(y_true=y, y_pred=predicted)
        # mape = mean_absolute_percentage_error(y_true=y, y_pred=predicted)
        #
        # mae_df  = pd.DataFrame(
        #     {
        #         "y_true": y.values
        #         , 'y_predict': predicted
        #     }
        # )
        # mae_df['mae']  = abs(mae_df['y_predict']-mae_df['y_true'])
        # details_mae = pd.DataFrame(round(mae_df['mae'].describe([0.05*i for i in range(20)]), 2))
        # res = f"MAE: {mae}; MAPE: {mape}"
        # # print(res)
        # with open(os.path.join(fig_dir, performance_result_file), 'w+') as f:
        #     f.write(res)
        # details_mae.to_csv(os.path.join(fig_dir, 'detailed_mae.csv'))
        # logging.info(f"Model eval result saved in {fig_dir}")
        # return mae, mape