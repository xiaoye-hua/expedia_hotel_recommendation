# -*- coding: utf-8 -*-
# @File    : XGBRegressionPipeline.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler
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


class RidgeReg(BasePipeline):
    def __init__(self, model_path: str, model_training=False, model_params={}, **kwargs):
        super(RidgeReg, self).__init__(model_path=model_path, model_training=model_training)
        if self.model_training:
            self.pipeline = None
        else:
            self.load_pipeline()
        self.model_params = model_params

    def train(self, X, y, train_params) -> None:
        train_valid = train_params.get("train_valid", False)
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
        df_for_encode_train = train_params['df_for_encode_train']

        # onehot_feature = train_params.get('onehot_feature', None)
        sparse_feature = train_params.get('sparse_features', None)
        # assert onehot_feature is not None
        assert sparse_feature is not None
        data_transfomer = ColumnTransformer(
            transformers=[

                ('onehot', OneHotEncoder(handle_unknown='ignore'), sparse_feature),
                # , ('ordianl', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=100), list(set(sparse_feature)-set(onehot_feature)))
                ('minmax', MinMaxScaler(clip=True), list(set(X.columns)-set(sparse_feature)))
            ]
            ,  remainder='passthrough'
        )

        data_transfomer.fit_transform(df_for_encode_train)
        pipeline_lst.append(
            ('data_transformer', data_transfomer)
        )
        train_X = data_transfomer.transform(train_X)
        logging.info(f"Train data shape after data process: {train_X.shape}")
        grid_search_dict = train_params.get('grid_search_dict', None)
        self.model = Ridge()
        self.pipeline = Pipeline(pipeline_lst)
        if train_valid:
            eval_X = test_X.copy()
            eval_y = test_y.copy()
            eval_X = data_transfomer.transform(eval_X)
            logging.info(f"Eval data shape after data process: {eval_X.shape}")
            self.model.fit(X=train_X, y=train_y)
            predict_train = self.model.predict(X=train_X)
            predict_eval = self.model.predict(X=eval_X)
            mae_train = mean_absolute_error(y_true=train_y, y_pred=predict_train)
            mae_eval = mean_absolute_error(y_true=eval_y, y_pred=predict_eval)
            logging.info(f"Train MAE: {mae_train}; Eval MAE: {mae_eval}")
            # print(f"Model params are {self.model.get_params()}")
            # self._plot_eval_result()
        else:
            self.model.fit(X=train_X, y=train_y)
        pipeline_lst.append(('model', self.model))
        self.pipeline = Pipeline(pipeline_lst)
        if train_valid:
            self.eval(X=test_X, y=test_y, default_fig_dir=os.path.join(self.eval_result_path, 'eval_data'))

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

    def _plot_eval_result(self, metric='mae'):
        # # retrieve performance metrics
        # # results = self.model.evals_result()
        # # plot learning curves
        # plt.plot(results['validation_0'][metric], label='train')
        # plt.plot(results['validation_1'][metric], label='test')
        # # show the legend
        # plt.legend()
        # # show the plot
        # plt.savefig(os.path.join(self.eval_result_path, 'xgb_train_eval.png'))
        # plt.show()
        pass

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=True, performance_result_file='performance.txt', **kwargs) -> Tuple[float, float]:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        logging.info(f"Saving eval result to {fig_dir}")
        transfomers = self.pipeline[self.data_transfomer_name].transformers
        feature_cols = []
        for name, encoder, features_lst in transfomers:
            if name == self.onehot_encoder_name and len(features_lst)>0:
                original_ls = features_lst
                features_lst = self.pipeline[self.data_transfomer_name].named_transformers_[self.onehot_encoder_name].get_feature_names()
                for lst_idx, col in enumerate(features_lst):
                    index, cate= col.split('_')
                    index = int(index[1:])
                    original = original_ls[index]
                    features_lst[lst_idx] = '_'.join([cate, original])
            feature_cols += list(features_lst)
        logging.info(f"features num: {len(feature_cols)}")
        logging.info(f"feature_col is {feature_cols}")

        df = pd.DataFrame(
            {
                'feature': feature_cols,
                'coef': self.pipeline['model'].coef_,
                'abs_coef': abs(self.pipeline['model'].coef_)
            }
        ).sort_values('abs_coef', ascending=False)
        df.to_csv(os.path.join(fig_dir, 'coef.csv'), index=False)

        # if importance:
        #     show_feature_num = min(30, len(X.columns))
        #     plot_feature_importances(model=self.pipeline['model'], feature_cols=list(X.columns), show_feature_num=show_feature_num,
        #                              fig_dir=fig_dir)

        predicted = self.predict(X=X.copy())
        mae = mean_absolute_error(y_true=y, y_pred=predicted)
        mape = mean_absolute_percentage_error(y_true=y, y_pred=predicted)

        mae_df  = pd.DataFrame(
            {
                "y_true": y.values
                , 'y_predict': predicted
            }
        )
        mae_df['mae']  = abs(mae_df['y_predict']-mae_df['y_true'])
        details_mae = pd.DataFrame(round(mae_df['mae'].describe([0.05*i for i in range(20)]), 2))
        res = f"MAE: {mae}; MAPE: {mape}"
        # print(res)
        with open(os.path.join(fig_dir, performance_result_file), 'w+') as f:
            f.write(res)
        details_mae.to_csv(os.path.join(fig_dir, 'detailed_mae.csv'))
        logging.info(f"Model eval result saved in {fig_dir}")
        return mae, mape