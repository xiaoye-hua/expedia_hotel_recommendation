# -*- coding: utf-8 -*-
# @File    : DeepFMPipeline.py
# @Author  : Hua Guo
# @Disc    :
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, KBinsDiscretizer
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import logging
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from src.utils.plot_utils import binary_classification_eval, plot_feature_importances
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.BaseClass.BasePipeline import BaseDNNPipeline
from src.DataPreprocess.DeepFMDataProcess import DeepFMDataProcess
logging.getLogger(__name__)


class DeepFMPipeline(BaseDNNPipeline):
    def __init__(self, model_path: str, task: str, model_training=False, **kwargs):
        super(DeepFMPipeline, self).__init__(model_path=model_path, model_training=model_training, **kwargs)
        self.model_file_name = 'model.pb'
        self.preprocess_file_name = 'preprocess.pkl'
        self.model_training = model_training
        if self.model_training:
            self.pipeline = None
            self.model = None
        else:
            self.pipeline, self.model = self.load_pipeline()
        assert task in ['binary', 'regression'], 'refer to https://deepctr-doc.readthedocs.io/en/latest/deepctr.models.deepfm.html'
        self.task = task

    def train(self, X, y, train_params):
        train_valid = train_params.get("train_valid", False)
        if train_valid:
            if train_params.get('eval_X', None) is not None:
                print('Eval data from train_params: ..')
                train_X, train_y = X.copy(), y.copy()
                eval_X, eval_y = train_params["eval_X"].copy(), train_params['eval_y'].copy()
            else:
                train_X, eval_X, train_y, eval_y = train_test_split(X, y, test_size=0.2)
        else:
            train_X, train_y = X.copy(), y.copy()

        df_for_encode_train = train_params['df_for_encode_train']
        batch_size = train_params['batch_size']
        epoches = train_params['epoches']
        dense_to_sparse = train_params['dense_to_sparse']
        assert epoches is not None
        assert batch_size is not None
        assert dense_to_sparse is not None

        self.dense_features = train_params['dense_features']
        self.sparse_features = train_params['sparse_features']

        self.pipeline = DeepFMDataProcess(
            # dense_feature=self.dense_features,
                                          sparse_feature=self.sparse_features
                                        , dense_feature=[]
                                          # sparse_feature=[]
                                          , dense_to_sparse=dense_to_sparse
                                         )
        logging.info(self.pipeline)

        df_for_encode_train = self.pipeline.fit_transform(df_for_encode_train)
        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=df_for_encode_train[feat].max() + 1, embedding_dim=15)
                                  for i, feat in enumerate(self.sparse_features)] + [DenseFeat(feat, 1, )
                                                                                for feat in self.dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        # train data
        train_X = self.pipeline.transform(train_X)
        train_model_input = self._process_train_data(train_X)
        train_label = train_y.values
        # eval data
        eval_X = self.pipeline.transform(eval_X)
        eval_model_input = self._process_train_data(eval_X)
        eval_label = eval_y.values

        self.model = DeepFM(linear_feature_columns, dnn_feature_columns, task=self.task)
        self.model.summary()
        if self.task == 'binary':
            self.model.compile(optimizer="adam",
                               loss="binary_crossentropy",
                                metrics=['binary_crossentropy',
                                         # tf.keras.metrics.AUC()
                                         ],
                               )
        elif self.task == 'regression':
            self.model.compile(optimizer="adam",
                               loss=tf.keras.losses.MeanSquaredError(),
                                # metrics=[tf.keras.losses.MeanSquaredError()],
                               )
        else:
            raise ValueError

        self.model.fit(train_model_input, train_label,
                        batch_size=batch_size,
                       epochs=epoches,
                       validation_data=(eval_model_input, eval_label)
                       , callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)])
        dot_img_file = os.path.join(self.model_path, 'model_structure.png')
        logging.info(f'Saving model to {dot_img_file}')
        tf.keras.utils.plot_model(self.model, to_file=dot_img_file, show_shapes=True)

    def _process_train_data(self, X):
        train_model_input = {}
        if self.model_training:
            for col in self.sparse_features+self.dense_features:
                train_model_input[col] = X[col]
        else:
            for idx, col in enumerate(self.sparse_features+self.dense_features):
                target_col = f'input_{idx+1}'
                train_model_input[target_col] = X[col]
        return train_model_input

    def predict(self, X):
        X = self.pipeline.transform(X)
        trian_input = self._process_train_data(X)
        prob = self.model.predict(trian_input)
        return prob

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=False,
             performance_result_file='performance.txt', **kwargs) -> None:
        if default_fig_dir is None:
            fig_dir = self.eval_result_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        predicted = self.predict(X=X.copy())
        if self.task == 'binary':
            binary_classification_eval(test_y=y, predict_prob=predicted, fig_dir=fig_dir)
        elif self.task == 'regression':
            mae = mean_absolute_error(y_true=y, y_pred=predicted)
            mse = mean_squared_error(y_true=y, y_pred=predicted)

            mae_df = pd.DataFrame(
                {
                    "y_true": y.values
                    , 'y_predict': predicted.reshape([predicted.shape[0], ])
                }
            )
            mae_df['mae'] = abs(mae_df['y_predict'] - mae_df['y_true'])
            details_mae = pd.DataFrame(round(mae_df['mae'].describe([0.05 * i for i in range(20)]), 2))
            res = f"MAE: {mae}; MSE: {mse}"
            # print(res)
            with open(os.path.join(fig_dir, performance_result_file), 'w+') as f:
                f.write(res)
            details_mae.to_csv(os.path.join(fig_dir, 'detailed_mae.csv'))
            logging.info(f"Model eval result saved in {fig_dir}")
        else:
            raise ValueError