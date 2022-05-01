# -*- coding: utf-8 -*-
# @File    : DNNPipeline.py
# @Author  : Hua Guo
# @Time    : 2021/11/4 上午6:55
# @Disc    :
import tensorflow as tf
import pandas as pd
import os
import logging
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from typing import Tuple

from src.Pipeline import BasePipeline
from src.NewOrdinalEncoder import NewOrdinalEncoder
from src.utils.plot_utils import binary_classification_eval
from src.config import cate_encode_cols
from src.DLModel.FM_model import FM


class DNNPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=True, model_params={}):
        super(DNNPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.model_path = model_path
        self.model_params = model_params
        self.model_file_name = 'model.pb'
        self.preprocess_file_name = 'preprocess.pkl'
        if self.model_training:
            self.pipeline = self._init_model()
            self.preprocess_pipeline = None
        else:
            self.preprocess_pipeline, self.pipeline = self.load_model()
        self._check_dir(self.model_path)
        self.cate_encode_cols = cate_encode_cols

    def _init_model(self) -> tf.keras.models.Model:
        model_type = self.model_params.get('model_type', 'dnn')
        assert model_type in ['dnn', 'fm'], model_type
        input_dim = self.model_params['input_dim']
        lr = self.model_params['learning_rate']
        if model_type == 'dnn':
            model= tf.keras.models.Sequential([
                keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
                keras.layers.Dense(1024, activation='relu'),
                keras.layers.Dense(512, activation='relu'),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dropout(0.2),
                # keras.layers.Dense(128, activation='relu'),
                # keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])
        else:
            embed_dim = self.model_params['embed_dim']
            model = FM(feature_num=input_dim
                       , embed_dim=embed_dim
                       , output_dim=1
                       , sigmoid=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer,
                      loss=tf.losses.BinaryCrossentropy(from_logits=False),
                              metrics=[
                                  # tf.metrics.BinaryAccuracy()
                                  tf.metrics.Recall()
                                  , tf.metrics.AUC()
                                  , tf.metrics.Precision()
                                       ],
                      )
        # print(model.summary())
        return model

    def _transform_data(self, df: pd.DataFrame) -> tf.Tensor:
        # if one_hot:
        #     tensor = tf.convert_to_tensor(df.to_numpy().T)
        #     tensor = tf.squeeze(tf.one_hot(tensor, depth=2))
        # else:
        tensor = tf.convert_to_tensor(df.to_numpy())
        # tensor = tf.expand_dims(tensor, axis=1)
        return tensor

    def train(self, X: pd.DataFrame, y: pd.DataFrame, train_params: dict) -> None:
        X = X.fillna(0)
        epoch = train_params['epoch']
        batch_size = train_params['batch_size']
        pca_component_num = train_params['pca_component_num']

        df_for_encode_train = train_params['df_for_encode_train']
        transformer = NewOrdinalEncoder(category_cols=self.cate_encode_cols)
        min_max = MinMaxScaler()
        pca = PCA(n_components=pca_component_num)
        transformer.fit(df_for_encode_train)

        X = transformer.transform(X)
        X = min_max.fit_transform(X)
        X = pd.DataFrame(pca.fit_transform(X))

        self.preprocess_pipeline = Pipeline([
            ("new_ordinal_transformer", transformer)
            , ('min_max', min_max)
            , ('pca', pca)
        ])
        # X= pd.DataFrame(self.preprocess_pipeline.transform(X))

        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
        train_X = self._transform_data(train_X)
        train_y = self._transform_data(train_y)
        test_X = self._transform_data(test_X)
        test_y = self._transform_data(test_y)
        self.pipeline.fit(train_X,
                  train_y,
                  epochs=epoch,
                  validation_data=(test_X, test_y),
                    batch_size=batch_size
                  )

    def predict(self, X) -> tf.Tensor:
        X = X.fillna(0)
        X = pd.DataFrame(self.preprocess_pipeline.transform(X))
        return self.pipeline.predict(self._transform_data(X))

    def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None, importance=None) -> None:
        X = X.fillna(0)
        if default_fig_dir is None:
            fig_dir = self.model_path
        else:
            fig_dir = default_fig_dir
        self._check_dir(fig_dir)
        X = pd.DataFrame(self.preprocess_pipeline.transform(X.copy()))
        predict_prob = self.pipeline.predict(self._transform_data(X))
        binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)

    def load_model(self) -> Tuple[Pipeline, tf.keras.models.Model]:
        pipeline = tf.keras.models.load_model(self.model_path)
        pre_pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )
        return pre_pipeline, pipeline

    def save_model(self) -> None:
        file_name = joblib.dump(
            value=self.preprocess_pipeline,
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )[0]
        self.pipeline.save(self.model_path)
        logging.info(f'Model saved in {self.model_path}')