# -*- coding: utf-8 -*-
# @File    : BasePipeline.py
# @Author  : Hua Guo
# @Disc    :
from abc import ABCMeta, abstractmethod
import os
import tensorflow as tf
from typing import Tuple
import joblib
from sklearn.pipeline import Pipeline
import pandas as pd
import logging
from src.utils.plot_utils import binary_classification_eval, plot_feature_importances
logging.getLogger(__name__)

from src.utils.plot_utils import binary_classification_eval


class BasePipeline(metaclass=ABCMeta):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        self.model_training = model_training
        self.model_path = model_path
        self.eval_result_path = os.path.join(self.model_path, 'eval_test')
        self._check_dir(self.model_path)
        self._check_dir(self.eval_result_path)
        self.model_file_name = 'pipeline.pkl'
        self.data_transfomer_name = 'data_transformer'
        self.model_name = 'model'
        self.onehot_encoder_name = 'onehot'

    @abstractmethod
    def train(self, X, y, train_params):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_pipeline(self):
        pass

    @abstractmethod
    def load_pipeline(self):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)

    # def eval(self, X: pd.DataFrame, y: pd.DataFrame, default_fig_dir=None) -> None:
    #     if default_fig_dir is None:
    #         fig_dir = self.eval_result_path
    #     else:
    #         fig_dir = default_fig_dir
    #     self._check_dir(fig_dir)
    #     predict_prob = self.predict(X)
    #     binary_classification_eval(test_y=y, predict_prob=predict_prob, fig_dir=fig_dir)
    #     logging.info(f"Model eval result saved in {fig_dir}")


class BaseDNNPipeline(BasePipeline):
    def __init__(self, model_path: str, model_training=False, **kwargs):
        super(BaseDNNPipeline, self).__init__(model_path=model_path, model_training=model_training)
        self.model_file_name = 'model.pb'
        self.preprocess_file_name = 'preprocess.pkl'
        if model_training:
            self.pipeline = None
            self.model = None
        else:
            self.pipeline, self.model = self.load_pipeline()

    def predict(self, X):
        X = self.pipeline.transform(X)
        return self.model.predict(X)

    def load_pipeline(self) -> Tuple[Pipeline, tf.keras.models.Model]:
        pipeline = tf.keras.models.load_model(self.model_path)
        pre_pipeline = joblib.load(
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )
        return pre_pipeline, pipeline

    def save_pipeline(self) -> None:
        self._check_dir(self.model_path)
        file_name = joblib.dump(
            value=self.pipeline,
            filename=os.path.join(self.model_path, self.preprocess_file_name)
        )[0]
        tf.keras.models.save_model(model=self.model, filepath=os.path.join(self.model_path, self.model_file_name))
        logging.info(f'Model saved in {self.model_path}')

    def train(self, X, y, train_params):
        pass