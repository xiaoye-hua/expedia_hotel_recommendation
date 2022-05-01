# -*- coding: utf-8 -*-
# @File    : BasePipeline.py
# @Author  : Hua Guo
# @Disc    :
from abc import ABCMeta, abstractmethod
import os
import pandas as pd
import logging
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
