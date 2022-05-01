# -*- coding: utf-8 -*-
# @File    : BaseFeatureCreator.py
# @Author  : Hua Guo
# @Disc    :
import os
from abc import ABCMeta, abstractmethod


class BaseFeatureCreator(metaclass=ABCMeta):

    @abstractmethod
    def get_features(self, **kwargs):
        pass

    def _check_dir(self, directory):
        if not os.path.isdir(directory):
            os.makedirs(directory)