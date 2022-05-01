# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/10/30 上午9:01
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