# -*- coding: utf-8 -*-
# @File    : __init__.py.py
# @Author  : Hua Guo
# @Time    : 2021/11/4 上午8:01
# @Disc    :
from abc import abstractmethod
from tensorflow.keras import Model


class DLModel(Model):
    def __init__(self):
        super(DLModel, self).__init__()
        self.layers_lst = []

    @abstractmethod
    def call(self, inputs, training=None, mask=None):
        for layer in self.layers_lst:
            inputs = layer(inputs)
        return inputs