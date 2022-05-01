# -*- coding: utf-8 -*-
# @File    : FM_model.py
# @Author  : Hua Guo
# @Time    : 2021/11/4 上午7:51
# @Disc    :
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Embedding, Activation

from src.DLModel import DLModel


class LinerPart(Layer):
    def __init__(self, hidden_unit=1) -> None:
        super(LinerPart, self).__init__()
        self.layer = Dense(units=hidden_unit, use_bias=True)

    def call(self, inputs, **kwargs):
        return self.layer(inputs)


class NonLinearPart(Layer):
    def __init__(self, feature_num: int, hidden_dim:int) -> None:
        super(NonLinearPart, self).__init__()
        self.embedding = Embedding(input_dim=feature_num, output_dim=hidden_dim)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        embeding = self.embedding(inputs)
        square_of_sum = tf.square(tf.reduce_sum(embeding, axis=1))
        sum_of_square = tf.reduce_sum(tf.square(embeding), axis=1)
        final = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=1, keepdims=True)
        return final


class FM(DLModel):
    def __init__(self, feature_num: int, embed_dim: int, output_dim=1, sigmoid=True) -> None:
        super(FM, self).__init__()
        self.linear = LinerPart(hidden_unit=output_dim)
        self.non_linear = NonLinearPart(feature_num=feature_num, hidden_dim=embed_dim)
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.ac = Activation(activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        net_sum = self.linear(inputs) + self.non_linear(inputs)
        if self.sigmoid:
            final = self.ac(net_sum)
            return final
        return net_sum