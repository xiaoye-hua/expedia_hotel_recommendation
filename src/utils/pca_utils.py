# -*- coding: utf-8 -*-
# @File    : pca_utils.py
# @Author  : Hua Guo
# @Disc    :
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


def plot_pca_res(train: pd.DataFrame, file_name=None, detail_component_num=50) -> None:
    sc_V = MinMaxScaler()
    sc_V.fit(train)
    train = sc_V.transform(train)
    plt.figure(figsize=(25,6))
    pca = PCA().fit(train)
    plt.plot(range(1, train.shape[1]+1),np.cumsum(pca.explained_variance_ratio_), "bo-")
    plt.xlabel("Component Count")
    plt.ylabel("Variance Ratio")
    plt.xticks(range(1, train.shape[1]+1))
    plt.grid()
    if file_name is not None:
        plt.savefig(file_name, pad_inches='tight')
    plt.show()
    if train.shape[1] > 2*detail_component_num:
        plt.figure(figsize=(25, 6))
        plt.plot(range(1, detail_component_num + 1), np.cumsum(pca.explained_variance_ratio_[:detail_component_num]), "bo-")
        plt.xlabel("Component Count")
        plt.ylabel("Variance Ratio")
        plt.xticks(range(1, detail_component_num + 1))
        plt.grid()
        plt.title(f"Top {detail_component_num} plot")
        plt.show()