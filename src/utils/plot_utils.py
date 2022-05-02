# -*- coding: utf-8 -*-
# @File    : plot_utils.py
# @Author  : Hua Guo
# @Disc    :
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
logging.getLogger(__name__)
import os
from xgboost.sklearn import XGBModel
from typing import List
from sklearn.metrics import roc_curve, classification_report, roc_auc_score, confusion_matrix
from src.utils.confusion_matrix_pretty_print import pretty_plot_confusion_matrix


def count_plot(df: pd.DataFrame, col: str, xytext=(0, 0), show_details=True, regression_target=None, max_cate=100) -> None:
    '''
    custom count plot
    Args:
        df:
        col:
        xytext:

    Returns:
    '''
    try:
        df = df.copy()
        plt.figure(figsize=(20, 6))
        unique_num = len(df[col].unique())
        if unique_num > max_cate:
            print(f"Unique num too huge: {unique_num}; cut to {10} categories")
            df[col] = pd.qcut(df[col], 10)
        ax = sns.countplot(data=df, x=col)
        if show_details:
            for bar in ax.patches:
                ax.annotate('%{:.2f}\n{:.0f}'.format(100*bar.get_height()/len(df),bar.get_height()), (bar.get_x() + bar.get_width() / 2,
                            bar.get_height()), ha='center', va='center',
                            size=11, xytext=xytext,
                            textcoords='offset points')
        if regression_target is not None:
            target_col = 'mean_'+regression_target
            stats = df.groupby(col).agg({regression_target: [np.mean, np.max, np.min]})[regression_target].reset_index().rename(columns={'mean':target_col})
            ax_twin = ax.twinx()
            # ax_twin = \
            sns.pointplot(x=col, y=target_col, data=stats,
                          color='black', legend='avg_delivery_time',
                          order=np.sort(df[col].dropna().unique()),
                          linewidth=0.1,
                          )
        plt.show()
        plt.figure(figsize=(20, 6))
        sns.boxplot(x=col, y=target_col, data=df)
        plt.show()
    except:
        return

def plot_feature_importances(model: XGBModel, feature_cols: List[str], show_feature_num=10, figsize=(20, 10), fig_dir=None):
    """
    plot feature importance of xgboost model
    Args:
        model:
        feature_cols:
        show_feature_num:
        figsize:

    Returns:

    """
    all_feature_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    feature_imp = all_feature_imp[:show_feature_num]
    plt.figure(figsize=figsize)
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.title("Feature Importance")
    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, 'feature_importance.png'))
        pd.DataFrame(all_feature_imp).to_csv(os.path.join(fig_dir, 'feature_imp.csv'))
    else:
        plt.show()


def plot_shap_importance(pipeline, X, fig_dir=None):
    import shap
    explainer = shap.Explainer(pipeline['model'])
    features = pipeline['data_transformer'].transform(X.copy())
    df = pd.DataFrame(features, columns=X.columns)
    shap_values = explainer(df)
    plt.figure(figsize=(10, 20))
    shap.plots.beeswarm(shap_values, max_display=30)
    plt.title("Shap Feature Importance")
    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, 'shap_feature_importance.png'))
    else:
        plt.show()


def plot_auc_plot(y_test: pd.DataFrame, pred_prob: pd.DataFrame, fig_dir=None) -> None:
    auc = roc_auc_score(y_test, pred_prob)
    false_positive_rate, true_positive_rate, thresolds = roc_curve(y_test, pred_prob)
    plt.figure(figsize=(5, 5), dpi=100)
    plt.axis('scaled')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title("AUC & ROC Curve")
    plt.plot(false_positive_rate, true_positive_rate, 'g')
    plt.fill_between(false_positive_rate, true_positive_rate, facecolor='lightgreen', alpha=0.7)
    plt.text(0.95, 0.05, 'AUC = %0.4f' % auc, ha='right', fontsize=12, weight='bold', color='blue')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    if fig_dir is not None:
        plt.savefig(os.path.join(fig_dir, 'auc_plot.png'))
    else:
        plt.show()


def binary_classification_eval(test_y: pd.DataFrame, predict_prob: pd.DataFrame, fig_dir=None) -> None:
    # plot_confusion_matrix(model, test_X, test_y, values_format='')
    optimal_threshold = get_optimal_threshold(y=test_y, y_score=predict_prob)
    test_label_optimal = [0 if ele < optimal_threshold else 1 for ele in predict_prob]
    print("*"*20)
    print(f"Optimal threshold: {optimal_threshold}")
    print("*"*20)
    print(classification_report(test_y, test_label_optimal))
    print("*"*20)
    # confusion matrix
    pretty_plot_confusion_matrix(df_cm=pd.DataFrame(confusion_matrix(test_y, test_label_optimal)), fig_dir=fig_dir)
    # auc
    y_test = test_y
    # y_pred = predict_prob
    plot_auc_plot(y_test=y_test, pred_prob=predict_prob, fig_dir=fig_dir)


def get_optimal_threshold(y: pd.DataFrame, y_score: pd.datetime) -> float:
    fpr, tpr, threshold = roc_curve(y_true=y, y_score=y_score)
    objective_func = abs(fpr + tpr - 1)
    idx = np.argmin(objective_func)
    optimal_threshold = threshold[idx]
    return optimal_threshold


def labels(ax, df, xytext=(0, 0)):
    for bar in ax.patches:
        ax.annotate('%{:.2f}\n{:.0f}'.format(100*bar.get_height()/len(df),bar.get_height()), (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                    size=11, xytext=xytext,
                    textcoords='offset points')


def cate_features_plot(df, col, target, target_binary=True, figsize=(20,6), regression=False):
    fig, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)

    plt.subplot(121)
    if target_binary:
        tmp = round(pd.crosstab(df[col], df[target], normalize='index'), 2)
        tmp = tmp.reset_index()
        # tmp.rename(columns={0: 'NoFraud', 1: 'Fraud'}, inplace=True)

    ax[0] = sns.countplot(x=col, data=df, hue=target,
                          order=np.sort(df[col].dropna().unique()),
                          )
    ax[0].tick_params(axis='x', rotation=90)
    labels(ax[0], df[col].dropna(), (0, 0))
    if target_binary:
        ax_twin = ax[0].twinx()
        # sns.set(rc={"lines.linewidth": 0.7})
        ax_twin = sns.pointplot(x=col, y=1, data=tmp, color='black', legend=False,
                                order=np.sort(df[col].dropna().unique()),
                                linewidth=0.1)
    # if regression:
    #     stats = df.groupby(col).agg({target: [np.mean, np.max, np.min]})[target].reset_index()
    #     ax_twin = ax[0].twinx()
    #     ax_twin = sns.pointplot(x=col, y='mean', data=stats,
    #                   color='black', legend='avg_delivery_time',
    #                   order=np.sort(df[col].dropna().unique()),
    #                   linewidth=0.1,
    #                   )
    ax[0].grid()

    plt.subplot(122)
    ax[1] = sns.countplot(x=df[col].dropna(),
                          order=np.sort(df[col].dropna().unique()),
                          )
    ax[1].tick_params(axis='x', rotation=90)
    labels(ax[1], df[col].dropna())
    plt.show()


# import numpy as np
# import pandas as pd
#
# def weighted_mean(x):
#     arr = np.ones((1, x.shape[1]))
#     arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
#     return arr
#
# df = pd.DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
#
# df.rolling(2, method="table", min_periods=0).apply(weighted_mean, raw=True, engine="numba")  # noqa:E501



def value_count_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    stats = pd.DataFrame(df[col].value_counts()).reset_index().rename(columns={'index':col, col: 'sample_num'})
#     if round_data:
    stats['frac'] = stats.apply(lambda row: row['sample_num']/df.shape[0], axis=1)
    stats['cumsum'] = stats['frac'].cumsum()
    return stats

def stats_plot(stats):
    plt.figure(figsize=(25,10))
    plt.plot(stats.index+1, stats['cumsum'])
    plt.xlabel("Number of categories")
    plt.ylabel('Cumulative Fraction')
    plt.title('Number of categories VS Cumulative Fraction')
    plt.grid()
    plt.show()
