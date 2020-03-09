import numpy as np
from math import sqrt

#分类的评价标准
def accuracy_score(y_true, y_predict):
    if y_true.shape[0] != y_predict.shape[0]:
        raise ValueError("真实值与预测值的size要相等！")

    return sum(y_true == y_predict) / len(y_true)

#计算y_true和y_predict之间的MSE
def mean_squared_error(y_true, y_predict):
    assert len(y_true) == len(y_predict), \
        "the size of y_true must be equal to the size of y_predict"

    mse_score = np.sum((y_true - y_predict) ** 2) / len(y_true)
    return mse_score

#计算y_true和y_predict之间的RMSE
def root_mean_squared_error(y_true, y_predict):
    rmse_score = sqrt(mean_squared_error(y_true, y_predict))
    return rmse_score

def mean_absolute_error(y_true, y_predict):
    mae_score = np.sum(np.absolute(y_predict- y_true)) / len(y_true)
    return mae_score

#计算y_true和y_predict之间的R Square
def r2_score(y_true, y_predict):
    r2_score = 1 - mean_squared_error(y_true, y_predict) / np.var(y_true)
    return r2_score