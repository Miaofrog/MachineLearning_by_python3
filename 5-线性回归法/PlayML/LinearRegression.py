import numpy as np
from PlayML.metrics import r2_score

#自己底层实现多元线性回归算法
class LinearRegression(object):

    #初始化Linear Regression模型
    def __init__(self):
        self.coef_ = None   #特征系数1-n
        self.intercept_ = None #截距，表示偏移量
        self._theta = None  #theta = (intercept_, coef_)

    #根据训练数据集X_train, y_train训练Linear Regression模型
    def fit_normal(self, X_train, y_train):
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train"

        X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
        self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        return self

    #给定待预测数据集X_predict，返回表示X_predict的结果向量
    def predict(self, X_predict):
        assert self.intercept_ is not None and self.coef_ is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == len(self.coef_), \
            "the feature number of X_predict must be equal to X_train"

        X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
        y_predict = X_b.dot(self._theta)
        return y_predict

    def score(self, X_test, y_test):
        y_predict = self.predict(X_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "LinearRegression()"