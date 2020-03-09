#自己封装简单线性回归模型
import numpy as np
from PlayML.metrics import r2_score

#第一个简单线性模型：不使用向量化
class SimpleLinearRegression1(object):
    #初始化Simple Linear Regression 模型
    def __init__(self):
        self.a_ = None
        self.b_ = None

    #根据训练数据集x_train,y_train训练Simple Linear Regression模型
    def fit(self, x_train, y_train):
        #判断一些条件
        if x_train.ndim != 1:
            raise ValueError("简单线性回归模型只能处理一个特征的样本！")

        if len(x_train) != len(y_train):
            raise ValueError("训练特征的个数要与输出标签值的个数相等！")

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        sum_fenzi = 0.0
        sum_fenmu = 0.0
        for xi, yi in zip(x_train, y_train):
            sum_fenzi += (xi - x_mean) * (yi - y_mean)
            sum_fenmu += (xi - x_mean) ** 2

        self.a_ = sum_fenzi / sum_fenmu
        self.b_ = y_mean - self.a_ * x_mean

        return self.a_, self.b_

    #给定待预测数据集X_predict，返回表示X_predict的结果向量
    def predict(self, X_predict):
        if X_predict.ndim != 1:
            raise ValueError("简单线性回归模型只能处理一个特征的样本！")

        if self.a_ is None or self.b_ is None:
            raise ValueError("predict之前要fit！")
        return np.array([self.__pedict(x_predict) for x_predict in X_predict])

    #给定单个待预测数据x，返回x的预测结果值
    def __pedict(self, x_predict):
        return self.a_ * x_predict + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"

#第二个简单线性模型：使用向量化
class SimpleLinearRegression2(object):
    #初始化Simple Linear Regression 模型
    def __init__(self):
        self.a_ = None
        self.b_ = None

    #根据训练数据集x_train,y_train训练Simple Linear Regression模型
    def fit(self, x_train, y_train):
        #判断一些条件
        if x_train.ndim != 1:
            raise ValueError("简单线性回归模型只能处理一个特征的样本！")

        if len(x_train) != len(y_train):
            raise ValueError("训练特征的个数要与输出标签值的个数相等！")

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        #使用向量化计算，不使用for循环
        sum_fenzi = (x_train - x_mean).dot(y_train - y_mean)
        sum_fenmu = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = sum_fenzi / sum_fenmu
        self.b_ = y_mean - self.a_ * x_mean

        return self.a_, self.b_

    #给定待预测数据集X_predict，返回表示X_predict的结果向量
    def predict(self, X_predict):
        if X_predict.ndim != 1:
            raise ValueError("简单线性回归模型只能处理一个特征的样本！")

        if self.a_ is None or self.b_ is None:
            raise ValueError("predict之前要fit！")
        return np.array([self.__pedict(x_predict) for x_predict in X_predict])

    #给定单个待预测数据x，返回x的预测结果值
    def __pedict(self, x_predict):
        return self.a_ * x_predict + self.b_

    def score(self, x_test, y_test):
        y_predict = self.predict(x_test)
        return r2_score(y_test, y_predict)

    def __repr__(self):
        return "SimpleLinearRegression2()"