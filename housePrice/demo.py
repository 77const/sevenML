#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/14 13:48
# @Author  : Seven
# @Site    : 
# @File    : demo.py
# @Software: PyCharm

# 导入相关数据包
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.model_selection import learning_curve


def readCsv():  # 读取数据
    trainData = pd.read_csv('train.csv')
    testData = pd.read_csv('train.csv')
    # print(trainData.columns)  # 打印数据的ID标签
    # print(trainData.info())  # 打印数据信息，查看缺失值
    # print(trainData.head(5))  # 打印前5行的数据信息
    return trainData, testData


def analyzeData():  # 数据特征分析（统计学与绘图）
    trainData, testData = readCsv()
    trainDataCorr = trainData.drop('Id', axis=1).corr()
    # print(trainDataCorr)
    # 画出相关热力图
    # a = plt.subplots(figsize=(20, 12))  # 调整画布的大小
    # a = sns.heatmap(trainDataCorr, vmax=.8, square=True)  # 画热力图，annot=True 显示系数
    # plt.show()

    # 寻找k个最相关的特征信息
    '''
    1. GarageCars 和 GarageAre 相关性很高、就像双胞胎一样，所以我们只需要其中的一个变量，例如：GarageCars。
    2. TotalBsmtSF  和 1stFloor 与上述情况相同，我们选择 TotalBsmtS
    3. GarageAre 和 TotRmsAbvGrd 与上述情况相同，我们选择 GarageAre
    '''
    # k = 10
    # cols = trainDataCorr.nlargest(k, 'SalePrice')['SalePrice'].index
    # cm = np.corrcoef(trainData[cols].values.T)
    # sns.set(font_scale=1.5)
    # hm = plt.subplots(figsize=(20, 12))
    # hm = sns.heatmap(cm,
    #                  cbar=True,
    #                  annot=True,
    #                  square=True,
    #                  fmt='.2f',
    #                  annot_kws={'size': 10},
    #                  yticklabels=cols.values,
    #                  xticklabels=cols.values)
    # plt.show()
    # SalePrice与相关变量之间的散点图
    # sns.set()
    # cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    # sns.pairplot(trainData[cols], size=2.5)
    # plt.show()
    # print(trainData[['SalePrice', 'OverallQual', 'GrLivArea','GarageCars',
    #                  'TotalBsmtSF', 'FullBath', 'YearBuilt']].info())


def disposeData():  # 特征工程
    trainData, testData = readCsv()
    # 缺失值分析
    # 根据业务, 常识, 以及第二步的数据分析构造特征工程.
    # 将特征转换为模型可以辨别的类型(如处理缺失值, 处理文本进行等)
    # total = trainData.isnull().sum().sort_values(ascending=False)
    # percent = (trainData.isnull().sum()/trainData.isnull().count()).sort_values(ascending=False)
    # missingData = pd.concat([total, percent], axis=1, keys=['total', 'Lost Percent'])
    # print(missingData.head(20))
    '''
    1. 对于缺失率过高的特征，例如 超过15% 我们应该删掉相关变量且假设该变量并不存在
    2. GarageX 变量群的缺失数据量和概率都相同，可以选择一个就行，例如：GarageCars
    3. 对于缺失数据在5%左右（缺失率低），可以直接删除/回归预测
    '''
    # trainData = trainData.drop((missingData[missingData['total'] > 1]).index, axis=1)
    # trainData = trainData.drop(trainData.loc[trainData['Electrical'].isnull()].index)
    # print(trainData.isnull().sum().max())

    # 异常值处理
    # 单因素分析
    # 这里的关键在于如何建立阈值，定义一个观察值为异常值。我们对数据进行正态化，意味着把数据值转换成均值为 0，方差为 1 的数据
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax1.hist(trainData.SalePrice)
    # ax2.hist(np.log1p(trainData.SalePrice))
    # plt.show()
    '''
    从直方图中可以看出：

    * 偏离正态分布
    * 数据正偏
    * 有峰值
    '''
    # 数据偏度和峰度度量：

    # print("Skewness: %f" % trainData['SalePrice'].skew())
    # print("Kurtosis: %f" % trainData['SalePrice'].kurt())

    '''
    低范围的值都比较相似并且在 0 附近分布。
    高范围的值离 0 很远，并且七点几的值远在正常范围之外。
    '''
    # 双变量分析
    # 1.GrLivArea 和 SalePrice 双变量分析

    # var = 'GrLivArea'
    # data = pd.concat([trainData['SalePrice'], trainData[var]], axis=1)
    # data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    # plt.show()
    '''
    从图中可以看出：

    1. 有两个离群的 GrLivArea 值很高的数据，我们可以推测出现这种情况的原因。
        或许他们代表了农业地区，也就解释了低价。 这两个点很明显不能代表典型样例，所以我们将它们定义为异常值并删除。
    2. 图中顶部的两个点是七点几的观测值，他们虽然看起来像特殊情况，但是他们依然符合整体趋势，所以我们将其保留下来。
    '''
    # 删除点
    # trainData.sort_values(by='GrLivArea', ascending=False)[:2]
    # trainData = trainData.drop(trainData[trainData['Id'] == 1299].index)
    # trainData = trainData.drop(trainData[trainData['Id'] == 524].index)
    # plt.show()

    # TotalBsmtSF 和 SalePrice 双变量分析
    # var = 'TotalBsmtSF'
    # data = pd.concat([trainData['SalePrice'], trainData[var]], axis=1)
    # data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
    # plt.show()

    # 正态性
    # 应主要关注以下两点：直方图 – 峰度和偏度。
    # 正态概率图 – 数据分布应紧密跟随代表正态分布的对角线。
    # SalePrice绘制直方图和正态概率图：
    # sns.distplot(trainData['SalePrice'], fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(trainData['SalePrice'], plot=plt)
    # plt.show()

    '''
    可以看出，房价分布不是正态的，显示了峰值，正偏度，但是并不跟随对角线。
    可以用对数变换来解决这个问题
    '''
    # 进行对数变换：
    # trainData['SalePrice'] = np.log(trainData['SalePrice'])
    # # 绘制变换后的直方图和正态概率图：
    #
    # sns.distplot(trainData['SalePrice'], fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(trainData['SalePrice'], plot=plt)
    # plt.show()

    # GrLivArea绘制直方图和正态概率曲线图：
    # sns.distplot(trainData['GrLivArea'], fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(trainData['GrLivArea'], plot=plt)
    # plt.show()

    # 进行对数变换：
    # trainData['GrLivArea'] = np.log(trainData['GrLivArea'])
    # # 绘制变换后的直方图和正态概率图：
    #
    # sns.distplot(trainData['GrLivArea'], fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(trainData['GrLivArea'], plot=plt)
    # plt.show()

    # TotalBsmtSF绘制直方图和正态概率曲线图：
    # sns.distplot(trainData['TotalBsmtSF'], fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(trainData['TotalBsmtSF'], plot=plt)
    # plt.show()
    '''
    从图中可以看出：
    * 显示出了偏度
    * 大量为 0(Y值) 的观察值（没有地下室的房屋）
    * 含 0(Y值) 的数据无法进行对数变换
    '''
    # 去掉为0的分布情况
    # tmp = np.array(trainData.loc[trainData['TotalBsmtSF'] > 0, ['TotalBsmtSF']])[:, 0]
    # sns.distplot(tmp, fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(tmp, plot=plt)
    # plt.show()

    # 我们建立了一个变量，可以得到有没有地下室的影响值（二值变量），我们选择忽略零值，只对非零值进行对数变换。
    # 这样我们既可以变换数据，也不会损失有没有地下室的影响。

    # print(trainData.loc[trainData['TotalBsmtSF'] == 0, ['TotalBsmtSF']].count())
    # trainData.loc[trainData['TotalBsmtSF'] == 0, 'TotalBsmtSF'] = 1
    # print(trainData.loc[trainData['TotalBsmtSF'] == 1, ['TotalBsmtSF']].count())
    #
    # # 进行对数变换：
    # print(trainData['TotalBsmtSF'].head(20))
    # trainData['TotalBsmtSF'] = np.log(trainData['TotalBsmtSF'])
    # print(trainData['TotalBsmtSF'].head(20))
    #
    # # 绘制变换后的直方图和正态概率图：
    #
    # tmp = np.array(trainData.loc[trainData['TotalBsmtSF'] > 0, ['TotalBsmtSF']])[:, 0]
    # sns.distplot(tmp, fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(tmp, plot=plt)
    # plt.show()

    # 同方差性：
    # 最好的测量两个变量的同方差性的方法就是图像。
    # SalePrice和GrLivArea同方差性
    # 绘制散点图
    # plt.scatter(trainData['GrLivArea'], trainData['SalePrice'])
    # plt.show()

    # SalePrice with TotalBsmtSF 同方差性
    # 绘制散点图
    # plt.scatter(trainData[trainData['TotalBsmtSF'] > 0]['TotalBsmtSF'], trainData[trainData['TotalBsmtSF'] > 0]['SalePrice'])
    # plt.show()
    # 可以看出 SalePrice 在整个 TotalBsmtSF 变量范围内显示出了同等级别的变化。


def modelTrain():  # 模型选择
    """
    可选单个模型模型有 线性回归（Ridge、Lasso）、树回归、GBDT、XGBoost、LightGBM 等.
    也可以将多个模型组合起来,进行模型融合,比如voting,stacking等方法
    好的特征决定模型上限,好的模型和参数可以无线逼近上限.
    我测试了多种模型,模型结果最高的随机森林,最高有0.8.
    :return:
    """
    # 数据标准化
    trainData, testData = readCsv()
    x_train = trainData[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]
    y_train = trainData[["SalePrice"]].values.ravel()
    x_test = testData[['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']]

    # ridge = Ridge(alpha=15)
    # # bagging 把很多小的分类器放在一起，每个train随机的一部分数据，然后把它们的最终结果综合起来（多数投票）
    # # bagging 算是一种算法框架
    # params = [1, 10, 15, 20, 25, 30, 40]
    # test_scores = []
    # for param in params:
    #     clf = BaggingRegressor(base_estimator=ridge, n_estimators=param)
    #     # cv=5表示cross_val_score采用的是k-fold cross validation的方法，重复5次交叉验证
    #     # scoring='precision'、scoring='recall'、scoring='f1', scoring='neg_mean_squared_error' 方差值
    #     test_score = np.sqrt(-cross_val_score(clf, x_train, y_train, cv=10, scoring='neg_mean_squared_error'))
    #     test_scores.append(np.mean(test_score))
    #
    # plt.plot(params, test_scores)
    # plt.title('n_estimators vs CV Error')
    # plt.show()

    ridge = Ridge(alpha=15)
    #
    # train_sizes, train_loss, test_loss = learning_curve(ridge, x_train, y_train, cv=10,
    #                                                     scoring='neg_mean_squared_error',
    #                                                     train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1])
    #
    # # 训练误差均值
    # train_loss_mean = -np.mean(train_loss, axis=1)
    # # 测试误差均值
    # test_loss_mean = -np.mean(test_loss, axis=1)
    #
    # # 绘制误差曲线
    # plt.plot(train_sizes / len(x_train), train_loss_mean, 'o-', color='r', label='Training')
    # plt.plot(train_sizes / len(x_train), test_loss_mean, 'o-', color='g', label='Cross-Validation')
    #
    # plt.xlabel('Training data size')
    # plt.ylabel('Loss')
    # plt.legend(loc='best')
    # plt.show()

    mode_br = BaggingRegressor(base_estimator=ridge, n_estimators=25)
    mode_br.fit(x_train, y_train)
    # y_test = np.expm1(mode_br.predict(x_test))
    y_test = mode_br.predict(x_test)
    # 提交结果
    submission_df = pd.DataFrame(data={'Id': x_test.index, 'SalePrice': y_test})
    print(submission_df.head(10))
    submission_df.to_csv('sample_submission.csv', columns=['Id', 'SalePrice'], index=False)


if __name__ == "__main__":
    modelTrain()