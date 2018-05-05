#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/4 13:57
# @Author  : Seven
# @Site    : 
# @File    : knn.py
# @Software: PyCharm
# Github: https://github.com/apachecn/kaggle

import csv
import time
import pandas as pd
from numpy import ravel, shape
from sklearn.neighbors import KNeighborsClassifier


def opencsv():                                                                       # 加载数据
    data = pd.read_csv('datasets/getting-started/digit-recognizer/input/train.csv')  # 使用 pandas 打开训练数据
    data1 = pd.read_csv('datasets/getting-started/digit-recognizer/input/test.csv')  # 使用 pandas 打开测试数据

    train_data = data.values[0:, 1:]  # 读入全部训练数据,  [行，列]
    train_label = data.values[0:, 0]  # 读取列表的第一列
    test_data = data1.values[0:, 0:]  # 测试全部测试个数据
    return train_data, train_label, test_data


def saveResult(result, csvName):                 # 保存数据
    with open(csvName, 'w') as myFile:           # 创建记录输出结果的文件（w 和 wb 使用的时候有问题）
        myWriter = csv.writer(myFile)            # 对文件执行写入
        myWriter.writerow(["ImageId", "Label"])  # 设置表格的列名
        index = 0                                # 创建索引值
        for i in result:                         # 循环遍历预测的结果值
            tmp = []                             # 创建空列表
            index = index + 1                    # 索引值增加
            tmp.append(int(index))               # 往列表中追加当前索引值
            tmp.append(int(i))                   # 往列表中追加测试集的标签值
            myWriter.writerow(tmp)               # 把数据写入到Result_sklearn_knn.csv


def knnClassify(trainData, trainLabel):       # 模型分类训练
    knnClf = KNeighborsClassifier()           # 默认k = 5,由自己定义:KNeighborsClassifier(n_neighbors=10)
    knnClf.fit(trainData, ravel(trainLabel))  # ravel返回一个连续的扁平数组。
    return knnClf


def dRecognition_knn():                          # 基于knn算法的数字识别
    start_time = time.time()                     # 获取当前时间
    trainData, trainLabel, testData = opencsv()  # 加载数据

    # 返回数据的类型和维度
    # print("trainData==>", type(trainData), shape(trainData))
    # print("trainLabel==>", type(trainLabel), shape(trainLabel))
    # print("testData==>", type(testData), shape(testData))

    print("load data finish")                                     # 数据读取完成
    stop_time_l = time.time()                                     # 获取
    print('load data time used:%f' % (stop_time_l - start_time))  # 读取数据所花费的时间

    knnClf = knnClassify(trainData, trainLabel)                   # 模型训练
    testLabel = knnClf.predict(testData)                          # 结果预测

    saveResult(testLabel, 'datasets/getting-started/digit-recognizer/output/Result_sklearn_knn.csv')  # 结果的输出
    print("finish!")                                             # 处理完成
    stop_time_r = time.time()                                    # 获取当前时间
    print('classify time used:%f' % (stop_time_r - start_time))  # 分类训练所花费时间


if __name__ == '__main__':
    dRecognition_knn()
