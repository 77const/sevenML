#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/5/5 12:44
# @Author  : Seven
# @Site    : 
# @File    : knn_v2.py
# @Software: PyCharm
import time
from numpy import *
import operator
import csv


def toInt(array):        # 将字符串转换成整数
    array = mat(array)   # 将数据解释为矩阵
    m, n = shape(array)  # 获取矩阵的行和列
    newArray = zeros((m, n))  # 生成对应的0矩阵
    for i in range(m):
        for j in range(n):
            newArray[i, j] = int(array[i, j])  # 把对应得int数据赋值到新矩阵
    return newArray


def nomalizing(array):  # 归一化算法
    """
    将所有非0的数字，即1～255都归一化为1
    :param array:
    :return:
    """
    m, n = shape(array)
    for i in range(m):
        for j in range(n):
            if array[i, j] != 0:
                array[i, j] = 1
    return array


def loadTrainData():
    """
    train.csv是训练样本集，大小42001*785，第一行是文字描述，所以实际的样本数据大小是42000*785，
    其中第一列的每一个数字是它对应行的label，可以将第一列单独取出来，得到42000*1的向量trainLabel，
    剩下的就是42000*784的特征向量集trainData，所以从train.csv可以获取两个矩阵trainLabel、trainData。
    :return:
    """
    trainData = []
    with open('datasets/getting-started/digit-recognizer/input/train.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            trainData.append(line)  # 生成一个数据42001*785
    trainData.remove(trainData[0])  # 移除第一行文字描述
    trainData = array(trainData)    # 转换成对应矩阵
    label = trainData[:, 0]         # 取每一行的第一个数据42000*1
    data = trainData[:, 1:]         # 取得特征向量集42000*784
    return nomalizing(toInt(data)), toInt(label)  # label 1*42000  data 42000*784
    # return data,label


def loadTestData():
    """
    test.csv里的数据大小是28001*784，第一行是文字描述，因此实际的测试数据样本是28000*784，与train.csv不同，
    没有label，28000*784即28000个测试样本，我们要做的工作就是为这28000个测试样本找出正确的label。
    所以从test.csv我们可以得到测试样本集testData
    :return:
    """
    testdata = []
    with open('datasets/getting-started/digit-recognizer/input/test.csv') as file:
        lines = csv.reader(file)
        for line in lines:
            testdata.append(line)
    # 28001*784
    testdata.remove(testdata[0])  # 去除文字描述一行
    data = array(testdata)
    return nomalizing(toInt(data))  # data 28000*784


def loadTestResult():  # 数据集结果验证
    l=[]
    with open('datasets/getting-started/digit-recognizer/input/sample_submission.csv') as file:
         lines=csv.reader(file)
         for line in lines:
             l.append(line)
    # 28001*2
    l.remove(l[0])
    label=array(l)
    return toInt(label[:, 1])  # label 28000*1


# dataSet:m*n   labels:m*1  inX:1*n
def classify(inX, dataSet, labels, k):
    """
    ### kNN算法，目的就是为了找最近的距离 ###
    :param inX:输入的单个样本，是一个特征向量
    :param dataSet:训练样本trainData
    :param labels:标签向量trainLabel
    :param k:是knn算法选定的k，一般选择0～20之间的数字
    :return:
    """
    inX = mat(inX)  # 把数据矩阵化
    dataSet = mat(dataSet)
    labels = mat(labels)
    dataSetSize = dataSet.shape[0]                   # shape返回行列数 shape[0]是行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile 复制inX，使其与dataset行数一致
    sqDiffMat = array(diffMat) ** 2                  # **表示平方
    sqDistances = sqDiffMat.sum(axis=1)              # 按行将计算结果求和，axis表示轴
    distances = sqDistances ** 0.5                   # 计算欧拉公式
    sortedDistIndicies = distances.argsort()         # 使用argsort排序，返回由小到大排列索引值，例如[4,2,5]返回[1,0,2]
    classCount = {}                                  # 新建一个字典，用于计数，记录结果
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i], 0]  # 按照距离给标签依次进行计数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 对字典进行抓取，若没有key值，则加1在赋值给key值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 返回一个列表，按照第二个元素降序排列
    return sortedClassCount[0][0]  # 返回出现次数最多的那一个label的值


def saveResult(result):  # 保存训练的数据
    with open('datasets/getting-started/digit-recognizer/output/Result_seven_knn.csv', 'w') as myFile:
        myWriter = csv.writer(myFile)
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)


def handwritingClassTest():
    start_time = time.time()  # 获取当前时间
    trainData, trainLabel = loadTrainData()
    testData = loadTestData()
    testLabel = loadTestResult()
    print("load data finish")  # 数据读取完成
    stop_time_l = time.time()  # 获取
    print('load data time used:%f' % (stop_time_l - start_time))  # 读取数据所花费的时间

    m, n = shape(testData)
    errorCount = 0
    resultList = []
    for i in range(m):
        classifierResult = classify(testData[i], trainData, trainLabel.transpose(), 5)  # 模型训练
        resultList.append(classifierResult)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, testLabel[0, i]))
        if classifierResult != testLabel[0, i]:
            errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(m)))
    saveResult(resultList)
    stop_time_r = time.time()  # 获取当前时间
    print('classify time used:%f' % (stop_time_r - start_time))  # 分类训练所花费时间


if __name__=="__main__":
    handwritingClassTest()
