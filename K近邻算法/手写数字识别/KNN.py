#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/28 18:36
# @Author  : Seven
# @Site    : 
# @File    : KNN.py
# @Software: PyCharm

from numpy import *          # 导入numpy模块
import operator              # 导入运算符模块
from os import listdir       # 导入os模块中的listdir，listdir读取目录中的文件名


def classify0(inX, dataSet, labels, k):
    """
    ### kNN算法，目的就是为了找最近的距离 ###
    :param inX:
    :param dataSet:
    :param labels:
    :param k:
    :return:
    """
    dataSetSize = dataSet.shape[0]  # 返回行列数 shape[0]是行数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile 复制inX，使其与dataset行数一致
    sqDiffMat = diffMat ** 2  # **表示平方
    sqDistances = sqDiffMat.sum(axis=1)  # 按行将计算结果求和，axis表示轴
    distances = sqDistances ** 0.5  # 计算欧拉公式
    sortedDistIndicies = distances.argsort()  # 使用argsort排序，返回由小到大排列索引值，例如[4,2,5]返回[1,0,2]
    classCount = {}  # 新建一个字典，用于计数，计算结果
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]  # 按照距离给标签依次进行计数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 对字典进行抓取，若没有key值，则加1在赋值给key值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # 返回一个列表，按照第二个元素降序排列
    return sortedClassCount[0][0]  # 返回出现次数最多的那一个label的值

#
# def file2matrix(filename):
#     fr = open(filename)
#     numberOfLines = len(fr.readlines())  # 读取文件的行数
#     returnMat = zeros((numberOfLines, 3))  # 创建文件行数行3列的0矩阵
#     classLabelVector = []
#     fr = open(filename)
#     index = 0
#     for line in fr.readlines():
#         line = line.strip()
#         listFromLine = line.split('\t')  # 删除行前面的空格
#         returnMat[index, :] = listFromLine[0:3]  # 根据分隔符划分
#         classLabelVector.append(int(listFromLine[-1]))  # 取得每一行的内容保存起来
#         index += 1
#     return returnMat, classLabelVector


# def autoNorm(dataSet):
#     """
#     归一化处理
#     :param dataSet:
#     :return:
#     """
#     minVals = dataSet.min(0)  # 找出样本的最小值
#     maxVals = dataSet.max(0)  # 找出样本集中的最大值
#     ranges = maxVals - minVals  # 最大值与最小值的差
#     normDataSet = zeros(shape(dataSet))  # 创建与样本集一样大小的零矩阵
#     m = dataSet.shape[0]
#     normDataSet = dataSet - tile(minVals, (m, 1))  # 样本集中的元素与最小值的差值
#     normDataSet = normDataSet / tile(ranges, (m, 1))  # 数据相除，归一化处理
#     return normDataSet, ranges, minVals


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# def datingClassTest():
#     # 选取多少数据测试分类器
#     hoRatio = 0.1
#     #  从datingTestSet2.txt中获取数据
#     datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#     # 归一化数据
#     normMat, ranges, minVals = autoNorm(datingDataMat)
#     m = normMat.shape[0]
#     # 设置测试个数
#     numTestVecs = int(m*hoRatio)
#     # 记录错误数量
#     errorCount = 0.0
#     for i in range(numTestVecs):
#         # 分类算法
#         classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
#         print("分类器返回了: %d, 正确答案是: %d" % (classifierResult, datingLabels[i]))
#         if classifierResult != datingLabels[i]: errorCount += 1.0
#     # 计算错误率
#     print("错误率是: %f" % (errorCount/float(numTestVecs)))


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # 读取测试数据
    m = len(trainingFileList)  # 读取数据的行数
    trainingMat = zeros((m, 1024))  # 创建对应的0矩阵
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 保存每一个训练的文本
        fileStr = fileNameStr.split('.')[0]  # 取得文本名
        classNumStr = int(fileStr.split('_')[0])  # 分隔文本名
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')  # iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]  # take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("分类器返回了:%d，真正的答案是:%d" % (classifierResult, classNumStr))
        if classifierResult != classNumStr: errorCount += 1.0
    print("\n错误的总数是: %d" % errorCount)
    print("\n错误率是：: %f" % (errorCount / float(mTest)))


if __name__ == "__main__":
    handwritingClassTest()
