# -*- coding:utf-8 -*-
from numpy import *
import operator
from os import listdir


# 转化为1*1024的特征向量
def featurevector(file):
    feature_vector = zeros((1, 1024))
    img_file = open(file)
    for i in range(32):
        line_str = img_file.readline()
        for j in range(32):
            feature_vector[0, 32*i+j] = int(line_str[j])
    return feature_vector


# 分类主体程序，计算欧式距离，选择距离最小的k个，返回k个中出现频率最高的类别
# inX是所要测试的向量
# dataSet是训练样本集，一行对应一个样本。dataSet对应的标签向量为labels
# k是所选的最近邻数目
# classify0(vectorUnderTest, trainingMat, hwLabels, 3)
def classify0(inX, dataSet, labels, k):
    print("所要测试的向量inX : %s" % inX)
    print("训练样本集dataSet : %s" % dataSet)
    print("训练样本集对应的标签向量labels : %s" % labels)
    print("最近邻居数目 : %s" % k)
    dataSetSize = dataSet.shape[0]  # shape[0]得出dataSet的行数，即样本个数
    print("1、训练样本集的行数（即样本个数）: %s" % dataSetSize)
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile(A,(m,n))将数组A作为元素构造m行n列的数组
    print("2、将所要测试的向量作为元素构造以样本个数为行，1列的数组 : %s" % diffMat)
    sqDiffMat = diffMat**2
    print("3、对第2步求2次幂 : %s" % sqDiffMat)
    sqDistances = sqDiffMat.sum(axis=1)  # array.sum(axis=1)按行累加，axis=0为按列累加
    print("对第3步按行累加 : %s" % sqDistances)
    distances = sqDistances**0.5
    print("距离集合 : %s" % distances)
    sortedDistIndicies = distances.argsort()  # array.argsort()，得到每个元素的排序序号
    print("得到每个元素的排序序号 : %s" % sortedDistIndicies)
    classCount={}  # sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get(key,x)从字典中获取key对应的value，没有key的话返回0
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    return sortedClassCount[0][0]


def handwritingClassTest():
    # 加载训练集到大矩阵trainingMat
    hwLabels = []
    trainingFileList = listdir('trainingDigits')  # os模块中的listdir('str')可以读取目录str下的所有文件名，返回一个字符串列表
    print("所有训练集文件 : %s" % trainingFileList)
    m = len(trainingFileList)
    print("所有训练集文件的个数 : %s" % m)
    trainingMat = zeros((m, 1024))
    print("生成大矩阵 : %s" % trainingMat)
    for i in range(m):
        fileNameStr = trainingFileList[i]  # 训练样本的命名格式：1_120.txt
        fileStr = fileNameStr.split('.')[0]  # string.split('str')以字符str为分隔符切片，返回list，这里去list[0],得到类似1_120这样的
        classNumStr = int(fileStr.split('_')[0])  # 以_切片，得到1，即类别
        hwLabels.append(classNumStr)
        trainingMat[i, :] = featurevector('trainingDigits/%s' % fileNameStr)
    print("处理训练集文件名labels : %s" % hwLabels)
    print("将所有训练文件转换为特征向量dataSet : %s" % trainingMat)
        # 逐一读取测试图片，同时将其分类
    testFileList = listdir('testDigits')
    print("测试集 : %s" % testFileList)
    errorCount = 0.0
    mTest = len(testFileList)
    print(" 测试集个数 : %s" % mTest)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = featurevector('testDigits/%s' % fileNameStr)
        print("将每一个测试集文件准换成特征向量inX: %s" % vectorUnderTest)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))
handwritingClassTest()

