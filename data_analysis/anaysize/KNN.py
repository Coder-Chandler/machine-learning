# -*- coding:utf-8 -*-
from numpy import *
import operator
from os import listdir


# 转化为1*1024的特征向量
def featurevector(file):
    feature_vector = zeros((1, 1024))
    img_file = open(file)
    for i in range(32):
        read_line = img_file.readline()
        # print(read_line)
        for j in range(32):
            # print("i:%s" % i)
            # print("j:%s" % j)
            feature_vector[0, 32*i+j] = int(read_line[j])
            # print(feature_vector)
    return feature_vector


file = "/Users/chandler/Documents/Projects/machine-learning/data_analysis/anaysize/trainingDigits/3_9.txt"
print(featurevector(file))


# 转换成特征向量之后测试转换后的图片
def test_featurevector(feature_vector):
    for value in feature_vector:
        for count in range(32):
            test = list(value[32*count:32*count+32])
            print(test)
# test_featurevector(featurevector(file))


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
    print("4、对第3步按行累加 : %s" % sqDistances)
    distances = sqDistances**0.5
    print("5、对第4步开根号得到距离集合 : %s" % distances)
    sortedDistIndicies = distances.argsort()  # array.argsort()，得到每个元素的排序序号
    print("得到每个元素的排序序号（距离按照从小到大排序） : %s" % sortedDistIndicies)
    classCount={}  # sortedDistIndicies[0]表示排序后排在第一个的那个数在原来数组中的下标
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        print("voteIlabel : %s" % voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # get(key,x)从字典中获取key对应的value，没有key的话返回0
        print("classCount : %s" % classCount)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  # sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    print("sortedClassCount : %s" % sortedClassCount)
    return sortedClassCount[0][0]


def HandWriting():
    # 加载训练集到大矩阵trainingMat
    hwLabels = []
    training_file_list = listdir('trainingDigits')  # os模块中的listdir('str')可以读取目录str下的所有文件名，返回一个字符串列表
    print("所有训练集文件 : %s" % training_file_list)
    row = len(training_file_list)  # 矩阵的行数就是trainingDigits文件的个数
    print("所有训练集文件的个数 : %s" % row)
    trainingdigits_matrix = zeros((row, 1024))
    print("生成训练集大矩阵 : %s" % trainingdigits_matrix)
    for i in range(row):
        training_file_name = training_file_list[i]  # 遍历训练集中所有训练文件：1_120.txt
        training_file = training_file_name.split('.')[0]  # 去除.txt后缀, 1_120.txt --> 1_120
        training_digit = int(training_file.split('_')[0])  # 以_切片，得到1，1_120 --> 1
        hwLabels.append(training_digit)
        trainingdigits_matrix[i, :] = featurevector('trainingDigits/%s' % training_file_name)  # 将遍历的每一个训练文件转换为特征向量
    print("处理训练集文件名labels : %s" % hwLabels)
    print("将所有训练文件转换为特征向量dataSet : %s" % trainingdigits_matrix)

    # 逐一读取测试图片，同时将其分类
    testfile_list = listdir('testDigits')
    print("测试集 : %s" % testfile_list)
    error_count = 0.0
    testfile_nums = len(testfile_list)
    print(" 测试集个数 : %s" % testfile_nums)
    for i in range(testfile_nums):
        testfile_name = testfile_list[i]
        testfile = testfile_name.split('.')[0]
        test_digit = int(testfile.split('_')[0])
        vectorUnderTest = featurevector('testDigits/%s' % testfile_name)
        print("将每一个测试集文件准换成特征向量inX: %s" % vectorUnderTest)
        classifierResult = classify0(vectorUnderTest, trainingdigits_matrix, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, test_digit))
        if (classifierResult != test_digit): error_count += 1.0
    print("\nthe total number of errors is: %d" % error_count)
    print("\nthe total error rate is: %f" % (error_count / float(testfile_nums)))





