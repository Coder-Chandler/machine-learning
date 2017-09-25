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
        # # print(read_line)
        for j in range(32):
            # # print("i:%s" % i)
            # # print("j:%s" % j)
            feature_vector[0, 32*i+j] = int(read_line[j])
            # # print(feature_vector)
    return feature_vector


# file = "/Users/chandler/Documents/Projects/machine-learning/data_analysis/analysis/trainingfile/3_9.txt"
# # print(featurevector(file))


# 转换成特征向量之后测试转换后的图片
def test_featurevector(feature_vector):
    for value in feature_vector:
        for count in range(32):
            test = list(value[32*count:32*count+32])
            # print(test)
# file = "/Users/chandler/Documents/Projects/machine-learning/data_analysis/analysis/trainingfile/3_9.txt"
# test_featurevector(featurevector(file))


# 欧氏距离公式
def euclidean_distance(x, y):
    # 首先求平方
    sq = (x - y)**2
    # 再对平方结果求和并且开方
    distance = (sq.sum(axis=1))**0.5
    return distance


def classify(testfile_matrix, trainingfile_matrix, handwriting_labels, k):
    # print("所要测试的向量testfile_matrix : %s" % testfile_matrix)
    # print("训练样本集trainingfile_matrix : %s" % trainingfile_matrix)
    # print("训练样本集对应的文件标签（0, 1, 2, 3.....9）handwriting_Labels : %s" % handwriting_labels)
    # print("最近邻居数目 : %s" % k)
    trainingfile_matrix_size = trainingfile_matrix.shape[0]
    # print("1、训练样本集的行数（即样本个数）: %s" % trainingfile_matrix_size)
    testfile_array = tile(testfile_matrix, (trainingfile_matrix_size, 1))
    # print("2、将所要测试的向量作为元素构造以样本个数为行，1列的数组testfile_array : %s" % testfile_array)
    testfile_trainingfile_diff = testfile_array - trainingfile_matrix
    # print("3、将数组testfile_array减去训练样本集矩阵数组trainingfile_matrix : %s" % testfile_trainingfile_diff)
    trainingfile_matrix_array_square = testfile_trainingfile_diff**2
    # print("4、对第2步求2次幂 : %s" % trainingfile_matrix_array_square)
    # array.sum(axis=1)按行累加，axis=0为按列累加
    trainingfile_matrix_array_sum = trainingfile_matrix_array_square.sum(axis=1)
    # print("5、对第3步的每一行单独求和 : %s" % trainingfile_matrix_array_sum)
    euclidean_distance = trainingfile_matrix_array_sum**0.5
    # print("6、对第4步开根号得到距离集合 : %s" % euclidean_distance)
    # array.argsort()，得到每个元素的排序序号
    sort_euclidean_distance = euclidean_distance.argsort()
    # print("得到每个元素的排序序号集合（距离按照从小到大排序） : %s" % sort_euclidean_distance)
    knn_dict={}
    for i in range(k):
        """
        k若等于3，那么就拿出sort_euclidean_distance的前三个下标元素（距离最小的前三个元素）
        并且按照这前三个下标找到在handwriting_Labels中对应下标的手写数字没，这个数字就是NearestNeighbor，最近邻居，当然一共有3个
        """
        nearest_neighbor = handwriting_labels[sort_euclidean_distance[i]]
        # print("NearestNeighbor : %s" % nearest_neighbor)
        """
        KNN_dict就是用来计数的，三个邻居谁出现频率高就确定测试数字是和哪个邻居一个类别，即同一个数字
        """
        knn_dict[nearest_neighbor] = knn_dict.get(nearest_neighbor, 0) + 1
        # print("KNN_dict : %s" % knn_dict)
    # sorted()函数，按照第二个元素即value的次序逆向（reverse=True）排序
    sorted_knn_dict = sorted(knn_dict.items(), key=operator.itemgetter(1), reverse=True)
    # print("最近邻居出现频率sorted_KNN_dict : %s" % sorted_knn_dict)
    return sorted_knn_dict[0][0]


def handwriting_guess():
    # 加载训练集到矩阵中
    handwriting_labels = []
    # os模块中的listdir('str')可以读取目录str下的所有文件名，返回一个字符串列表
    training_file_list = listdir('trainingfile')
    # print("所有训练集文件 : %s" % training_file_list)
    # 矩阵的行数就是trainingfile文件的个数
    row = len(training_file_list)
    # print("所有训练集文件的个数 : %s" % row)
    trainingfile_matrix = zeros((row, 1024))
    # print("生成训练集矩阵 : %s" % trainingfile_matrix)

    for i in range(row):
        # 遍历训练集中所有训练文件：1_120.txt
        training_file_name = training_file_list[i]
        # 去除.txt后缀, 1_120.txt --> 1_120
        training_file = training_file_name.split('.')[0]
        # 以_切片，得到1，1_120 --> 1
        training_digit = int(training_file.split('_')[0])
        # 把处理好的文件名放进列表handwriting_labels中待后用
        handwriting_labels.append(training_digit)
        # 将遍历的每一个训练文件转换为特征向量
        trainingfile_matrix[i, :] = featurevector('trainingfile/%s' % training_file_name)
    # print("处理训练集文件名handwriting_Labels : %s" % handwriting_labels)
    # print("将所有训练文件转换为特征向量trainingfile_matrix : %s" % trainingfile_matrix)

    # 逐一读取测试图片，同时将其分类
    testfile_list = listdir('testfile')
    # print("测试集 : %s" % testfile_list)
    error_count = 0.0
    testfile_nums = len(testfile_list)
    # print(" 测试集个数 : %s" % testfile_nums)

    for i in range(testfile_nums):
        testfile_name = testfile_list[i]
        testfile = testfile_name.split('.')[0]
        test_digit = int(testfile.split('_')[0])
        # 将遍历的每一个测试文件转换为特征向量
        testfile_matrix = featurevector('testfile/%s' % testfile_name)
        # print("将每一个测试集文件准换成特征向量testfile_matrix: %s" % testfile_matrix)
        knn_result = classify(testfile_matrix, trainingfile_matrix, handwriting_labels, 3)
        print("KNN识别的结果: %d, 真实的结果: %d" % (knn_result, test_digit))
        if knn_result != test_digit:
            error_count += 1.0
    print("\n识别出错数量: %d" % error_count)
    print("\n识别错误率: %f" % (error_count / float(testfile_nums)))
# 运行KNN算法识别手写数字
handwriting_guess()



