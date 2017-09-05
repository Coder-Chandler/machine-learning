# _*_ coding:utf-8 _*_

# 泰坦尼克号事故分析 03 进阶模型

"""
01. 随机森林简介

随机森林(Random Forest)的基本单元是决策树(Decision Tree)，决策树可以提炼出数据中的非线性趋势，因此可用于分类和回归。

Age	Sex	Survived
38	0	1
26	0	1
28	1	1
35	1	0
很明显Age和Survied之间没有明确的线性关系，35岁的不幸罹难，26和38岁的却幸存了下来。回归可能行不通的情况下，
改用决策树来挖掘Age,Sex和Survived之间的关系：从根节点(Root)出发，持续分叉直至形成叶节点(Leave)。

初始分叉。Sex为0的行向左，为1的向右。
根节点左边的全部存活，故设为叶节点并为Survived赋值1。
右边分组输出不一，所以继续基于Age列分叉，得到两个叶节点。
使用此决策树判断一个新行：

Age	Sex	Survived
22	1	?
依上例，首先划分到右边然后到左边，预测结果是幸存(Survived=1)。
但如果我们代入训练集第一个样本(Age=22, Sex=1, Survived=0)会发现这个预测并不对，这是因为训练集过小所致。

决策树有一个很大的问题，就是对训练数据的过拟合(Overfit)。因为通过分叉，我们创建了一颗非常“深”的决策树，
最终得到的规律受训练数据中奇点（噪音）的影响，不能一般化地推广到新数据集。

随机森林算法应运而生，随机森林包括成百上千的决策树，这些树的输入数据和分叉依据都经过一定的随机化处理。
每个决策树对应的都是训练集的随机一部分，每次分叉也是基于某列的一部分数据。最后对所有树取平均，把过拟合的风险降至最低。
"""
# 使用pandas库来读取.csv文件




"""
随机森林实现之前还是要把数据清洗一下，和前面的titanic_disaster_test一样
"""
import pandas as pd

# 创建pandas dataframe对象并赋值予变量titanic
titanic = pd.read_csv("/Users/yinchuchu/Desktop/Data/titanic/train.csv")

# 输出dataframe的前5行
print(titanic.head(5))

print("\n-----------------------------------以上输出dataframe的前5行-------------------------------------------\n")

# 输出dataframe的描述信息
print(titanic.describe())

print("\n-----------------------------------以上输出dataframe的描述信息-----------------------------------------\n")

# Dataframe对象titanic已准备就绪
# 调用.median()属性获取中位数
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe()) # 查看用中位数填充age之后的dataframe描述信息

print("\n----------------------------以上中位数填充age之后的dataframe描述信息--------------------------\n")

# 将Sex一列中的female替换为0
# 输出Sex所有数据
print(titanic["Sex"].unique())

print("\n---------------------------------以上输出Sex所有数据-----------------------------------------\n")

# 将male替换为1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 1

# 将female替换为0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0
# 查看用0，1替换male,female之后的dataframe描述信息
print(titanic.head(5))# 查看用0，1替换male,female之后的dataframe前5行展示
print("\n-------------------------以上用0，1替换male,female之后的dataframe前5行展示-----------------------------\n")

# 输出"Embarked"的所有数据
print(titanic["Embarked"].unique())

print("\n---------------------------------以上输出Embarked所有数据-----------------------------------------\n")

# 首先把所有缺失值替换为"S"
titanic["Embarked"] = titanic["Embarked"].fillna("S")

# 将"S"替换为0
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
# 将"C"替换为1
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
# 将"Q"替换为2
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
print(titanic.head(10))# 查看用0，1，2替换S,C,Q之后的dataframe前5行展示
print("\n-------------------------以上用0，1，2替换S,C,Q之后的dataframe前10行展示-----------------------------\n")





"""
02. 随机森林实现

训练数据集已导入并存为变量titanic。

使用titanic[predictors]预测titanic["Survived"]，并将交叉验证得分赋值予scores。
创建KFold对象，指定n_folds=3。
使用cross_validation.cross_val_score()，将KFold对象传递给参数cv。 请按注释提示，补完整个程序
"""
# 随机森林与交叉验证
import pandas as pd
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# n_estimators：决策树的数量
# min_samples_split：单个分叉包含的最小行数
# min_samples_leaf：单个叶节点的最小样本数量
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)

# 计算交叉验证的得分
kf = cross_validation.KFold(titanic.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# 输出得分的平均值
print(scores.mean())
print("\n---------------------------------以上输出随机森林预测值-----------------------------------------\n")
"""
03 参数调教

为提高随机森林预测精度，虽简单的办法就是增加决策树的数量。调整min_samples_split和min_samples_leaf两个参数也可以减少过拟合。
因为数量不足的决策树会把训练集的噪音也拟合进模型中，从而生成过深的分支，而这不能代表真实的数据特征。
所以提高min_samples_split和min_samples_leaf有助于让算法适应新数据，但是在训练集上的得分会降低。

重新训练随机森林模型，保持random_state=1，使随机森林有50棵树，每个分叉至少含有4行，每个叶节点至少含有2个样本。
"""
# 随机森林的参数调整
# 创建随机森林分类器对象alg
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)

# 计算交叉验证得分
kf = cross_validation.KFold(titanic.shape[0], 3, random_state=1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)

# 输出得分平均值
print(scores.mean())
print("\n---------------------------------以上输出优化后的随机森林预测值----------------------------------------\n")

"""
04. 提取新特征

之前的predictors都是已有特征的子集，为了更好的分类效果，可以尝试生成新的特征。比如：

名字的长度：可能表征了这个人所处的阶层，与他在泰坦尼克号上的位置也有关
船上家人的总数(SibSp+Parch)
DataFrame对象的.apply()方法可以轻松地实现这些功能，将自定义的函数应用到Dataframe或Series的每一个元素上。
而lambda函数则可以不换行地自定义函数。为提取姓名长度，这个lambda函数定义为lambda(x):len(x)，其中x就是乘客的姓名。
:右边的函数将应用到x上并返回输出结果。.apply方法会将所有输出结果生成一个新的列。

"""