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
# 生成新特征列：家庭规模
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
print(titanic["FamilySize"])
print("\n---------------------------------以上输出新特征列:家庭规模----------------------------------------\n")
# 用.apply()方法生成新特征列：姓名长度
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
print(titanic["NameLength"])
print("\n---------------------------------以上输出新特征列:姓名长度----------------------------------------\n")

"""
05. 称谓

从乘客的姓名中将称谓如Master., Mr., Mrs.等单独提取出来，有的称谓很常用，有的称谓则只有个别人专享，可能是贵族或授勋。

首先利用正则表达式提取称谓，再映射到某个整数值，之后生成一个新的数值特征列。
"""
# 正则表达式提取特征
import re

# 从姓名中提取称谓的函数
def get_title(name):
    # 正则表达式检索称谓，称谓总以大写字母开头并以句点结尾
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # 如果称谓存在则返回其值
    if title_search:
        return title_search.group(1)
    return ""

# 创建一个新的Series对象titles，统计各个头衔出现的频次
titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))
print("\n---------------------------------以上输出各个头衔出现的频次----------------------------------------\n")
# 将每个称谓映射到一个整数，有些太少见的称谓可以压缩到一个数值
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v

# 验证转换结果
print(pd.value_counts(titles))
print("\n---------------------------------以上输出各个头衔转换成数字后出现的频次----------------------------------------\n")
# Add in the title column.
titanic["Title"] = titles # 增加头衔列表
print(titanic.head(5))
print("\n---------------------------------以上把Title（头衔）加入表中之后的前5行数据----------------------------------------\n")

"""
06. 家庭组

我们可以生成一个新的特征来指示某位乘客属于哪个家庭，因为幸存者很大程度上依靠家人和身边的人互助互救。
“家庭ID”通过连接姓氏和FamilySize生成，再映射成一个整数特征。
"""
import operator

# 映射姓氏到家庭ID的字典
family_id_mapping = {}

# 从行信息提取家庭ID的函数
def get_family_id(row_info):
    # 分割逗号获取姓氏
    last_name = row_info["Name"].split(",")[0]
    # 创建家庭ID列表
    family_id = "{0}{1}".format(last_name, row_info["FamilySize"])
    # 从映射中查询ID
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # 遇到新的家庭则将其ID设为当前最大ID+1
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

# 用.apply()方法获得家庭ID
family_ids = titanic.apply(get_family_id, axis=1)
print(family_ids)
print("\n---------------------------------以上输出家庭id----------------------------------------\n")
# 家庭数量过多，所以将所有人数小于3的家庭压缩成一类
family_ids[titanic["FamilySize"] < 3] = -1

# 输出每个家庭ID的数量
print(pd.value_counts(family_ids))
print("\n---------------------------------以上输出每一类家庭的id的个数----------------------------------------\n")
titanic["FamilyId"] = family_ids
print(titanic.head(8))
print("\n---------------------------------以上把家庭id加入表中之后的前8行数据----------------------------------------\n")

"""
07. 最佳特征

所有机器学习任务中，最重要的是特征工程，目前的案例中就尚有很多特征待挖掘。我们还需要从特征中遴选出哪些是最有用的，
比如单变量特征提取，就是逐列计算找出哪一列与预测目标(Survived)相关性最高。

Scikit-learn提供选择特征用的函数SelectKBest，自动从数据中提取最佳特征。
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

# 特征选择
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])
# 得到每个特征列的p值，再转换为交叉验证得分
scores = -np.log10(selector.pvalues_)
print(scores)
print("\n---------------------------------以上是每个特征列的p值，转换为交叉验证得分---------------------------------------\n")
# 绘制得分图像，观察哪个特征是最好的
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

# 只选取最好的四个特征
predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)

# 计算交叉验证得分
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
print(scores)
print("\n---------------------------------计算交叉验证得分--------------------------------------\n")
# 输出交叉验证得分的均值
print(scores.mean())
print("\n---------------------------------计算交叉验证得分的平均值---------------------------------------\n")

"""
08. 梯度提升(Gradient Boosting)

梯度提升是另一种基于决策树的分类器，与随机森林不同，梯度提升中的决策树是一个接一个生成的，并将上一个树的误差迭代到下一个树当中。
所以每个决策树都是建立在之前所有的树上，如果树太多的话，这会导致过拟合，在本例中我们限制树的数量上限为25。

另一预防过拟合的策略是在梯度提升中限制每个树的深度，这里限制为3层。

09. 整合

为了改进预测精度可以尝试整合不同的分类器，也就是对多个分类器的拟合结果取平均值。一般来说，整合的模型的差异化越大，预测精度越高。
差异化代表模型根据不同的特征列或不同的算法产生预测结果。整合随机森林和决策树大概起不到太大作用，因为它们太接近了，
但是线性回归和随机森林的整合结果就好得多。需要注意的是，参与整合的分类器首先需要本身就有差不多的精度，如果其中一个分类器太差，
最终结果可能还不如不整合。

针对本题，我们整合一个逻辑回归和梯度提升分类器，其中逻辑回归采用最线性的特征，梯度提升则面向所有特征。
两个分类器的原始输出（0到1的概率）首先求平均值，再把高于0.5的映射为1；低于或等于0.5的映射为0。
"""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3),
     ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

# 初始化交叉验证
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_target = titanic["Survived"].iloc[train]
    full_test_predictions = []
    # 对每个交叉验证分组，分别使用两种算法进行分类
    for alg, predictors in algorithms:
        # 用训练集拟合算法
        alg.fit(titanic[predictors].iloc[train,:], train_target)
        # 选择并预测测试集上的输出
        # .astype(float) 可以把dataframe转换为浮点数类型
        test_predictions = alg.predict_proba(titanic[predictors].iloc[test,:].astype(float))[:,1]
        full_test_predictions.append(test_predictions)
    # 对两个预测结果取平均值
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    # 大于0.5的映射为1；小于或等于的映射为0
    test_predictions[test_predictions <= .5] = 0
    test_predictions[test_predictions > .5] = 1
    predictions.append(test_predictions)

# 将预测结果存入一个数组
predictions = np.concatenate(predictions, axis=0)
print(predictions)
# 与训练集比较以计算精度
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print(accuracy)

"""
10. 测试集数据清洗
"""
titanic_test = pd.read_csv("http://jizhi-10061919.file.myqcloud.com/kaggle_sklearn/titanic_test.csv")
# 添加“称谓”列
titles = titanic_test["Name"].apply(get_title)

# 在映射字典里添加“Dona”这个称谓，因为训练集里没有
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
for k,v in title_mapping.items():
    titles[titles == k] = v
titanic_test["Title"] = titles

# 检查各个称谓的数量
print(pd.value_counts(titanic_test["Title"]))

# 添加“家庭规模”列
titanic_test["FamilySize"] = titanic_test["SibSp"] + titanic_test["Parch"]

# 添加“家庭ID”列
print(family_id_mapping)

family_ids = titanic_test.apply(get_family_id, axis=1)
family_ids[titanic_test["FamilySize"] < 3] = -1
titanic_test["FamilyId"] = family_ids

# 添加“姓名长度”列
titanic_test["NameLength"] = titanic_test["Name"].apply(lambda x: len(x))

"""
11. 测试集机器学习
"""
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]

algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]

full_predictions = []
for alg, predictors in algorithms:
    # 用训练集拟合模型
    alg.fit(titanic[predictors], titanic["Survived"])
    # 将所有数据转换为浮点数，用测试集做预测
    predictions = alg.predict_proba(titanic_test[predictors].astype(float))[:,1]
    full_predictions.append(predictions)

# 梯度提升的预测效果更好，所以赋予更高的权重
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4

# 将predictions转换为0/1：小于或等于0.5 -> 0；大于0.5 -> 1
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
# 将predicitons全部转换为整数类型
predictions = predictions.astype(int)
# 生成新的DataFrame对象submission，内含"PassengerId"和"Survived"两列
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })