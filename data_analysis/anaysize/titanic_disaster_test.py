# _*_ coding:utf-8 _*_


"""
从这里下载原始数据:https://www.kaggle.com/c/titanic/data

数据的每一行代表一位乘客及其信息，下面逐列考察：
PassengerId	每个乘客的数字ID
Survived	该乘客幸存(1)与否(0)，也是推测的目标
Pclass	乘客所在客舱等级：一等舱(1)，二等舱(2)，三等舱
Name	姓名
Sex	性别
Age	年龄
SibSp	该乘客在船上的配偶/同辈亲属人数
Parch	该乘客在船上的父母/子女人数
Ticket	船票编号
Fare	票价
Cabin	所在客舱
Embarked	上船地点

在定量计算之前，最好先定性分析一下：那些变量从逻辑上会影响幸存率？

已知妇女儿童幸存率更高，所以Age和Sex将是重要推测因子。乘客所属客舱等级Pclass也应该颇有影响，因为一等舱离甲板更近。
而与客舱直接相关的就是票价Fare，也会透露些许信息。同行人数（父母子女Parch或同辈Sibsp）的影响可能是双重的——或者救你的人更多，
或者拖累你的人更多。而有些变量大概无甚相干，比如Embarked，Ticket，Name。

在机器学习中，与上一步类似的工作至关重要。因为任何算法终究要合乎逻辑，是解决具体问题的辅助手段，相关领域知识储备才是最终效用的保证。
比如要做商业数据分析，首先也要懂商务的逻辑才行。
"""
print("泰坦尼克号事故分析 01 数据分析\n-------------->>>>>>>")
"""
泰坦尼克号事故分析 01 数据分析
01. 分析数据

我们将主要使用Python3和pandas，scikit-learn这两个扩展库来分析数据。

下面来更进一步地了解数据，可以使用pandas.describe()方法：

"""
# 使用pandas库来读取.csv文件
import pandas as pd

# 创建pandas dataframe对象并赋值予变量titanic
titanic = pd.read_csv("/Users/yinchuchu/Desktop/Data/titanic/train.csv")

# 输出dataframe的前5行
print(titanic.head(5))

print("\n-----------------------------------以上输出dataframe的前5行--------------------------------------------\n")

# 输出dataframe的描述信息
print(titanic.describe())

print("\n-----------------------------------以上输出dataframe的描述信息------------------------------------------\n")

"""
02.缺失数据

在上一题中，当使用datafram对象titanic的.describe() 时，你或许已经注意到，其他大部分列都有891个元素，而年龄列Age却只有714个元素。
这说明该列存在缺失数据，因为null，NA，not a number都不统计在内。

数据还需要进一步清洗，但既不能暴力去除所有缺失数据的行，因为这些行含有的数据对训练算法仍有帮助；
更不能消除整列，如前所述，年龄信息对这个问题还是很重要的。

清洗数据的策略有很多，比较简单的一种是用全列的中位数填充。选中dataframe中的一列与字典操作相似：
titanic["Age"]
再对该列调用.fillna()方法替换件缺失值。.fillna()的输入值是用来填充的值，比如用填充中位数：
titanic["Age"].fillna(titanic["Age"])
"""
# Dataframe对象titanic已准备就绪
# 调用.median()属性获取中位数
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
print(titanic.describe()) # 查看用中位数填充age之后的dataframe描述信息

"""
03. 非数值列

在之前使用.describe()方法时，只有数值形式的列显示出来了。有的列并非数值形式（比如字符，布尔值等），
这些列无法直接应用在机器学习当中，要么剔除掉（Name, Sex, Cabin, Embarked, Ticket），或者想法将其转换为数值列。

Ticket, Cabin和Name这几列可以忽略。Cabin一列的大部分数值都有缺失（总共只有204条数据），
并且逻辑上与幸存率关系可能不大。Ticket和Name也作用甚微，除非有其他额外信息，比如姓名和家族背景的直接映射关系。
"""

"""
04.性转

Sex列不是数值格式的，但性别信息关系重大，所以必须转换为数值列才能应用到机器学习算法中。
首先需要确定这一列总共出现了多少种数据，虽然一般情况下只有男和女，但是还存在数据缺失、填写错误，
在一些特别强调政治正确的地方甚至还有其他性别的选项。（视male为1，female为2。）

首先选中Sex这一列中所有值为male的行，再赋值为1
titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
注意： 因为之前已经对dataframe对象进行了修改，如果重新刷新页面，请先将之前的程序再执行一次（下同）。
"""

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

"""
05.转换“登船地”列

对Embarked列做与Sex类似的处理。这一列的值有S,C,Q和missing(nan)，每个字母是一个地名的缩写。

令S=0， C=1， Q=2：
"""
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
print(titanic.head(40))# 查看用0，1，2替换S,C,Q之后的dataframe前5行展示
print("\n-------------------------以上用0，1，2替换S,C,Q之后的dataframe前40行展示-----------------------------\n")


print("\n泰坦尼克号事故分析 02 机器学习\n--------------->>>>>>")
"""
泰坦尼克号事故分析 02 机器学习

01. 交叉验证

如果想要避免过拟合(Overfitting)，很重要的一点就是要在不同的数据上进行算法训练。过拟合就是一个模型把“噪音”也当作信号统计在内。
而且任何数据集都不可能完美地涵盖所有情况，总会存在一些偏差甚至坏点。比如要根据一辆车的功率或其他参数来推测其最高时速，
但是如果训练集全是顶级超跑和专业赛车，那么训练出的模型势必会高估普通车的速度。解决办法就是用训练集以外的数据对模型进行评估。

所有机器学习算法都有过拟合风险，当然有些比如线性回归，过拟合的倾向较轻。如果我们在同一个数据集上既做训练又做测试，
最后取得了很好表现，但究竟是因为过拟合了噪音，还是算法真的很好，就无从得知了。

好在有交叉验证(cross validation)这样比较简单的方法来避免过拟合。在交叉验证中，
我们将数据划分为若干份（folds），以3 folds为例：取其中两个folds的数据训练模型，在第三个fold上做预测。

将上述过程重复3次，就获得了对整个数据集的预测，并且避免了使用相同数据做训练和测试。
"""


"""
02. 预测

统计学习模型来自Python扩展库Scikit-learn，其中内置了交叉验证、模型训练和数据拟合。
"""
# 导入线性回归类
from sklearn.linear_model import LinearRegression
# 从交叉验证模块中导入KFold
from sklearn.cross_validation import KFold

# 定义纳入训练过程的数据列
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

# 初始化算法类
alg = LinearRegression()
# 准备数据集的交叉验证，此函数返回训练/测试集的序列
# 定义random_state为1，确保每次执行所得结果一致
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)
print(titanic.shape)
print(titanic.shape[0])
# titanic.shape方法可求出多少行多少列，>>>[891, 12]
# 而titanic.shape[0]求出多少行，>>>[891]
print("\n----------------------以上输出仅为查看学习shape方法-------------------\n")
predictions = []
for train, test in kf:
    # 提取出用作训练的数据行（不含拟合目标）
    train_predictors = (titanic[predictors].iloc[train,:])
    # print("train_predictors:\n %s"%train_predictors)
    # 提取用于训练的拟合目标
    train_target = titanic["Survived"].iloc[train]
    # print("train_target:\n %s"%train_target)
    # 基于训练数据和拟合目标训练模型
    alg.fit(train_predictors, train_target)
    # 接下来在测试集上执行预测
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)
    # print("train:\n %s" % train)
    # print("test:\n %s" % test)
# print("\n------------------------以上查看for循环中的train---------------------------\n")
print(predictions)
print(alg.intercept_)
for i in zip(predictors, alg.coef_):
    print(i)
"""
通过alg.intercept_和zip(predictors, alg.coef_可以得到变量的系数，所以得到存活率函数：

Pclass= input("enter  Pclass: ")
Sex= input("enter  Sex: ")
Age= input("enter  Age: ")
SibSp= input("enter  SibSp: ")
Parch= input("enter  Parch: ")
Fare= input("enter  Fare: ")
Embarked= input("enter  Embarked: ")

def test(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked):
    x=(-0.1673305943595034)*float(Pclass)
    y=(-0.52222839164542101)*float(Sex)
    z=(-0.0053560327484279725)*float(Age)
    a=(-0.041643391854120811)*float(SibSp)
    b=(0.0054941757863996594)*float(Parch)
    c=(-0.00033548031800063698)*float(Fare)
    d=0.045317312526614133*float(Embarked)
    return a+b+c+d+x+y+z+1.27831789091
print(test(Pclass, Sex, Age, SibSp, Parch, Fare, Embarked))

"""
print("\n------------------------以上查看预测---------------------------\n")


"""
03. 误差评估

有了预测结果之后，可以评估误差了。首先需要定义一套误差度量衡，在Kaggle竞赛的标准中，
误差是由正确预测的百分比决定的，这里也采用同样标准。

接下来就是找到predictions与titanic["Survived"]相应行一致的数量，再除以样本总量。
首先需要调用Numpy的concatenate()函数把交叉验证所划分的3个子集合并。

找出predictions中与titanic["Survived"]相同的元素所占的比例，将结果赋值予accuracy。
"""
# 计算predictions的正确率
import numpy as np

# axis=0 因为现在数组只有一维
predictions = np.concatenate(predictions, axis=0)
# 将浮点数结果映射为二进制结果（0/1表示幸存与否）
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0

# 计算predictions并赋值予accuracy
accuracy = len(predictions[predictions == titanic["Survived"]]) / len(predictions)
print(accuracy)
print("\n------------------------以上查看预测---------------------------\n")

"""
04.逻辑回归

线性回归预测的表现并不尽如人意，只有约78.3%，下面尝试逻辑回归。

逻辑回归可以看作是把线性回归的结果映射为一个0到1的值，映射过程即logisitc函数。因为本例是一个二值问题，所以尤为契合。
Scikit-learn配置了相关的类以供调用。
"""
# 逻辑回归与交叉验证
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation

# 初始化逻辑回归模型对象
alg = LogisticRegression(random_state=1)

# 计算交叉验证的得分
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)

# 求scores的平均值并输出
print(scores.mean())
print("\n------------------------以上查看scores的平均值---------------------------\n")
"""
05.测试集

逻辑回归的结果差强人意，之后还可以尝试更多的方法再进一步优化。为向Kaggle竞赛提交结果，需要将训练集上的处理应用到测试集上。
首先是数据清洗：
"""
titanic_test = pd.read_csv("http://jizhi-10061919.file.myqcloud.com/kaggle_sklearn/titanic_test.csv")

# 用中位数替换"Age"的缺失数据
titanic_test["Age"] = titanic_test["Age"].fillna(titanic["Age"].median())

# 用中位数替换"Fare"的缺失数据
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())

# 将"Sex"和"Embarked"数值化
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 1
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 0

titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
print(titanic_test)
print("\n------------------------以上查看测试集titanic_test---------------------------\n")
"""
06. 生成提交文件

提交文件也是.csv格式，由乘客ID和相应预测结果构成。生成文件的命令是submission.to_csv("kaggle.csv", index=False)
"""
# 初始化逻辑回归类
alg = LogisticRegression(random_state=1)

# 训练算法
alg.fit(titanic[predictors], titanic["Survived"])

# 对测试集做预测
predictions = alg.predict(titanic_test[predictors])
print(predictions)
print("\n------------------------以上查看预测---------------------------\n")
# 创建新的dataframe对象submission，仅含"PassengerID"和"Survived"两列。
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })

# 将submission输出为.csv格式的提交文件
submission.to_csv("kaggle.csv", index=False)



print("\n------------------------以下泰坦尼克号事故分析 03 进阶模型---------------------------\n\n\n")








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
#titanic = pd.read_csv("/Users/yinchuchu/Desktop/Data/titanic/train.csv")
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
