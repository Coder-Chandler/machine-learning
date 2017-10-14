import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
path = "/Users/chandler/Documents/Projects/machine-learning/data_analysis/StatisticsJob/Statistics.py"

df = pd.read_csv(path)
log = pd.DataFrame(df)

"""
直方图
"""
# 按时间字段对网站流量进行求和汇总
log_times = log.groupby('time')['traffic'].agg(sum)

# 创建一个一维数组赋值给a
a = np.array([1, 2, 3, 4, 5, 6, 7])

# 创建柱状图，数据源为按时间汇总的网站流量，设置颜色，透明度和外边框颜色
plt.bar([1, 2, 3, 4, 5, 6, 7], log_times, color='#000000', alpha=0.8, align='center', edgecolor='white')

# 设置x轴标签
plt.xlabel('Time')

# 设置y周标签
plt.ylabel('Traffic')

# 设置图表标题
plt.title('Traffic from 17:38 to 23:59')

# 设置图例的文字和在图表中的位置
plt.legend(['Traffic'], loc='upper left')

# 设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.4)

# 设置数据分类名称
plt.xticks(a, ('17:38~18:00', '18:00~19:00', '19:00~20:00', '20:00~21:00', '21:00~22:00', '22:00~23:00', '23:00~23:59'))

# 展示图表
plt.show()

"""
饼图
"""

# 设置饼图中每个数据分类的颜色
colors = ["#99CC01", "#FFFF01", "#0000FE", "#FE0000", "#A6A6A6", "#D9E021", "#C7E011"]

# 设置饼图中每个数据分类的名称
name = ['17:38~18:00', '18:00~19:00', '19:00~20:00', '20:00~21:00', '21:00~22:00', '22:00~23:00', '23:00~23:59']

# 创建饼图，设置分类标签，颜色和图表起始位置等
plt.pie(log_times, labels=name, colors=colors, explode=(0, 0, 0, 0, 0, 0, 0.15), startangle=60, autopct='%1.1f%%')

# 添加图表标题
plt.title('Traffic from 17:38 to 23:59')

# 添加图例，并设置显示位置
plt.legend(['17:38~18:00', '18:00~19:00', '19:00~20:00', '20:00~21:00', '21:00~22:00', '22:00~23:00', '23:00~23:59'],  loc='upper left')

# 显示图表
plt.show()


"""
箱线图
"""

# 创建箱线图，数据源为贷款来源，设置横向显示
plt.boxplot(log['traffic'], 1, 'rs', vert=False)
# 添加x轴标题
plt.xlabel('Traffic')
# 添加图表标题
plt.title('Traffic from 17:38 to 23:59')
# 设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)
# 显示图表
plt.show()


"""
散点图
"""

# 创建散点图，x轴，y轴，设置颜色，标记点样式和透明度等
name = ['17', '18', '19', '20', '21', '22', '23']
plt.scatter(log_times, name, 60, color='white', marker='o', edgecolors='#0D8ECF', linewidth=3, alpha=0.8)

# 添加x轴标题
plt.xlabel('贷款金额')

# 添加y轴标题
plt.ylabel('利息收入')

# 添加图表标题
plt.title('贷款金额与利息收入')

# 设置背景网格线的颜色，样式，尺寸和透明度
plt.grid(color='#95a5a6', linestyle='--', linewidth=1, axis='both', alpha=0.4)

# 显示图表
plt.show()

"""
散点图
"""
data = df[['time', 'traffic', 'area']]
plt.scatter(data['traffic'], data['time'], alpha=0.3)
plt.title('Traffic from 17:38~23:59')
plt.show()
# 矩阵散点图
pd.plotting.scatter_matrix(data, diagonal='kde', color='k', alpha=1)
