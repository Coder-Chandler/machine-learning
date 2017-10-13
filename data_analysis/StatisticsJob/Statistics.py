#导入需要的模块
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import datetime
path = "/Users/chandler/Desktop/log_test/day=20130530/part-00000-34e4fe9d-faaf-4a14-90dc-636438f0a3d0.c000.csv"
#读取CSV数据为numpy record array记录
r = mlab.csv2rec(path)
r.sort()
#形成Y轴坐标数组
N = len(r)
ind = np.arange(N)  # the evenly spaced plot indices
#ind1这里是为了把图撑大一点
ind1 = np.arange(N+3)


def format_date(x, pos=None):
    if not x % 1 and x < N:
        thisind = np.clip(x, 0, N-1)
        return r.datetime[thisind].strftime('%Y-%m-%d')
    else:
        return ''


#绘图
fig = plt.figure()
ax = fig.add_subplot(111)
#下行为了将图扩大一点，用白色线隐藏显示
ax.plot(ind1,ind1,'-',color='white')
#正常要显示的bug总数折线
ax.plot(ind, r['traffic'], 'o-',label='traffic')
#图标的标题
ax.set_title(u"traffic")
#线型示意说明
ax.legend(loc='upper left')

#在折线图上标记数据，-+0.1是为了错开一点显示数据
datadotxy=tuple(zip(ind-0.1,r['traffic']+0.1))
for dotxy in datadotxy:
    ax.annotate(str(int(dotxy[1]-0.1)),xy=dotxy)

#将X轴格式化为日期形式
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
fig.autofmt_xdate()

#显示图片
plt.show()


