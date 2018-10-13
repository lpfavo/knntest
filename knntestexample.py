# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 11:29:12 2017

@author: lpfavo231
"""

from sklearn import neighbors
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#随机 6 组 50个的正态分布
x1 = np.random.normal(50, 6, 200)#均值，标准差，输出值的个数
y1 = np.random.normal(5, 0.5, 200)

x2 = np.random.normal(30,6,200)
y2 = np.random.normal(4,0.5,200)

x3 = np.random.normal(45,6,200)
y3 = np.random.normal(2.5, 0.5, 200)

'''
x1、x2、x3 作为 x 坐标，y1、y2、y3 作为 y 坐标，两两配对。
(x1,y1) 标为 1 类，(x2, y2) 标为 2 类，(x3, y3)是 3 类。
将它们画出得到下图，1 类是蓝色，2 类红色，3 类绿色。
'''
#具体参考：http://blog.csdn.net/u013634684/article/details/49646311
plt.scatter(x1,y1,c='b',marker='s',s=50,alpha=0.8)#
plt.scatter(x2,y2,c='r', marker='^', s=50, alpha=0.8)
plt.scatter(x3,y3, c='g', s=50, alpha=0.8)

#开始找距离给定点最近的k个点

#把所有的 x 坐标和 y 坐标放在一起
x_val=np.concatenate((x1,x2,x3))#数组拼接
y_val=np.concatenate((y1,y2,y3))

#计算距离的归一化问题，计算x 值的最大差还有 y 值的最大差
x_diff=max(x_val)-min(x_val)
y_diff=max(y_val)-min(y_val)

#将坐标除以这个差以归一化，再将 x 和 y 值两两配对
x_normalized = [x/(x_diff) for x in x_val]
y_normalized = [y/(y_diff) for y in y_val]
xy_normalized = list(zip(x_normalized, y_normalized))#list
#print(xy_normalized)

'''
训练使用的特征数据已经准备好了，还需要生成相应的分类标签。生成一个长度150的list，
前50个是1，中间50个是2，最后50个是3，对应三种标签。
''' 

labels= [1]*200+[2]*200+[3]*200

'''
生成 sklearn 的最近 k 邻分类功能了。参数中，n_neighbors 设为30，其他的都使用默认值即可。
'''
clf = neighbors.KNeighborsClassifier(30)

#下面就要进行拟合了。归一化的数据是 xy_normalized，分类标签是 labels，
clf.fit(xy_normalized, labels)


#实现功能

#k最近邻
#首先，我们想知道 (50,5) 和 (30,3) 两个点附近最近的 5 个样本分别都是什么。啊，坐标别忘了除以 x_diff 和 y_diff 来归一化
nearests = clf.kneighbors([(50/x_diff,5/y_diff),(30/x_diff,3/y_diff)],5, False)
print('nearests:')
print(nearests)

#预测
prediction = clf.predict([(50/x_diff,5/y_diff),(30/x_diff,3/y_diff)])
print('prediction:')
print(prediction)

#概率预测
prediction_proba = clf.predict_proba([(50/x_diff,5/y_diff),(30/x_diff,3/y_diff)])
print('prediction_proba:')
print(prediction_proba)

#准确率打分
x1_test = np.random.normal(50, 6, 100)
y1_test = np.random.normal(5, 0.5, 100)

x2_test = np.random.normal(30, 6,100)
y2_test = np.random.normal(4, 0.5,100)

x3_test = np.random.normal(45, 6, 100)
y3_test = np.random.normal(2.5, 0.5, 100)

xy_test_normalized = list(zip(np.concatenate((x1_test,x2_test,x3_test))/x_diff,np.concatenate((y1_test,y2_test,y3_test))/y_diff))

latels_test = [1]*100+[2]*100+[3]*100

score = clf.score(xy_test_normalized, latels_test)
print(score)

