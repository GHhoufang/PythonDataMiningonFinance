# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 10:50:43 2021

@author: houfang
"""
# 人工神经网络分类算法
import numpy as np
import pandas as pd
#构建数据集
def createDataSet():
    dataSet = [[0,2,0,0,0],[0,2,0,1,0],[1,2,0,0,1],[2,1,0,0,1], 
               [2,0,1,0,1],[2,0,1,1,0],[1,0,1,1,1],[0,1,0,0,0], 
               [0,0,1,0,1],[2,1,1,0,1],[0,1,1,1,1],[1,1,0,1,1], 
               [1,2,1,0,1],[2,1,0,1,0]]
    
    labels = ['Age', 'Income', 'Job', 'Credit','Class']
    return dataSet,labels

ds1,lab = createDataSet()
X=np.array(ds1)[:,0:4]
Y=np.array(ds1)[:,4].reshape(14,1)
print(X)
print(Y)

#响应函数f
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#中间层/隐藏层系数矩阵syn0
#输出层系数syn1
syn0 = np.random.random((4,4)) 
syn1 =np.random.random((4,1))

print ("算法开始时，随机系数矩阵:") 
print ("syn0:" + str(syn0))
print ("syn1:" + str(syn1) ) 

#迭代1000次训练网络
for j in range(1000):
#计算误差
    layer0 = X
    layer1 = nonlin(np.dot(layer0,syn0))
    layer2 = nonlin(np.dot(layer1,syn1))
    
    layer2_error = Y - layer2
#每循环200次输出系数矩阵和误差，
#可以通过这个输出看到系数的不断调整和误差的不断收敛
    if ((j)% 200) == 0:
        print ("After:" +str(j))
        print ("Error:" + str(np.mean(np.abs(layer2_error))))
        print ("syn0:" + str(syn0))
        print ("syn1:" + str(syn1))
#梯度下降法调整系数矩阵
    layer2_delta = layer2_error*nonlin(layer2,deriv=True)
    layer1_error = layer2_delta.dot(syn1.T)
    layer1_delta = layer1_error * nonlin(layer1,deriv=True)
 
    t= layer1.T.dot(layer2_delta)
    syn1 += t
    syn0 += layer0.T.dot(layer1_delta)

print ("最终模型输出 :" + str(layer2))
#输入测试样本[ Age':<30, Income :高, Job:不稳定, Credit:好]
#该样本对应的数值是[0,2,0,1]
layer0 = [0,2,0,1]
layer1 = nonlin(np.dot(layer0,syn0))
layer2 = nonlin(np.dot(layer1,syn1))
print ("Test input :" + str(layer0)+ "Test output :" + str(layer2))
