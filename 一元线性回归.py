#一元线性回归的算法

import matplotlib.pyplot as plt
import numpy as np


#导入数据点，注意文件路径
data = np.genfromtxt("/Users/xieyu/Development/data.csv",delimiter=",")
# ：前后缺省，代表从头到尾
x_data = data[:,0]
y_data = data[:,1]

#设置学习率lr，初始斜率和截距k,b，最大学习次数50
lr = 0.0001
b = 0
k = 0
epochs = 50

#求出初始的误差平方J(k,b) = total_error
def compute_error(b,k,x_data,y_data):
    total_error = 0
    for i in range(len(x_data)):
        total_error += (y_data[i] - (k*x_data[i] +b))**2
    return total_error / float(len(x_data)) /2.0

#梯度下降法原理
def gradient(x_data,y_data,b,k,lr,epochs):
    m = float(len(x_data))
    for i in range (epochs):
        #偏微分b_grad 和 k_grad
        b_grad = 0
        k_grad = 0
        for j in range(0,len(x_data)):
            b_grad += (k*x_data[j] + b - y_data[j]) / m
            k_grad += x_data[j] * (k*x_data[j] + b - y_data[j]) / m
        #迭代，进行梯度下降
        b = b - lr*b_grad
        k = k - lr*k_grad
#得到最终的b和k
    return b,k


print('初始的误差=',compute_error(b,k,x_data,y_data))
b,k = gradient(x_data,y_data,b,k,lr,epochs)
print('k=',k,'b=',b)
print('最后的误差=',compute_error(b,k,x_data,y_data))
plt.scatter(x_data,y_data)
plt.plot(x_data,k*x_data+b,'r')
plt.show()

