import torch
from IPython import display
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# Fashion MNIST是一个常用的图像分类数据集，其中包含了10个类别的服装图片，每个样本都是28x28像素的灰度图像。
# load_data_fashion_mnist函数会返回两个迭代器，一个用于训练集(train_iter)，一个用于测试集(test_iter)
num_inputs = 784  # 输入特征的维度为784,因为Fashion MNIST中的每张图像都是28x28像素的
num_outputs = 10  # 10是输出类别的数量

w = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 这个函数用于生成服从指定均值和标准差的正态分布随机数。定义了真实权重
# 均值为0，标准差0.01，随机数的形状为(行数为num_inputs,列数为num_outputs)，需要梯度记录
b = torch.zeros(num_outputs, requires_grad=True)
# 偏置参数b被初始化为0，其形状为(10,)需要记录梯度

X=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(X.sum(0, keepdim=True),X.sum(1, keepdim=True))
