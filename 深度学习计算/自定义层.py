import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import layer_norm


class CenteredLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        return X-X.mean()
layer=CenteredLayer()
print("各个值分别减去均值，使得值中心化，让均值变为0：\n",layer(torch.FloatTensor([1,2,3,4,5])))
net=nn.Sequential(nn.Linear(8,128),CenteredLayer())
Y=net(torch.rand(4,8))
print("输出Y的均值：",Y.mean())
class MyLinear(nn.Module):
    #定义了一个名为 MyLinear 的类，该类继承自 torch.nn.Module，使其成为一个可用于神经网络构建的模块。
    def __init__(self,in_units,units):
        #这是类的构造函数，用于初始化 MyLinear 对象。它接受两个参数：in_units（输入特征的数量）和 units（输出特征的数量）。
        super().__init__()
        #调用父类 nn.Module 的构造函数，是 PyTorch 自定义模块的标准做法。
        self.weight=nn.Parameter(torch.randn(in_units,units))
        #创建一个形状为 (in_units, units) 的权重矩阵，并将其注册为模块的参数。torch.randn 生成的是标准正态分布（均值为0，方差为1）的随机数，用于权重的初始化。
        self.bias=nn.Parameter(torch.randn(units,))
        #创建一个长度为 units 的偏置向量，并将其注册为模块的参数。同样，偏置参数也使用了标准正
    def forward(self,X):
        #定义了模块的前向计算，它接受一个输入张量 X，完成模块的前向计算，并返回结果。
        linear=torch.matmul(X,self.weight.data)+self.bias.data
        #完成了线性计算，其中 torch.matmul(X, self.weight.data) 是矩阵乘法，self.weight.data 和 self.bias.data 分别是权重和偏置的参数张量。
        return F.relu(linear)
        #应用 ReLU 激活函数到线性变换的结果上，并返回激活后的结果。ReLU 函数是一种常用的激活函数，对于正数输入保持不变，对于负数输入则输出0。
dense=MyLinear(5,3)#创建了一个 MyLinear 实例 dense，输入特征数量为5，输出特征数量为3。
print("权重参数：",dense.weight,"\n偏置参数",dense.bias)
print("dense(X)：",dense(torch.rand(2,5)))
net=nn.Sequential(MyLinear(64,8),MyLinear(8,1))
print("net(X)：",net(torch.rand(2,64)))