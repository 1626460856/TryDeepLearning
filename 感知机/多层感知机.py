import os

import matplotlib.pyplot as plt
import torch
from torch import nn
from d2l import torch as d2l
# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
batch_size = 256  # 批次数据大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256  # 输入尺寸28*28 输入数据10个类别的概率 隐藏层输出尺寸256
W1 = nn.Parameter(torch.zeros(num_inputs, num_hiddens, requires_grad=True))  # 权重初始为0，权重矩阵R[输入尺寸,类别数]
# 教程是平均数randn，但是这里改成了0减少方差，不然后面损失大了要报错很难受
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))  # 偏差初始为0，偏置尺寸R[类别数]
W2 = nn.Parameter(torch.randn(num_hiddens, num_outputs, requires_grad=True))  # 隐藏层输出作为输入尺寸，类别10
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))  # 输出层偏置[类别数]
parames = [W1, b1, W2, b2]  # parames 是一个列表，其中包含了神经网络中需要优化的参数


# 激活函数
def relu(X):
    a = torch.zeros_like(X)  # 创建了一个与输入张量 X 具有相同形状的全零张量，
    return torch.max(X, a)  # 它会逐元素比较两个张量 X 和 a，并返回对应位置上的较大值。


# 模型实现
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return H @ W2 + b2


loss = nn.CrossEntropyLoss()
#  交叉熵损失（Cross Entropy Loss）函数。
# 训练
num_epochs,lr = 10,0.01
updater = torch.optim.SGD(parames, lr=lr)
print(parames)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,updater)
plt.show()
print(parames)