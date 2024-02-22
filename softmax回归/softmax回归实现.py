import os

import torch
from IPython import display
from d2l import torch as d2l
from d2l.torch import Accumulator, Animator
from matplotlib import pyplot as plt

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
# Fashion MNIST是一个常用的图像分类数据集，其中包含了10个类别的服装图片，每个样本都是28x28像素的灰度图像。
# load_data_fashion_mnist函数会返回两个迭代器，一个用于训练集(train_iter)，一个用于测试集(test_iter)
# 这些数据迭代器返回的每个样本都是一个包含图像和标签的元组 (X, y)，其中：
# X 是一个形状为 (batch_size, 1, 28, 28) 的张量，表示一个大小为 28x28 像素的灰度图像，batch_size 表示每个批次的样本数。
# y 是一个形状为 (batch_size,) 的张量，表示每个图像对应的标签，即图像所代表的服装类别，范围从 0 到 9。
num_inputs = 784  # 输入特征的维度为784,因为Fashion MNIST中的每张图像都是28x28像素的
num_outputs = 10  # 10是输出类别的数量

W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
# 这个函数用于生成服从指定均值和标准差的正态分布随机数。定义了真实权重
# 均值为0，标准差0.01，随机数的形状为(行数为num_inputs,列数为num_outputs)，需要梯度记录
b = torch.zeros(num_outputs, requires_grad=True)
# 偏置参数b被初始化为0，其形状为(10,)需要记录梯度


# 演示一下按维求和
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(X.sum(0, keepdim=True), X.sum(1, keepdim=True))


# 书写softmax函数
def softmax(X):
    X_exp = torch.exp(X)  # 首先，对输入的张量 X 中的每个元素进行指数化运算，得到指数化后的张量 X_exp
    partition = X_exp.sum(1, keepdim=True)  # 对 X_exp 中的每行进行求和，得到一个列向量 partition，其中每个元素是对应行的指数化值的总和
    return X_exp / partition
    # 最后，将指数化后的张量 X_exp 中的每个元素除以相应行的总和（即 partition 中的元素），得到归一化后的概率值，即softmax函数的输出。返回归一化后的张量。


# 验证这个函数
X = torch.normal(0, 1, size=(2, 5))
print(X)
print(softmax(X))
print(softmax(X).sum(1))  # 对softmax函数的输出结果沿着第1维度进行求和，即对每一行的元素求和，然后打印结果。


# 实现softmax回归模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)


# 参数 X 代表着神经网络的输入数据。在神经网络中，输入数据通常是一批样本，每个样本都是一个特征向量。
# X.reshape((-1, W.shape[0]))：首先，将输入张量 X 进行形状变换，将其变换为一个二维张量，其中行数为 -1，表示自动计算，
# 列数为 W.shape[0]，即权重矩阵 W 的行数
# torch.matmul(X.reshape((-1,W.shape[0])), W)：接着，进行矩阵乘法运算，将形状变换后的输入张量与权重矩阵 W 进行相乘。
# softmax(torch.matmul(X.reshape((-1,W.shape[0])), W) + b)：将矩阵乘法的结果与偏置向量 b 相加，
# 然后将相加的结果传递给 softmax 函数进行处理。这一步实现了神经网络的线性变换和激活函数处理（softmax）。
# 最终，softmax 函数对线性变换的结果进行处理，得到样本属于各个类别的概率分布，
# 输出的是一个概率分布向量，其中每个元素表示样本属于对应类别的概率。

# 演示提取预测正确的概率
y = torch.tensor([0, 2])  # 有两个样本的真实类别分别是0和2
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 这两个样本的0，1，2类概率的预测结果分别是这样
print(y_hat[[0, 1], y])


# 提取出预测结果中实际真实符合的那一项的概率，比如第一个样本属于第一类，那么就把第一个样本的预测概率中第一类的概率也就是0.1提出来
# 通过这个我们可以知道y_hat[从0到len(y_hat),真实的y样本]由于预测值之和为1，那么就能直接返回预测正确的概率
# 这个返回的概率属于[0,1],所以可以使用f=-ln（x）函数进行效果扩大，当预测正确率x趋于0，那么f就趋于正无穷
# 当预测正确率x趋于1，那么f就趋于0
# 交叉熵损失函数，返回值的大小可以衡量预测的效果，返回值越大，那么预测得越不好
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


print(cross_entropy(y_hat, y))


# 计算预测正确的数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 如果 y_hat 是一个二维数组且有多个类别（即列数大于 1），则将其转换为一维数组，取每一行中最大值所在的索引作为预测标签。
    cmp = y_hat.type(y.dtype) == y  # 将预测标签与真实标签 y 进行比较，生成一个布尔类型的数组 cmp，表示预测是否正确。
    return float(cmp.type(y.dtype).sum())  # 计算正确预测的数量，并除以总样本数，得到模型的准确率。


print(accuracy(y_hat, y) / len(y))


# 用于评估神经网络模型在给定数据迭代器上的准确率
def evaluate_accuracy(net, data_iter):  # net 表示神经网络模型，data_iter 表示数据迭代器，用于遍历数据集
    if isinstance(net, torch.nn.Module):
        net.eval()  # 如果输入的 net 是 torch.nn.Module 的实例，则将其设为评估模式
    metric = Accumulator(2)  # 创建一个累加器（metric）来计算准确率，初始化为两个累加器，用于记录预测正确的样本数量和总样本数量。
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
        # 每一批小样本，累加器的第一个元素累加一次判断正确类别的样本数量，第二个元素累加一次这一批样本的总数量
    return metric[0] / metric[1]  # 返回所有样本的总预测成功率


# 由于操作系统是windows，所以不能用那个预设的4进程，在torch.py文件中找到d2l.torch.get_dataloader_workers函数，将返回值修改为0
print(evaluate_accuracy(net, test_iter))


# softmax回归训练
def train_epoch_ch3(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()  # 首先检查 net 是否是 torch.nn.Module 类型的实例，如果是的话，将模型置为训练模式。
    metric = Accumulator(3)
    # 创建了一个累加器 metric，用来累积三种指标：训练损失、训练准确率的分子（正确预测的样本数量）、训练准确率的分母（总样本数量）。
    for X, y in train_iter:  # 对训练集迭代器进行循环，每次获取一个batch的训练数据 X 和对应的标签 y
        y_hat = net(X)  # 将输入数据 X 输入到神经网络 net 中，得到模型的预测输出 y_hat
        l = loss(y_hat, y)  # 使用给定的损失函数 loss 计算模型的预测输出 y_hat 与真实标签 y 之间的损失
        if isinstance(updater, torch.optim.Optimizer):
            # 检查是否使用了优化器，如果是的话，则执行优化器相关的操作，包括反向传播、参数更新以及统计训练指标
            updater.zero_grad()  # 清零优化器中参数的梯度，避免梯度累积
            l.backward()  # 执行反向传播计算梯度。
            updater.step()  # 根据计算得到的梯度执行参数更新。
            metric.add(  # 将训练损失、训练准确率的分子和分母累加到累加器中。
                float(l) * len(y), accuracy(y_hat, y),
                y.size().numel())
        else:
            l.sum().backward()
            # 计算损失函数的和（因为每个样本都有一个损失值，这里将它们求和作为一个batch的损失），并执行反向传播计算梯度。
            updater(X.shape[0])  # 执行自定义的参数更新操作，这里参数 updater 可能是一个自定义的更新函数。
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())  # 同样将训练损失、训练准确率的分子和分母累加到累加器中。
    return metric[0] / metric[2], metric[1] / metric[2]
    # metric[0] / metric[2] 计算的是平均损失，metric[1] / metric[2] 计算的是平均准确率


lr = 0.1


def updatar(batch_size):
    return d2l.sgd([W, b], lr, batch_size)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    # net: 要训练的神经网络模型。
    # train_iter: 训练数据集的迭代器，用于遍历训练数据。
    # test_iter: 测试数据集的迭代器，用于评估模型性能。
    # loss: 损失函数，用于计算模型预测和真实标签之间的损失。
    # num_epochs: 训练的总轮数。
    # updater: 参数更新器，用于更新模型的参数。
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    # 使用 Animator 对象创建了一个动画，用于实时可视化训练过程中的损失和准确率。
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        # 调用 train_epoch_ch3 函数来训练模型一个轮次，并获取训练指标（包括训练损失和准确率）
        test_acc = evaluate_accuracy(net, test_iter)
        # 使用 evaluate_accuracy 函数计算模型在测试数据集上的准确率。
        animator.add(epoch + 1, train_metrics + (test_acc,))
        # 将当前轮次的训练指标和测试准确率添加到动画中
    train_loss, train_acc = train_metrics
    # 在训练结束后，对训练损失和准确率进行断言，确保它们满足一定的条件。
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def predict_ch3(net, test_iter, n=6):
    """Predict labels (defined in Chapter 3)."""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true + '\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(d2l.reshape(X[0:n], (n, 28, 28)), 1, n,
                    titles=titles[0:n], scale=3.0)

# 这个是一个验证的函数，取前6张图进行验证，我分了两次调用，训练前和训练后，并且在原来写的函数里面加了一个”, scale=3.0“，不然显示不完整
d2l.set_figsize()
predict_ch3(net, test_iter)
plt.show()

num_epochs = 5
print(W, b)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updatar)
plt.show()
print(W, b)

predict_ch3(net, test_iter)
plt.show()
