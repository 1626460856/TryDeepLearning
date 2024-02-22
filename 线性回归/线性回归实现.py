import os
import random
import torch
from d2l import torch as d2l

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 定义总数据集y=Xw+b+噪声
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 这一行创建了一个张量 X，其中包含了 num_examples 行（样本数）和 len(w) 列（特征数）。
    # 这些数据是从均值为0，标准差为1的正态分布中随机生成的。
    y = torch.matmul(X, w) + b  # 对每个样本的特征向量进行加权求和，然后加上偏置
    y += torch.normal(0, 0.01, y.shape)  # 这一行给生成的因变量 y 加上了一些噪声，这样可以使得数据更贴近真实情况。
    # 噪声是从均值为0，标准差为0.01的正态分布中随机生成的
    return X, y.reshape(-1, 1)


# y.reshape(-1, 1)对目标值张量 y 进行重新形状操作的一部分，它的目的是将目标值张量从原来的形状重新调整为一个列向量的形式


true_w = torch.tensor([2, -3.4])  # 是一个张量，包含了真实的权重值，其长度决定了特征向量的维度。
true_b = 4.2  # 是一个标量，表示真实的偏置值
features, labels = synthetic_data(true_w, true_b, 1000)
# features是一个包含1000行（样本数）和2列（特征数）的张量X
# labels是一个包含1000行和1列的张量，其中每一行对应一个样本的目标值

d2l.set_figsize()  # 调整当前图形的大小，以便在绘制图形时适应所选的大小
# 特征的第二列被用作 x 坐标，标签被用作 y 坐标，而散点的大小为1
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
# features[:, 1].detach().numpy() 选择了特征张量中的第二列，并将其转换为 NumPy 数组。
# .detach() 方法用于创建一个新的张量，该张量不再与计算图关联，.numpy() 方法将张量转换为 NumPy 数组
# labels.detach().numpy() 将标签张量转换为 NumPy 数组。
# 1 是指定散点的大小参数。这里，散点的大小被设置为1。
d2l.plt.show()


def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 计算了特征张量 features 中的样本数量，即数据集中的样本总数
    indices = list(range(num_examples))  # 创建了一个包含样本索引的列表 indices，它包含了从0到 num_examples-1 的所有整数
    random.shuffle(indices)  # 这一行随机打乱了索引列表 indices，这样可以确保在每次迭代中，样本的顺序都是随机的
    for i in range(0, num_examples, batch_size):  # 遍历所有的样本并生成小批量数据
        batch_indices = torch.tensor(
            indices[i:min(i + batch_size, num_examples)])
        # 创建了一个张量 batch_indices，其中包含了当前批次的样本索引。
        # min(i + batch_size, num_examples) 确保在最后一个批次中不会超出样本总数
        yield features[batch_indices], labels[batch_indices]
        # yield 关键字用于生成器函数，它将每次迭代生成的小批量数据返回给调用者，而不会中断函数的执行状态。
        # 这样，函数在每次调用时都会返回一个新的小批量数据，而不会重新开始执行。


# 数据迭代器函数，它用于生成用于训练的小批量数据,实际上是每次生成小批次的索引再对应回原数据


# 演示调用这个小批量生成函数
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 定义初始化模型参数
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# w这是一个形状为 (2, 1) 的张量，代表了线性回归模型的权重。它使用 torch.normal 函数从均值为0，标准差为0.01的正态分布中随机初始化。
# requires_grad=True 表示我们希望在后续的计算中对 w 进行梯度计算，以便使用反向传播算法进行模型训练。
b = torch.zeros(1, requires_grad=True)
#  b: 这是一个形状为 (1,) 的张量，代表了线性回归模型的偏置。它被初始化为全零张量，同样使用了 requires_grad=True 来启用梯度计算。


# 定义线性回归模型，对输入特征进行线性变换，得到对应的预测值，计算过程中梯度的计算便发挥了作用
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 定义损失函数1/2*（y_hat-y）^2
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    # 由于形状可能不匹配，所以使用了 y.reshape(y_hat.shape) 来调整真实标签的形状与预测值相匹配


# 定义优化算法-小批量随机梯度下降-实现随机梯度下降（SGD）优化算法的函数，用于更新模型参数以最小化损失函数
def sgd(params, lr, batch_size):
    # params 是一个包含模型参数的列表，每个参数都是一个张量。
    # lr 是学习率，表示每次更新时参数应该改变的大小。
    # batch_size 是批量大小，表示每次计算损失时使用的样本数量。
    with torch.no_grad():  # 在这个范围内不要跟踪梯度
        for param in params:  # 遍历模型的每个参数
            param -= lr * param.grad / batch_size
            # 这一步根据梯度下降的规则更新参数。
            # 对于每个参数，我们将其当前值减去学习率乘以梯度的平均值（即梯度除以批量大小）。
            # 这一步将沿着梯度的反方向更新参数，以尽量减少损失函数的值。
            param.grad.zero_()
            # 这一步将参数的梯度清零，以便下一轮迭代时重新计算新的梯度。


# 训练过程
lr = 0.1  # 学习率（步长）
num_epochs = 3  # 把整个数据扫三遍
net = linreg  # 模型规定为上述定义的线性回归模型也就是y，通过把y计算出来就能在这个过程中记录梯度
loss = squared_loss  # 同理也是指定损失函数
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):  # 分小批进行，遍历每个小批的数据
        l = loss(net(X, w, b), y)
        # net的模型计算通过前向传播计算模型对输入数据 X 的预测值，并计算预测值与真实标签 y 之间的损失
        # loss的损失计算返回的也是张量，而且是分批进行的
        l.sum().backward()
        # 反向传播，利用自动微分功能计算损失函数关于模型参数的梯度。
        sgd([w, b], lr, batch_size)  # 这个函数不断更新了params张量的数值，也就是w和b的数据
    # 下面是验证效果
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1},loss{float(train_l.mean()):f}')
        print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
        print(f'b的估计误差：{true_b - b}')
        print("当前训练之后的w：",w,";b:",b)

