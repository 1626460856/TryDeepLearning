import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

# 生成真实数据
# 这个直接调用的api是沐神自己写的，内容和上面详细演示是一致的
# 因此，features 是一个张量，包含了 10000 个样本的特征数据，每个样本有两个特征；
# labels 也是一个张量，包含了 10000 个样本的标签数据，每个样本对应一个标签值。
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 10000)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# 可以使用next对data_iter数据进行使用
batch_data = next(iter(data_iter))
# 获取特征和标签数据
features_batch, labels_batch = batch_data

print("分别赋值的方式访问labels_batch：", '\n', labels_batch)
print("直接索引的方式访问labels_batch：", '\n', batch_data[1])

# 使用框架的预定好的层
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))  # 需要输入“输入层维度”和“输出层维度”，使用的函数是Linear
# 这种网络模型的创建方式是使用 nn.Sequential 将多个层按顺序连接起来，形成一个网络模型。
# 在这里，只包含了一个线性层，即一个输入特征维度为 2，输出特征维度为 1 的线性层。


# 初始化模型参数,具体来说，它对模型 net 中的第一个层（即线性层）的权重和偏置项进行了初始化。
net[0].weight.data.normal_(0, 0.01)  # 这行代码对权重w(w1,w2)进行初始化，使用了正态分布（normal distribution），均值为 0，标准差为 0.01
net[0].bias.data.fill_(0)  # 这行代码对偏置项b进行初始化，将所有偏置项的值都设置为 0。.fill_() 方法用于将张量的所有元素设置为指定的值。

# 定义均方误差1/2|差值|^2
loss = nn.MSELoss()

# 实例化SGD实例
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# net.parameters()：获取了模型 net 中所有需要优化的参数。net.parameters() 返回一个包含了模型参数的迭代器。

# 训练过程
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l= loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l= loss(net(features),labels)
    print(f'epoch: {epoch+1}, loss: {l:f}')
    # 获取训练完成后的权重值
    weight = net[0].weight.data
    bias = net[0].bias.data

    # 打印权重值
    print("训练完成后的权重值：", weight)
    print("训练完成后的偏置值：", bias)
