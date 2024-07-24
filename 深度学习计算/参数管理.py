import torch
from torch import nn

net =nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,1))
X=torch.rand(size=(2,4))
print("随机生成了一个输入张量 X，形状为 (2, 4)：\n",X)
print("net模型结构：\n",net)
print("通过 net(X) 在前向传播中计算输出：\n",net(X))
print("################################################################################################################")
print("输出第三个模块（索引为 2 的模块，即 nn.Linear(8,1)）的状态字典。\n"
      "这个状态字典包含了该线性层的权重和偏置参数。具体来说，它会有两个键：'weight' 和 'bias'，对应于该层的权重和偏置。\n"
      "每个键映射到相应的参数张量。\n",net[2].state_dict())
print("偏置类型:\n",type(net[2].bias))
print("偏置参数的值和相关信息:\n",net[2].bias)
print("偏置参数的数据部分，忽略了关于梯度的信息:\n",net[2].bias.data)
print("################################################################################################################")
print("一次性访问所有参数\n",*[(name,param.shape) for name,param in net.named_parameters()])
print("访问第一个全连接层的参数\n",net[0].weight.data)
print("通过字符串名称访问“2.bias”数据：",net.state_dict()['2.bias'].data)
print("################################################################################################################")
def block1():
    return nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,4),nn.ReLU())
#block1 函数返回一个由两个全连接层和两个 ReLU 激活层组成的序列模型。
# 第一个全连接层将输入的特征从 4 维映射到 8 维，接着是一个 ReLU 激活层。
# 第二个全连接层将特征从 8 维映射回 4 维，后面再跟一个 ReLU 激活层。
def block2():
    net=nn.Sequential()
    for i in range(4):
        net.add_module(f'block {i}',block1())
    return net
#block2 函数构造了一个更大的序列模型，通过循环四次调用 block1 函数，
# 将每次调用的返回值（即一个 block1 返回的序列模型）作为一个模块添加到 net 中。
# 这样，block2 函数最终返回的是一个包含四个 block1 模块的序列模型。
rgnet=nn.Sequential(block2(),nn.Linear(4,1))
#rgnet 是一个更大的神经网络，它首先使用 block2 函数构造的模型处理输入，
# 然后通过一个全连接层将特征从 4 维映射到 1 维。
# 这个网络可以看作是一个更复杂的多层神经网络，其中包含了多个重复的模块（由 block1 定义），这些模块被组织在 block2 中，最后通过一个输出层产生最终的预测。
print("rgnet模型结构：\n",rgnet)
print("rgnet模型输出net(X)：\n",rgnet(X))
print("################################################################################################################")
print("演示初始化默认参数")
def init_normal(m):
    if type(m)==nn.Linear:#判断是否为线性层
        nn.init.normal_(m.weight,mean=0,std=0.01)
        nn.init.zeros_(m.bias)
net.apply(init_normal)
print("使用正态分布（均值为 0，标准差为 0.01）初始化模块的权重参数")
print("使用常数值 0 初始化模块的偏置参数")
print("初始化后的第一个全连接层的权重参数：\n",net[0].weight.data)
print("初始化后的第一个全连接层的偏置参数：\n",net[0].bias.data)
def init_constant(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,1)
        nn.init.zeros_(m.bias)
net.apply(init_constant)
print("使用常数值 1 初始化模块的权重参数\n"
              "但是一般不能全部初始化为常数，因为这样计算的梯度就一样，会影响计算")
print("使用常数值 0 初始化模块的偏置参数")
print("初始化后的第一个全连接层的权重参数：\n",net[0].weight.data)
print("初始化后的第一个全连接层的偏置参数：\n",net[0].bias.data)
print("根据您提供的代码和打印结果，看起来每个初始化函数确实被调用了两次。\n"
      "这种情况通常发生在网络中每个nn.Linear模块上，\n"
      "因为net.apply(init_function)会对网络中的每个模块递归地应用init_function函数。\n"
      "如果您的网络net中有两个nn.Linear模块，那么每个初始化函数就会被调用两次，一次针对每个nn.Linear模块。")
print("################################################################################################################")
print("对某些块应用不同的初始化方法")
def xavier(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m)==nn.Linear:
        nn.init.constant_(m.weight,42)
net[0].apply(xavier)
net[2].apply(init_42)
print("net模型结构:\n",net)
print("使用 Xavier 初始化方法初始化模块的权重参数")
print("使用常数值 42 初始化模块的权重参数")
print("初始化后的第一个全连接层的权重参数：\n",net[0].weight.data)
print("初始化后的第三个全连接层的权重参数：\n",net[2].weight.data)
print("################################################################################################################")
print("自定义初始化")
def my_init(m):
    if type(m)==nn.Linear:
        print("Init",*[(name,param.shape) for name,param in m.named_parameters()][0])
        nn.init.uniform_(m.weight,-10,10)
        m.weight.data *=m.weight.data.abs()>=5
net.apply(my_init)
print(net[0].weight[:2])
print("################################################################################################################")
print("共享模型参数")
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), shared, nn.ReLU(), shared,
                    nn.ReLU(), nn.Linear(8, 1))
print("net模型结构:\n",net)
print("shared 是一个 nn.Linear(8, 8) 层，它被用作 net 模型中的第三个（索引为 2）和第五个（索引为 4）模块。\n"
      "当你通过 net[2].weight.data[0,0] = 100 修改这个共享层的权重时，\n"
      "由于 net[2] 和 net[4] 实际上引用的是同一个 nn.Linear 实例，所以对 net[2] 的权重所做的任何修改都会反映在 net[4] 上，反之亦然。")
net[2].weight.data[0,0] = 100
print("net[2].weight[0,0]:\n",net[2].weight[0,0] )
print("net[4].weight[0,0]:\n",net[4].weight[0,0] )
net[4].weight.data[0, 0] = 200
print("net[2].weight[0,0]:\n",net[2].weight[0,0] )
print("net[4].weight[0,0]:\n",net[4].weight[0,0] )
print("################################################################################################################")
























