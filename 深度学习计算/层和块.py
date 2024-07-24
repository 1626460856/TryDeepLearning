import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential( nn.Linear(20,256), nn.ReLU(), nn.Linear(256, 10))
print("演示简单多层感知机-线性模型net - 输入尺寸20维度，输出尺寸10维度：\n",net)
#随机矩阵 2*20
X=torch.rand(2,20)
print("随机矩阵X-2*20：\n",X)

print("线性模型输出net(X)-2*10：\n",net(X))
print("################################################################################################################")
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,X):
        print("调用向前传播函数，为后续梯度的反向传播做准备：")
        return self.out(F.relu(self.hidden(X)))
print("定义子类MLP，继承nn.Module，实现forward函数：\n",MLP())
net=MLP()
print("MLP模型输出net(X)-2*10：\n",net(X))
print("################################################################################################################")
class MySequential(nn.Module):
    def __init__(self,*args):
        print("调用顺序快构造函数，传入任意数量的子模块列表：\n",args)
        super().__init__()
        print("通过调用 super().__init__()，它首先初始化父类 nn.Module。然后，它遍历所有传入的子模块，并将它们添加到内部字典 _modules 中")
        for block in args:
            self._modules[block]=block
    def forward(self,X):
        print("前向传播函数接收输入数据 X，然后按照 _modules 字典中存储的顺序，\n依次将 X 传递给每个子模块。每个子模块处理完 X 后的输出会成为下一个子模块的输入。最终，返回最后一个子模块的输出")
        for block in self._modules.values():
            X=block(X)
        return X
#这个自定义的 MySequential 类的目的是提供一种灵活的方式来定义一个序列模型，其中模型的各个部分可以按顺序执行。
# 但是，由于在添加模块到 _modules 字典时使��模块对象作为键的错误，这段代码实际上不能正确运行。正确的做法是为每个模块指定一个唯一的名称作为键。
net=MySequential(nn.Linear(20,256),nn.ReLU(),nn.Linear(256,10))
print("MySequential模型输出net(X)-2*10：\n",net(X))
print("################################################################################################################")
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight=torch.rand((20,20),requires_grad=False)
        #requires_grad=False 表示不记录梯度这个权重在训练过程中不会被更新
        self.linear=nn.Linear(20,20)
    def forward(self,X):

        X=self.linear(X)
        X=F.relu(torch.mm(X,self.rand_weight)+1)
        X=self.linear(X)
        while X.abs().sum()>1:
            X/=2
        return X.sum()
net=FixedHiddenMLP()
print("演示在正向传播函数中执行代码")
print("forward(self, X): 在前向传播函数中，输入 X 首先通过定义的线性层 self.linear 进行变换。\n"
              "接着，使用固定的随机权重 self.rand_weight 对结果进行矩阵乘法操作，并加上 1 后应用 ReLU 激活函数。\n"
              "之后，再次通过相同的线性层进行变换。\n"
              "最后，使用一个循环将输出 X 的绝对值之和大于 1 的部分除以 2，直到其绝对值之和不大于 1，然后返回 X 的所有元素之和。 ")
print("FixedHiddenMLP模型输出net(X)：\n",net(X))
print("################################################################################################################")
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        print("首先调用 super().__init__() 初始化父类。\n"
              "然后，通过 nn.Sequential 构建了一个顺序容器 self.net，其中包含两个线性层和两个 ReLU 激活层。\n"
              "这个顺序容器定义了一个从输入特征维度为 20 到 64，再到 32 的神经网络。\n"
              "接着，定义了一个额外的线性 self.linear，将特征维度从 32 映射到 16。")
        self.net=nn.Sequential(nn.Linear(20,64),
                               nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear=nn.Linear(32,16)
    def forward(self,X):
        print("在前向传播函数中，输入 X 首先通过 self.net 生成隐藏表示，然后传入 self.linear 生成模型输出")
        return self.linear(self.net(X))
chimera=nn.Sequential(NestMLP(),nn.Linear(16,20),FixedHiddenMLP())
print("创建了一个名为 chimera 的顺序模型，它首先包含了一个 NestMLP 实例，\n"
      "然后是一个将特征维度从 16 映射到 20 的线性层，最后是一个 FixedHiddenMLP 实例。\n"
      "这个组合模型 chimera 展示了如何将不同的自定义模型和标准层组合成一个更复杂的网络结构。\n"
      "在这个结构中，数据会首先通过 NestMLP 进行处理，然后通过一个线性层，最后通过 FixedHiddenMLP 完成最终的处理。")
print("演示嵌套模型",chimera(X))
print("################################################################################################################")




















