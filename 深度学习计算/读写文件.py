import torch
from torch import nn
from torch.nn import functional as F

x=torch.arange(4)
#创建一个从0到3的一维张量（数组），即[0, 1, 2, 3]。
torch.save(x,'x-file')
#将张量x保存到名为'x-file'的文件中。这个文件会在当前工作目录下创建。
x2=torch.load('x-file')
#从文件'x-file'中加载数据，将其赋值给变量x2。这里加载的数据是之前保存的张量x。
print(x2)
y=torch.zeros(4)
#创建一个形状为4的张量，张量的每个元素都是0。
torch.save([x,y],'x-files')
#将张量x和张量y保存到同一个文件中。
x2,y2=torch.load('x-files')
#从文件'x-files'中加载数据，将其分别赋值给变量x2和y2。
print("x2:",x2,"y2:",y2)
mydict = {'x': x, 'y': y}
#创建一个字典，其中包含张量x和张量y。字典的键值分别为'x'和'y'。
torch.save(mydict, 'mydict')
#将字典mydict保存到文件'mydict'中。
mydict2 = torch.load('mydict')
#从文件'mydict'中加载数据，将其赋值给变量mydict2。
print("mydict2:",mydict2)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20,256)
        self.out = nn.Linear(256,10)
    def forward(self,X):
        return self.out(F.relu(self.hidden(X)))
net=MLP()
X=torch.randn(size=(2,20))
Y=net(X)
#通过多层感知机完成前向计算
torch.save(net.state_dict(),'mlp.params')
#将多层感知机的模型参数保存到文件'mlp.params'中。
clone=MLP()
#实例化了一个新的多层感知机模型，它的模型参数是随机初始化的。
clone.load_state_dict(torch.load('mlp.params'))
#使用从文件中加载的模型参数，初始化了这个模型的参数。
print(clone.eval())
print("clone(X):",clone(X))
