import torch
from torch import nn
from d2l import torch as d2l
from torch.onnx.symbolic_opset9 import detach
from torch.xpu import device
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')



def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape # 提取卷积核的高和宽
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1),device=X.device) # Y是输出张量
    for i in range(Y.shape[0]): # 遍历Y的每一行
        for j in range(Y.shape[1]): # 遍历Y的每一列
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() # 计算Y的每个元素
    return Y
X=torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]])
K=torch.tensor([[0.0,1.0],[2.0,3.0]])
print("3x3-X:\n",X,"\n2x2-K:\n",K,"\n2x2-Y:\n",corr2d(X,K))
print("################################################################################################################")
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super().__init__()
        self.weight=nn.Parameter(torch.rand(kernel_size,device=try_gpu())) # 通过 nn.Parameter 创建一个形状为 kernel_size 的张量作为卷积核的权重参数
        self.bias=nn.Parameter(torch.zeros(1,device=try_gpu())) # 创建一个形状为 1 的张量作为卷积核的偏置参数
    def forward(self,X):
        return corr2d(X,self.weight)+self.bias

X=torch.ones((6,8),device=try_gpu())
X[:,2:6]=0
print("6x8-X:\n",X)
K=torch.tensor([[1.0,-1.0]],device=try_gpu())
Y=corr2d(X,K)
print("Y:\n",Y)
print("################################################################################################################")
conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False).to(try_gpu())
# 创建一个二维卷积层 conv2d，输入和输出通道数均为1，卷积核大小为 (1, 2)，不使用偏置。并将该卷积层移动到 GPU 上。
X=X.reshape((1,1,6,8)) # 将输入张量 X 重新调整形状为 (1, 1, 6, 8)，表示批量大小为1，通道数为1，高度为6，宽度为8。
Y=Y.reshape((1,1,6,7)) # 将目标张量 Y 重新调整形状为 (1, 1, 6, 7)，表示批量大小为1，通道数为1，高度为6，宽度为7。
for i in range(10):
    Y_hat=conv2d(X) # 前向计算
    l=(Y_hat-Y)**2 # 计算预测输出 Y_hat 和目标输出 Y 之间的平方误差 l。
    conv2d.zero_grad() # 梯度清零
    l.sum().backward() # 反向传播
    conv2d.weight.data[:]-=3e-2*conv2d.weight.grad # 使用学习率 3e-2 手动更新卷积层的权重。
    if (i+1)%2==0: # 每2次迭代打印一次误差
        print("训练得到的卷积核为：",conv2d.weight.data.reshape((1,2)))
        print(f'batch {i+1},loss {l.sum():.3f}')