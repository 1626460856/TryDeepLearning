import torch
from torch import nn

def comp_conv2d(conv2d,X):
    """这是一个简单的函数，用于计算二维卷积层的输出。它接受一个卷积层 conv2d 和一个输入张量 X，然后返回卷积层的输出。"""
    X=X.reshape((1,1)+X.shape) # 将输入张量 X 变形为卷积层要求的四维张量格式
    Y=conv2d(X) # 计算卷积层的输出，这里的 Y 也是一个四维张量
    return Y.reshape(Y.shape[2:]) # 将四维张量 Y 变形为二维张量，返回结果

conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1) # 创建一个二维卷积层，输入和输出通道数均为1，卷积核大小为 3，填充为 1
X=torch.rand(size=(8,8)) # 创建一个 8x8 的随机输入张量 X
print("X尺寸:",X.shape)
print("conv2d(X)尺寸:",comp_conv2d(conv2d,X).shape) # 输出卷积层的输出形状
print("#################################################################################################################")
conv2d=nn.Conv2d(1,1,kernel_size=(5,3),padding=(2,1)) # 创建一个二维卷积层，输入和输出通道数均为1，卷积核大小为 (5, 3)，填充为 (2, 1)
print("conv2d(X)尺寸:",comp_conv2d(conv2d,X).shape) # 输出卷积层的输出形状
print("#################################################################################################################")
conv2d=nn.Conv2d(1,1,kernel_size=3,padding=1,stride=2) # 创建一个二维卷积层，输入和输出通道数均为1，卷积核大小为 3，填充为 1，步幅为 2
print("conv2d(X)尺寸:",comp_conv2d(conv2d,X).shape) # 输出卷积层的输出形状
print("#################################################################################################################")
conv2d=nn.Conv2d(1,1,kernel_size=(3,5),padding=(0,1),stride=(3,4)) # 创建一个二维卷积层，输入和输出通道数均为1，卷积核大小为 (3, 5)，填充为 (0, 1)，步幅为 (3, 4)
print("conv2d(X)尺寸:",comp_conv2d(conv2d,X).shape) # 输出卷积层的输出形状
