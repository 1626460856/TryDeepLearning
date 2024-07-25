import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
def show_it(name, tensor):
    """
    输入一个像素张量，返回一个图像
    :param name: 图像的名称
    :param tensor: 输入的像素张量，形状为 (H, W) 或 (C, H, W) 或 (N, C, H, W)
    """
    tensor = tensor.cpu().numpy()

    if tensor.ndim == 2:
        # 2D tensor
        plt.imshow(tensor, cmap='gray')
        plt.title(name)
        plt.axis('off')
        plt.show()
    elif tensor.ndim == 3:
        # 3D tensor (C, H, W)
        C, H, W = tensor.shape
        fig, axes = plt.subplots(1, C, figsize=(C * 3, 3))
        for i in range(C):
            axes[i].imshow(tensor[i], cmap='gray')
            axes[i].set_title(f"{name} - Channel {i}")
            axes[i].axis('off')
        plt.show()
    elif tensor.ndim == 4:
        # 4D tensor (N, C, H, W)
        N, C, H, W = tensor.shape
        fig, axes = plt.subplots(N, C, figsize=(C * 3, N * 3))
        if N == 1:
            axes = [axes]
        if C == 1:
            axes = [[ax] for ax in axes]
        for n in range(N):
            for c in range(C):
                axes[n][c].imshow(tensor[n, c], cmap='gray')
                axes[n][c].set_title(f"{name} - Batch {n} - Channel {c}")
                axes[n][c].axis('off')
        plt.show()


def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# 将张量移动到 GPU
device = try_gpu()
print("device:",device)
def pool2d(X,pool_size,mode='max'):
    """二维池化层的简单实现"""
    p_h,p_w=pool_size # 池化窗口的高和宽
    Y=torch.zeros((X.shape[0]-p_h+1,X.shape[1]-p_w+1),device=device) # 输出张量 Y 的形状
    for i in range(Y.shape[0]): # 遍历 Y 的每一行
        for j in range(Y.shape[1]): # 遍历 Y 的每一列
            if mode=='max': # 最大池化
                Y[i,j]=X[i:i+p_h,j:j+p_w].max() # Y(i, j) 是 X(i:i+p_h, j:j+p_w) 的最大值
            elif mode=='avg': # 平均池化
                Y[i,j]=X[i:i+p_h,j:j+p_w].mean() # Y(i, j) 是 X(i:i+p_h, j:j+p_w) 的平均值
    return Y
X=torch.tensor([[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]],device=device)
print("验证最大,平均池化层:")
print("3x3-X:\n",X)
print("2x2-max_pool:\n",pool2d(X,(2,2)))
print("2x2-avg_pool:\n",pool2d(X,(2,2),'avg'))
print("################################################################################################################")
X=torch.arange(64,dtype=torch.float32,device=device).reshape((2,2,4,4))
# 创建一个 4x4 的输入张量,并将其变形为四维张量批量为1，通道为1
pool2d=nn.MaxPool2d(3,1,(1,1)) # 创建一个最大池化层，池化窗口形状为 3x3，步长为 1,填充为 1
print("演示填充与步幅")
print("4x4-X:\n",X)

print("max_pool:\n",pool2d(X))
show_it("X",X)
show_it("pool(X)",pool2d(X))






















