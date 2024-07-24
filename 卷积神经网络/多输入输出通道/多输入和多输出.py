import torch
from torch import nn
from d2l import torch as d2l
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# 将张量移动到 GPU
device = try_gpu()
print("device:",device)
def corr2d(X, K):
    """计算二维互相关运算。"""
    h, w = K.shape # 提取卷积核的高和宽
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1),device=X.device) # Y是输出张量，设置在gpu上
    for i in range(Y.shape[0]): # 遍历Y的每一行
        for j in range(Y.shape[1]): # 遍历Y的每一列
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum() # 计算Y的每个元素
    return Y
def corr2d_multi_in(X, K):
    """计算多输入通道的二维互相关运算"""
    return sum(corr2d(x, k) for x, k in zip(X, K))  # 对 X 和 K 中的所有元素分别进行二维互相关运算，然后将所有结果相加



X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]], device=device)
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],
                  [[1.0, 2.0], [3.0, 4.0]]], device=device)

print("2x3x3-X:\n", X, "\n2x2x2-K:\n", K, "\n1x2x2-Y:\n", corr2d_multi_in(X, K))
print("################################################################################################################")
def corr2d_multi_in_out(X, K):
    """计算多输入多输出通道的二维互相关运算"""
    # 对 K 中每个通道，计算多输入通道的二维互相关运算。所有结果都叠加在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

K = torch.stack((K, K + 1, K + 2), 0) # 将 K 和 K+1 连接在一起以构造一个具有 3 个输出通道的卷积核
print("K的形状：\n", K.shape)
print("2x3x3-X:\n", X, "\n3x2x2x2-K:\n", K, "\n3x2x2-Y:\n", corr2d_multi_in_out(X, K))
print("################################################################################################################")
def corr2d_multi_in_out_1x1(X, K):
    """使用 1x1 卷积核进行多输入多输出通道的二维互相关运算"""
    c_i, h, w = X.shape # 输入张量的通道数、高度和宽度
    c_o = K.shape[0] # 输出通道数
    X = X.reshape((c_i, h * w)) # 将输入张量变形为 c_i 行，h * w 列
    K = K.reshape((c_o, c_i)) # 将卷积核变形为 c_o 行，c_i 列
    Y = torch.matmul(K, X)  # 全连接层的矩阵乘法
    return Y.reshape((c_o, h, w)) # 将 Y 变形为输出张量的形状
X=torch.normal(0,1,(3,3,3),device=device) # 创建一个 3x3x3 的输入张量
K=torch.normal(0,1,(2,3,1,1),device=device) # 创建一个 2x3x1x1 的卷积核
Y1=corr2d_multi_in_out_1x1(X,K) # 使用 1x1 卷积核进行多输入多输出通道的二维互相关运算
Y2=corr2d_multi_in_out(X,K) # 使用普通的卷积核进行多输入多输出通道的二维互相关运算

print("Y1:\n",Y1)
print("Y2:\n",Y2)
print("(Y1-Y2).sum().item():", (Y1 - Y2).sum().item())  # 比较两个结果，输出更精确
print("################################################################################################################")

