from d2l import torch

import torch

A = torch.arange(20).reshape(5, 4)
print("A:", A)

# 矩阵按行求和
sum_A2 = A.sum(axis=1)
print("sum_A2:", sum_A2)
print("sum_A2 shape:", sum_A2.shape)

# 矩阵按行求和并保持维度
sum_A = A.sum(axis=1, keepdims=True)
print("sum_A:", sum_A)
print("sum_A shape:", sum_A.shape)

# 矩阵按指定轴求和
B = torch.arange(20).reshape(2, 2, 5)
print("B:",B)
# 按指定轴对 B 进行求和 第0轴和第2轴
sum_B = B.sum(axis=[0, 2], keepdims=True)
print("sum_B:", sum_B)

# 按指定轴对 B 进行求和（0，1，2）
sum_B2 = B.sum(axis=[0, 1, 2], keepdims=True)
print("sum_B2:", sum_B2)

# 使用列表生成器生成轴范围，然后对 B 进行求和
sum_B3 = B.sum(axis=list(range(0, 3)), keepdims=True)
print("sum_B3:", sum_B3)
