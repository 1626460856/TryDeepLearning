from d2l import torch

# 矩阵的转置
A = torch.arange(20).reshape(5, 4)
print(A)
A = A.T
print(A)

