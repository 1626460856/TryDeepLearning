
import torch

x = torch.arange(12.0).reshape(2, 6)
z=torch.arange(6.0).reshape(3, 2)
print("x:",x)
print("z:",z)
x.requires_grad_(True)
y = torch.matmul(z,x)
print("y.shape:",y.shape)
print("对于结果矩阵 y 中的第 i 行第 j 列的元素 y[i][j]，计算方法为：")
print("y[i][j] = z[i][0] * x[0][j] + z[i][1] * x[1][j] + z[i][2] * x[2][j]")
print("Y:",y)
# 创建一个和 y 同样形状的张量，用于表示梯度
gradient = torch.ones_like(y)

# 对 y 进行反向传播，传入自定义的梯度
y.backward(gradient)

# 输出 x 的梯度
print("x.grad:", x.grad)
