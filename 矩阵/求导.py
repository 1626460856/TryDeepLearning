import torch

print("示例求导，函数y=2xTx，关于列向量x进行求导")

x = torch.arange(4.0)
print("创建列向量x:", x)

print("定义x之后创建x梯度追踪，以后用x的gard属性对梯度进行访问")
x.requires_grad_(True)

y = 2 * torch.dot(x, x)
print("定义y函数y=2xTx,并对y进行计算:", y)

print("调用反向传播函数，追踪计算y过程中的每个x的梯度")
y.backward()
print(x.grad)
print("实际计算梯度是dy/dx=4x，依此来进行验证:")
print(x.grad == 4 * x)

print("计算x的另一个函数的时候要把x追踪的梯度清0")
x.grad.zero_()
print("第二个函数y=sum（x），跟踪x计算y：")
y=x.sum()
y.backward()
print("第二个函数跟踪的各个x梯度：",x.grad)
print("把清零梯度的函数注释掉再试一次")
