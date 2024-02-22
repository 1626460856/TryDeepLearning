
from mxnet import autograd, np, npx

npx.set_np()
# 创建向量变量
x = np.arange(5.0)

# 创建 autograd 记录梯度
x.attach_grad()

with autograd.record():
    # 调用函数
    y = x * x
# 当对向量值变量y（关于x的函数）调用backward时，将通过对y中的元素求和来创建
# 一个新的标量变量。然后计算这个标量变量相对于x的梯度
# 计算梯度
y.backward()

# 输出梯度
print(x.grad)  # dy/dx
