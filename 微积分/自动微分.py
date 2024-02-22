from mxnet import autograd, np, npx

npx.set_np()
# npx.set_np()是MXNet库提供的一个函数，用于设置默认的NumPy后端。它允许您在MXNet中使用类似于NumPy的接口和功能。

# 创建列向量x
x = np.arange(5.0)

# 通过调用attach_grad来为张量的梯度分配内存
x.attach_grad()
# 在计算关于x的梯度后，将能通过‘gard’属性访问它，它的初始值为0
# 将代码放到autograd.record内，建立计算图
with autograd.record():
    y = 2 * np.dot(x, x)
    y.backward()
    # 这个操作告诉 MXNet 计算引擎要对变量 y 所代表的函数进行反向传播，以计算出关于变量 x 的梯度。
    print(x.grad)
# 已知函数y真实的梯度是4x，验证得到的梯度是否正确，
print(x.grad == 4 * x)

with autograd.record():
    y = x.sum()
    y.backward()
print(x.grad)  # 被新计算的梯度覆盖
