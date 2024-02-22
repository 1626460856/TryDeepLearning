import matplotlib.pyplot as plt
import numpy as np
from mxnet import autograd, np, npx
npx.set_np()
def f(x):
    return np.sin(x)
# 生成自变量数据
x = np.linspace(-5, 5, 1000)  # 生成在区间[-5, 5]上均匀分布的1000个点

# 创建 autograd 记录梯度
x.attach_grad()
with autograd.record():
    y =f(x)
    y.backward()
    # 这个操作告诉 MXNet 计算引擎要对变量 y 所代表的函数进行反向传播，以计算出关于变量 x 的梯度。
    print(x.grad)

# 绘制图像
plt.plot(x, y)
# 绘制导数图像
plt.plot(x, x.grad)
# 显示图像
plt.grid(True)  # 添加网格线
plt.show()
