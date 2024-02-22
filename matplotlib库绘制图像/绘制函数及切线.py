import matplotlib.pyplot as plt
import numpy as np


# 定义自定义函数
def f(x):
    return x ** 3 - 2 * x ** 2 + x + 3

#计算函数在x0处斜率
def k_x0(f,x0):
    n=0.000001
    return (f(x0 + n) - f(x0)) / n

# 生成自变量数据
x = np.linspace(-2, 2, 100)  # 生成在区间[-2, 2]上均匀分布的100个点

# 生成因变量数据
y = f(x)
x0 = 1.9
y2 = k_x0(f,x0)*(x-x0) + f(x0)

# 绘制图像
plt.plot(x, y)
# 绘制切线
plt.plot(x, y2)
#添加横纵坐标的描述
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Graph of the function f(x) = x^3 - 2x^2 + x + 3')

# 显示图像
plt.grid(True)  # 添加网格线
plt.show()
