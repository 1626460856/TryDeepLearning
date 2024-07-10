import random
from mxnet import npx
# 在pycharm的IDE之中不能写from mxnet import npx，np而是要单独导入numpy库，我也不知道原因，不这样的话下面随机数生成不对
import numpy as np
from d2l import mxnet as d2l

# 调用此函数使mxnet的np模块与numpy兼容
npx.set_np()

#fair_probs = [1.0 / 6] * 6
fair_probs = [0.1,0.1,0.2,0.3,0.2,0.1]
# 定义了一个包含 6 个元素的列表，每个元素的取值都是 1/6，表示一个公平的六面骰子的各个面出现的概率

counts = np.random.multinomial(100, fair_probs, size=500)
# 该函数会根据多项分布在定义的元素列表里抽取n个样本，返回一个列表，
# 列表中有 6 个元素，表示每个类别被抽中的次数,size是返回列表的尺寸

print(counts[0])
cum_counts = counts.astype(np.float32).cumsum(axis=0)
# 将counts转换为float32类型，并对每一列进行累计求和，得到累计频率
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
# 这个方法将对数组进行求和，但是可以指定在哪个轴上进行求和。在这里，axis=1 表示沿着数组的第二维（也就是行）进行求和，即将每行的元素相加。
# 计算估计概率。将累计频率除以每一行的累计和，得到每个结果在每组实验中的估计概率

d2l.set_figsize((6, 4.5))
# 设置绘图窗口的尺寸为(6, 4.5)

# 循环遍历6个结果（骰子点数）
for i in range(6):
    d2l.plt.plot(estimates[:, i],
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()  # 显示图例，即标注每条线代表什么
d2l.plt.show()
