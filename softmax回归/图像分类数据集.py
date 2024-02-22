import os
import sys

import torch
import torchvision
from mxnet import gluon
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from d2l import torch as d2l

# 设置环境变量
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

d2l.use_svg_display()  # 它用于指示该教材在显示图形时使用 SVG（可缩放矢量图形）格式进行显示。
trans = transforms.ToTensor()  # 这个转换操作通常用于将图像数据准备成神经网络模型的输入，因为大多数深度学习模型都要求输入是张量格式
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
# mnist_train 包含 60000 个样本，每个样本由一个长度为 2 的元组组成，第一个元素是图像数据，第二个元素是对应的标签。
# 每个图像数据是一个张量，其形状为 [1, 28, 28]，表示该图像是单通道的，大小为 $28 \times 28$ 像素的灰度图像。
# mnist_test 包含 10000 个样本，同样每个样本由一个长度为 2 的元组组成，第一个元素也是图像数据，第二个元素是对应的标签。
# 图像数据的形状也是 [1, 28, 28]，表示该图像是单通道的，大小为 $28 \times 28$ 像素的灰度图像。
print(len(mnist_train), len(mnist_test))
print(len(mnist_train[0]), len(mnist_test[0]))
print(mnist_train[0][0].shape)
print(mnist_test[0][0].shape)
# 下面我们可以具体把这个数据的图片拿出来看一眼
import matplotlib.pyplot as plt

# 获取图像数据和标签
image, label = mnist_test[0]

# 将图像数据从张量转换为 NumPy 数组，并且将通道维度移到最后一个维度
image = image.squeeze().numpy()

# 显示图像
plt.imshow(image, cmap='gray')
plt.title(f'Label: {label}')
plt.axis('off')  # 关闭坐标轴
plt.show()


# *************************************************************************************************
# 下面这个显示图像可能正规一点吧，无所谓啦，目的只是拿来看看
# 返回Fashion-MNIST数据集的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 绘制图像列表
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


# 创建数据加载器
train_loader = DataLoader(mnist_train, batch_size=18)

# 获取一批数据
X_batch, y_batch = next(iter(train_loader))
print(X_batch.shape)  # 输出：torch.Size([18, 1, 28, 28])

# 显示图像
show_images(X_batch.squeeze(axis=1), 2, 9, titles=get_fashion_mnist_labels(y_batch))
plt.show()
# **************************************************************************************************
# 读取小批量数据
batch_size = 256


# """使用4个进程来读取数据"""
def get_dataloader_workers():
    return 0


# 通过ToTensor实例将图像数据从uint8格式变换成32位浮点数格式，并除以255使得所有像素的数值
# 均在0～1之间
# 在上文读取代码前的转化器中——trans = transforms.ToTensor()
train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers())
# 创建了一个数据加载器train_iter，它将Fashion-MNIST训练集mnist_train分成了大小为256的小批量数据，
# 并且进行了打乱处理（shuffle=True），同时使用了4个进程来读取数据。
timer = d2l.Timer()
for X, y in train_iter:
    continue
print(f'{timer.stop():.2f} sec')


# #@save
# 这个函数的主要目的是加载Fashion-MNIST数据集并返回训练集和测试集的数据加载器
# 返回训练集和测试集的数据加载器，其中数据加载器通过DataLoader类构建，用于以小批量方式加载数据
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))


train_data = load_data_fashion_mnist(10, resize=None)
# next 函数用于获取迭代器的下一个元素，而 iter 函数则用于创建一个迭代器。
# 因此，next(iter(train_data[0])) 返回的是训练集数据加载器的第一个批量数据,和前面线性回归的小批量数据调用是一致的方式
print(next(iter(train_data[0])))