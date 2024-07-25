import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



def show_it(name, tensor):
    """
    输入一个像素张量，返回一个图像
    :param name: 图像的名称
    :param tensor: 输入的像素张量，形状为 (H, W) 或 (C, H, W) 或 (N, C, H, W)
    """
    tensor = tensor.detach().cpu().numpy()

    if tensor.ndim == 2:
        # 2D tensor
        plt.imshow(tensor, cmap='gray')
        plt.title(name)
        plt.axis('off')
        plt.show()
    elif tensor.ndim == 3:
        # 3D tensor (C, H, W)
        C, H, W = tensor.shape
        fig, axes = plt.subplots(1, C, figsize=(C * 3, 3))
        for i in range(C):
            axes[i].imshow(tensor[i], cmap='gray')
            axes[i].set_title(f"{name} - Channel {i}")
            axes[i].axis('off')
        plt.show()
    elif tensor.ndim == 4:
        # 4D tensor (N, C, H, W)
        N, C, H, W = tensor.shape
        fig, axes = plt.subplots(N, C, figsize=(C * 3, N * 3))
        if N == 1:
            axes = [axes]
        if C == 1:
            axes = [[ax] for ax in axes]
        for n in range(N):
            for c in range(C):
                axes[n][c].imshow(tensor[n, c], cmap='gray')
                axes[n][c].set_title(f"{name} - Batch {n} - Channel {c}")
                axes[n][c].axis('off')
        plt.show()

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')
# 将张量移动到 GPU
device = try_gpu()
print("device:",device)
class Reshape(torch.nn.Module):
    """将图像大小重定型为给定的大小。"""
    def forward(self, x): # x 的形状：(b, c, h, w)
        return x.view(-1,1,28,28) # 输出形状：(b, 1, 28, 28) b是batch_size
net=torch.nn.Sequential(
    Reshape(), # 重定型
    nn.Conv2d(1,6,kernel_size=5,padding=2,device=device),nn.Sigmoid(), # 第一个卷积层+激活函数
    nn.AvgPool2d(kernel_size=2,stride=2), # 第一个池化层->均值池化层
    nn.Conv2d(6,16,kernel_size=5,device=device),nn.Sigmoid(), # 第二个卷积层+激活函数
    nn.AvgPool2d(kernel_size=2,stride=2), # 第二个池化层->均值池化层
    nn.Flatten(), # 将四维的输出转换成二维的输出，其形状为(批量大小, 通道, 高, 宽)
    nn.Linear(16*5*5,120,device=device),nn.Sigmoid(), # 第一个全连接层->批量*一维张量
    nn.Linear(120,84,device=device),nn.Sigmoid(), # 第二个全连接层
    nn.Linear(84,10,device=device) # 输出层
)
x=torch.rand(size=(4,1,28,28),dtype=torch.float32,device=device) # 输入数据 x 的形状：(b, c, h, w)
print("以下是网络中每一层的输出形状->")
for layer in net:
    x=layer(x) # x 作为输入，计算输出
    #show_it("layer(X)",x)
    print(layer.__class__.__name__,'output shape:\t',x.shape) # 打印每一层的输出形状
print("################################################################################################################")
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size=batch_size)
def evaluate_accuracy_gpu(net,data_iter,device=None):
    """使用GPU计算模型在数据集上的精度。"""
    if isinstance(net,torch.nn.Module):
        net.eval() # 将模型设置为评估模式
        if not device:
            device=next(iter(net.parameters())).device # 如果没有指定device，则使用net的device
    # 正确预测的数量，总预测的数量
    metric=d2l.Accumulator(2)
    for X,y in data_iter:
        if isinstance(X,list):
            # BERT微调所需的（之后将介绍）
            X=[x.to(device) for x in X]
        else:
            X=X.to(device)
        y=y.to(device)
        metric.add(d2l.accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]


#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):  # 初始化权重
        if type(m) == nn.Linear or type(m) == nn.Conv2d:  # 如果是线性层或卷积层
            nn.init.xavier_uniform_(m.weight) # 使用均匀分布的Xavier初始化权重
    net.apply(init_weights) # 对网络应用初始化权重
    print('training on', device) # 输出当前训练设备
    net.to(device) # 将网络移动到设备上
    optimizer = torch.optim.SGD(net.parameters(), lr=lr) # 使用随机梯度下降优化器
    loss = nn.CrossEntropyLoss() # 使用交叉熵损失函数
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc']) # 创建动画
    timer, num_batches = d2l.Timer(), len(train_iter) # 计时器，获取训练数据集的批次数量
    for epoch in range(num_epochs): # 训练轮数
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3) # 训练损失之和，训练准确率之和，样本数
        net.train() # 将网络设置为训练模式
        for i, (X, y) in enumerate(train_iter): # 遍历训练数据集
            timer.start() # 计时器开始
            optimizer.zero_grad() # 梯度清零
            X, y = X.to(device), y.to(device) # 将数据移动到设备上
            y_hat = net(X) # 计算预测值
            l = loss(y_hat, y) # 计算损失
            l.backward() # 反向传播
            optimizer.step() # 更新参数
            with torch.no_grad(): # 不追踪梯度
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0]) # 累加损失，准确率，样本数
            timer.stop() # 计时器停止
            train_l = metric[0] / metric[2] # 计算平均损失
            train_acc = metric[1] / metric[2] # 计算平均准确率
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1: # 每5批次或最后一批次
                animator.add(epoch + (i + 1) / num_batches, # 添加动画
                             (train_l, train_acc, None)) # 添加动画
        test_acc = evaluate_accuracy_gpu(net, test_iter)    # 计算测试集准确率
        animator.add(epoch + 1, (None, None, test_acc)) # 添加动画

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, ' # 输出训练损失，训练准确率
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec ' # 输出每秒处理的样本数
          f'on {str(device)}')
    plt.show()


lr,num_epochs=0.9,10
train_ch6(net,train_iter,test_iter,num_epochs,lr,device=device)
