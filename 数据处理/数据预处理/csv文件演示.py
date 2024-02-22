import os
import csv
import pandas as pd
import torch

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
# 创建一个名为data的目录（如果不存在），其中exist_ok=True表示如果目录已经存在则不会引发错误。
data_file = os.path.join('..', 'data', 'house_tiny.csv')
# 将文件路径'../data/house_tiny.csv存储在变量data_file中。
with open(data_file, 'w') as f:  # 打开house_tiny.csv文件以供写入。
    f.write('NumRooms,Alley,Price\n')  # 写入列名
    f.write('NA,Pave,127500\n')  # 写入第一行数据
    f.write('2,NA,106000\n')  # 写入第二行数据
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
    # 在填充数据时，"NA"通常用来表示该位置缺少有效值
data = pd.read_csv(data_file)
# 使用pandas库中的read_csv函数来读取名为data_file的CSV文件，并将其内容加载到名为data的DataFrame中
print(data)
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# 将DataFrame data 中的前两列作为输入数据存储在 inputs 变量中，将第三列作为输出数据存储在 outputs 变量中。
inputs = inputs.fillna(inputs.mean())
# 用输入数据的均值填充输入数据中的缺失值
print(inputs)
inputs = pd.get_dummies(inputs, dummy_na=True)
# 使用pd.get_dummies函数将inputs中的分类变量进行独热编码，并且设置dummy_na=True以处理缺失值情况
print(inputs)
x, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
# 将inputs中的数值转换为PyTorch张量x,将outputs中的数值转换为PyTorch张量y
print(x)
print(y)
