import subprocess

from conda.gateways.disk.update import touch

# 执行 'nvidia-smi' 命令并获取输出
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)

# 打印命令的输出
print(result.stdout)

import torch
from torch import nn

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu())
import torch
print(torch.cuda.is_available())
X=torch.ones(2,3,device=try_gpu())
print(X)