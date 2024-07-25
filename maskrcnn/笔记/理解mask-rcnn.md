train.py是是一个用于训练 Mask R-CNN 模型的脚本。
1. 在E盘创建对应的目录，指令如下
mkdir -p E:/PyTorch/data/coco2017/annotations
mkdir -p E:/PyTorch/data/coco2017/train2017
mkdir -p E:/PyTorch/data/coco2017/val2017
2. 下载COCO数据集，指令如下
# 下载 `2017 Train images [118K/18GB]` 到 `E:/PyTorch/data/coco2017/train2017`
curl -o E:/PyTorch/data/coco2017/train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip E:/PyTorch/data/coco2017/train2017.zip -d E:/PyTorch/data/coco2017/train2017

# 下载 `2017 Val images [5K/1GB]` 到 `E:/PyTorch/data/coco2017/val2017`
curl -o E:/PyTorch/data/coco2017/val2017.zip http://images.cocodataset.org/zips/val2017.zip
unzip E:/PyTorch/data/coco2017/val2017.zip -d E:/PyTorch/data/coco2017/val2017

# 下载 `2017 Train/Val annotations [241MB]` 到 `E:/PyTorch/data/coco2017/annotations`
curl -o E:/PyTorch/data/coco2017/annotations/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip E:/PyTorch/data/coco2017/annotations/annotations_trainval2017.zip -d E:/PyTorch/data/coco2017/annotations
3. 压缩文件如下
```
coco2017/
    annotations/
        instances_train2017.json
        instances_val2017.json
        ...
    train2017/
        000000000009.jpg
        ...
    val2017/
        000000000139.jpg
        ...
```
4. 运行训练脚本
![img.png](img.png)
5. 训练过程中，会在`E:/PyTorch/data/coco2017`目录下生成`maskrcnn_coco.pth`文件，该文件是训练好的模型参数。
6. 列出pytorch_mask_rcnn文件夹下的文件目录，指令如下->前缀自行修改->
dir "D:\视觉实验室\try\maskrcnn\pytorch_mask_rcnn" -Recurse
7. 项目文件结构
pytorch_mask_rcnn/
├── datasets/
│   ├── __pycache__/
│   ├── coco_dataset.py-------------------->处理 COCO 数据集的加载和预处理。
│   ├── coco_eval.py----------------------->用于评估 COCO 数据集的模型性能。
│   ├── generalized_dataset.py------------->通用数据集类，用于加载和预处理数据集。
│   ├── utils.py--------------------------->数据集相关的工具函数。
│   ├── voc_dataset.py--------------------->处理 VOC 数据集的加载和预处理。
│   └── __init__.py------------------------>初始化 datasets 模块。
├── model/
│   ├── __pycache__/
│   ├── box_ops.py------------------------->包含边界框操作的函数。
│   ├── mask_rcnn.py----------------------->定义 Mask R-CNN 模型的主要结构和功能。
│   ├── pooler.py-------------------------->实现 RoI 池化操作。
│   ├── roi_heads.py----------------------->定义 RoI 头部的结构和功能。
│   ├── rpn.py----------------------------->实现区域建议网络（RPN）。
│   ├── transform.py----------------------->图像和目标的预处理和后处理。
│   ├── utils.py--------------------------->模型相关的工具函数。
│   └── __init__.py------------------------>初始化 model 模块。
├── __pycache__/
├── engine.py------------------------------>训练和评估的主要逻辑。
├── gpu.py--------------------------------->GPU 相关的操作和工具函数。
├── utils.py------------------------------->通用工具函数。
├── visualizer.py-------------------------->可视化工具函数。
└── __init__.py---------------------------->初始化 pytorch_mask_rcnn 模块。
8. 训练模型脚本
训练模型脚本调用各个文件的流程分析
导入必要的库和模块：  
  导入torch和pytorch_mask_rcnn等库。
  pytorch_mask_rcnn模块包含了项目的所有核心功能。
定义try_gpu函数：  
  检查是否有可用的GPU，如果有则返回GPU设备，否则返回CPU设备。
定义main函数：  
  设备设置：调用try_gpu函数设置设备。
  显示GPU信息：如果使用GPU，调用pmr.get_gpu_prop显示GPU属性。
  准备数据加载器：
    加载训练数据集和验证数据集，使用pmr.datasets函数。
    训练数据集和验证数据集分别存储在d_train和d_test中。
  设置模型和优化器：
    根据数据集的类别数量，初始化Mask R-CNN模型，使用pmr.maskrcnn_resnet50函数。
    设置优化器torch.optim.SGD。
  加载检查点：
    查找并加载最新的检查点文件，恢复模型和优化器的状态。
  训练和评估：
    迭代训练模型，调用pmr.train_one_epoch进行每个epoch的训练。
    评估模型，调用pmr.evaluate进行评估。
    保存检查点，调用pmr.save_ckpt保存模型和优化器的状态。
    删除多余的检查点文件。
主程序入口：  
  使用argparse解析命令行参数。
  调用main函数开始训练。