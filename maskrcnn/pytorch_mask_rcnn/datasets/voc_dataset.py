import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
from collections import defaultdict

import torch
import numpy as np
import pycocotools.mask as mask_util
from torchvision import transforms

from .generalized_dataset import GeneralizedDataset


VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
) # VOC数据集的类别名称

def target_to_coco_ann(target):  # 将目标转换为COCO格式的注释
    image_id = target['image_id'].item() # 获取图像ID
    boxes = target['boxes'] # 获取边界框
    masks = target['masks'] # 获取掩码
    labels = target['labels'].tolist() # 获取标签并转换为列表

    xmin, ymin, xmax, ymax = boxes.unbind(1) # 解绑边界框
    boxes = torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1) # 转换边界框格式
    area = boxes[:, 2] * boxes[:, 3] # 计算面积
    area = area.tolist() # 转换为列表
    boxes = boxes.tolist() # 转换为列表
    
    rles = [ # 将掩码转换为RLE格式
        mask_util.encode(np.array(mask[:, :, None], dtype=np.uint8, order='F'))[0]
        for mask in masks
    ]
    for rle in rles: # 将RLE格式的counts字段解码为字符串
        rle['counts'] = rle['counts'].decode('utf-8')

    anns = [] # 初始化注释列表
    for i, rle in enumerate(rles): # 遍历掩码和RLE格式
        anns.append(
            {
                'image_id': image_id, # 图像ID
                'id': i, # 注释ID
                'category_id': labels[i], # 类别ID
                'segmentation': rle, # 掩码
                'bbox': boxes[i], # 边界框
                'area': area[i],    # 面积
                'iscrowd': 0, # 是否为困难样本
            }
        )
    return anns     


class VOCDataset(GeneralizedDataset): # 定义一个名为VOCDataset的类，继承自GeneralizedDataset。
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, split, train=False): # 定义类的初始化方法。
        super().__init__() # 调用父类的初始化方法。
        self.data_dir = data_dir # 存储数据目录路径。
        self.split = split # 存储数据集的划分。
        self.train = train # 存储是否为训练集。
        
        # instances segmentation task
        id_file = os.path.join(data_dir, "ImageSets/Segmentation/{}.txt".format(split)) # 存储ID文件路径
        self.ids = [id_.strip() for id_ in open(id_file)] # 获取所有图像的ID，并存储在列表中
        self.id_compare_fn = lambda x: int(x.replace("_", "")) # 定义ID比较函数
        
        self.ann_file = os.path.join(data_dir, "Annotations/instances_{}.json".format(split)) # 存储注释文件路径
        self._coco = None # 初始化COCO对象
        
        # classes's values must start from 1, because 0 means background in the model
        self.classes = {i: n for i, n in enumerate(VOC_CLASSES, 1)} # 获取所有类别的ID和名称，并存储在字典中。类别ID从1开始，因为0表示背景。
        
        checked_id_file = os.path.join(os.path.dirname(id_file), "checked_{}.txt".format(split)) # 存储检查后的ID文件路径
        if train: # 如果是训练集
            if not os.path.exists(checked_id_file): # 如果检查后的ID文件不存在
                self.make_aspect_ratios() # 计算所有图像的宽高比
            self.check_dataset(checked_id_file) # 检查数据集
            
    def make_aspect_ratios(self): # 定义计算宽高比的方法
        self._aspect_ratios = [] # 初始化宽高比列表
        for img_id in self.ids: # 遍历所有图像的ID
            anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id))) # 解析XML文件
            size = anno.findall("size")[0] # 获取图像尺寸
            width = size.find("width").text # 获取图像宽度
            height = size.find("height").text # 获取图像高度
            ar = int(width) / int(height) # 计算宽高比
            self._aspect_ratios.append(ar) # 添加宽高比到列表中

    def get_image(self, img_id): # 定义获取图像的方法
        image = Image.open(os.path.join(self.data_dir, "JPEGImages/{}.jpg".format(img_id))) # 打开图像文件
        return image.convert("RGB") # 将图像转换为RGB格式
        
    def get_target(self, img_id):
        masks = Image.open(os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id))) # 打开掩码文件
        masks = transforms.ToTensor()(masks) # 将掩码转换为张量
        uni = masks.unique() # 获取掩码的唯一值
        uni = uni[(uni > 0) & (uni < 1)] # 获取掩码的唯一值
        masks = (masks == uni.reshape(-1, 1, 1)).to(torch.uint8) # 将掩码转换为二值掩码
        
        anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id))) # 解析XML文件
        boxes = [] # 存储边界框
        labels = [] # 存储标签
        for obj in anno.findall("object"): # 遍历所有对象
            bndbox = obj.find("bndbox") # 获取边界框
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]] # 获取边界框坐标
            name = obj.find("name").text # 获取类别名称
            label = VOC_CLASSES.index(name) + 1 # 获取类别ID

            boxes.append(bbox) # 添加边界框
            labels.append(label) # 添加标签

        boxes = torch.tensor(boxes, dtype=torch.float32) # 转换边界框为张量
        labels = torch.tensor(labels) # 转换标签为张量

        img_id = torch.tensor([self.ids.index(img_id)]) # 获取图像ID
        target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks) # 创建目标字典
        return target # 返回目标字典
    
    @property # 将方法转换为只读属性
    def coco(self): # 定义获取COCO对象的方法
        if self._coco is None: # 如果COCO对象为空
            from pycocotools.coco import COCO # 从pycocotools库中导入COCO类
            self.convert_to_coco_format() # 将数据集转换为COCO格式
            self._coco = COCO(self.ann_file) # 使用COCO类加载注释文件
        return self._coco # 返回COCO对象
    
    def convert_to_coco_format(self, overwrite=False): # 定义将数据集转换为COCO格式的方法
        if overwrite or not os.path.exists(self.ann_file): # 如果覆盖或注释文件不存在
            print("Generating COCO-style annotations...") # 打印信息
            voc_dataset = VOCDataset(self.data_dir, self.split, True) # 创建VOCDataset对象
            instances = defaultdict(list) # 创建默认字典
            instances["categories"] = [{"id": i + 1, "name": n} for i, n in enumerate(VOC_CLASSES)] # 添加类别信息

            ann_id_start = 0 # 初始化注释ID
            for image, target in voc_dataset: # 遍历数据集
                image_id = target["image_id"].item() # 获取图像ID

                filename = voc_dataset.ids[image_id] + ".jpg" # 获取文件名
                h, w = image.shape[-2:] # 获取图像的高度和宽度
                img = {"id": image_id, "file_name": filename, "height": h, "width": w} # 创建图像字典
                instances["images"].append(img) # 添加图像字典

                anns = target_to_coco_ann(target) # 将目标转换为COCO格式的注释
                for ann in anns: # 遍历注释
                    ann["id"] += ann_id_start # 更新注释ID
                    instances["annotations"].append(ann) # 添加注释
                ann_id_start += len(anns) # 更新注释ID

            json.dump(instances, open(self.ann_file, "w")) # 将COCO格式的注释保存到文件中
            print("Created successfully: {}".format(self.ann_file)) # 打印信息
        
  