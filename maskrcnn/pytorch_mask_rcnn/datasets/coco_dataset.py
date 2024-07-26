import os
# 导入操作系统相关的模块，用于处理文件和目录路径。
from PIL import Image
# 从PIL（Python Imaging Library）库中导入Image模块，用于图像处理。

import torch
# 导入PyTorch库，用于深度学习和张量操作。

from .generalized_dataset import GeneralizedDataset
# 从当前包中导入GeneralizedDataset类，这是一个通用的数据集类，COCODataset将继承它。

        
class COCODataset(GeneralizedDataset):  # 定义一个名为COCODataset的类，继承自GeneralizedDataset。
    def __init__(self, data_dir, split, train=False): # 定义类的初始化方法。
        super().__init__() # 调用父类的初始化方法。
        from pycocotools.coco import COCO # 从pycocotools库中导入COCO类，用于处理COCO数据集。
        self.data_dir = data_dir # 存储数据目录路径。
        self.split = split # 存储数据集的划分。
        self.train = train # 存储是否为训练集。
        
        ann_file = os.path.join(data_dir, "annotations/instances_{}.json".format(split)) # 存储标注文件路径。
        self.coco = COCO(ann_file) # 使用COCO类加载注释文件。
        self.ids = [str(k) for k in self.coco.imgs] # 获取所有图像的ID，并将其转换为字符串列表。
        
        # 类的值必须从1开始，因为0表示模型中的背景
        self.classes = {k: v["name"] for k, v in self.coco.cats.items()}
        # 获取所有类别的ID和名称，并存储在字典中。类别ID从1开始，因为0表示背景。
        checked_id_file = os.path.join(data_dir, "checked_{}.txt".format(split)) # 存储检查后的ID文件路径。
        if train: # 如果是训练集。
            if not os.path.exists(checked_id_file): # 如果检查后的ID文件不存在。
                self._aspect_ratios = [v["width"] / v["height"] for v in self.coco.imgs.values()] # 计算所有图像的宽高比。
            self.check_dataset(checked_id_file) # 检查数据集。

    def get_image(self, img_id): # 定义获取图像的方法。
        img_id = int(img_id) # 将图像ID转换为整数。
        img_info = self.coco.imgs[img_id] # 获取图像信息。
        image = Image.open(os.path.join(self.data_dir, "{}".format(self.split), img_info["file_name"])) # 打开图像文件。
        return image.convert("RGB") # 将图像转换为RGB格式。

    @staticmethod # 静态方法装饰器。
    def convert_to_xyxy(boxes): # 框格式:(xmin, ymin, w, h)
        x, y, w, h = boxes.T # 转置框的坐标。
        return torch.stack((x, y, x + w, y + h), dim=1) # new_box format: (xmin, ymin, xmax, ymax) # 返回新的框格式。
        
    def get_target(self, img_id): # 定义获取目标的方法。
        img_id = int(img_id) # 将图像ID转换为整数。
        ann_ids = self.coco.getAnnIds(img_id) # 获取图像的注释ID。
        anns = self.coco.loadAnns(ann_ids) # 加载注释。
        boxes = [] # 存储框。
        labels = [] # 存储标签。
        masks = [] # 存储掩码。

        if len(anns) > 0: # 如果注释不为空。
            for ann in anns: # 遍历注释。
                boxes.append(ann['bbox']) # 添加边界框。
                labels.append(ann["category_id"]) # 添加标签。
                mask = self.coco.annToMask(ann) # 将注释转换为掩码。
                mask = torch.tensor(mask, dtype=torch.uint8) # 将掩码转换为张量。
                masks.append(mask) # 添加掩码。

            boxes = torch.tensor(boxes, dtype=torch.float32)  # [N, 4]
            boxes = self.convert_to_xyxy(boxes) # 转换框格式。
            labels = torch.tensor(labels) # 转换标签为张量。
            masks = torch.stack(masks) # 将掩码堆叠为张量。

        target = dict(image_id=torch.tensor([img_id]), boxes=boxes, labels=labels, masks=masks) # 创建目标字典。
        return target # 返回目标字典。
    
    