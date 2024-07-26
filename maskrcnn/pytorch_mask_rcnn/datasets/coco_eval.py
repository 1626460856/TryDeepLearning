import copy
import torch
import numpy as np

import pycocotools.mask as mask_util  # 导入pycocotools.mask模块并命名为mask_util
from pycocotools.cocoeval import COCOeval  # 从pycocotools.cocoeval模块导入COCOeval类
from pycocotools.coco import COCO  # 从pycocotools.coco模块导入COCO类


class CocoEvaluator: # 定义一个名为CocoEvaluator的类
    def __init__(self, coco_gt, iou_types="bbox"): # 定义类的初始化方法
        if isinstance(iou_types, str): # 如果iou_types是字符串
            iou_types = [iou_types] # 将iou_types转换为列表
            
        coco_gt = copy.deepcopy(coco_gt) # 深拷贝coco_gt
        self.coco_gt = coco_gt # 存储coco_gt
        self.iou_types = iou_types # 存储iou_types
        #self.ann_labels = ann_labels # 存储ann_labels
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type)
                         for iou_type in iou_types} # 创建COCOeval对象并存储在字典中
        
        self.has_results = False # 初始化has_results为False
            
    def accumulate(self, coco_results): # 定义累积方法
        if len(coco_results) == 0: # 如果coco_results为空
            return # 返回
        
        image_ids = list(set([res["image_id"] for res in coco_results])) # 获取所有图像ID并去重
        for iou_type in self.iou_types: # 遍历iou_types
            coco_eval = self.coco_eval[iou_type] # 获取COCOeval对象
            coco_eval.cocoDt = self.coco_gt.loadRes(coco_results) # 加载预测结果
            coco_eval.params.imgIds = image_ids # 设置参数imgIds
            coco_eval.evaluate() # 评估
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params) # 深拷贝参数

            coco_eval.accumulate()  # 累积
            
        self.has_results = True # 设置has_results为True
    
    def summarize(self): # 定义总结方法
        if self.has_results: # 如果有结果
            for iou_type in self.iou_types: # 遍历iou_types
                print("IoU metric: {}".format(iou_type)) # 打印IoU指标
                self.coco_eval[iou_type].summarize() # 总结
        else:
            print("evaluation has no results") # 打印没有结果
            
            
def prepare_for_coco(predictions): # 定义prepare_for_coco函数
    coco_results = [] # 初始化coco_results为空列表
    for original_id, prediction in predictions.items(): # 遍历predictions
        if len(prediction) == 0: # 如果预测结果为空
            continue # 继续

        boxes = prediction["boxes"] # 获取边界框
        scores = prediction["scores"] # 获取分数
        labels = prediction["labels"] # 获取标签
        masks = prediction["masks"] # 获取掩码

        x1, y1, x2, y2 = boxes.unbind(1) # 解绑边界框
        boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1) # 转换边界框格式
        boxes = boxes.tolist() # 转换为列表
        scores = prediction["scores"].tolist() # 转换为列表
        labels = prediction["labels"].tolist() # 转换为列表

        masks = masks > 0.5 # 获取掩码
        rles = [ # 创建rles列表
            mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
            for mask in masks # 遍历掩码
        ]
        for rle in rles: # 遍历rles
            rle["counts"] = rle["counts"].decode("utf-8") # 解码

        coco_results.extend( # 扩展coco_results
            [
                {
                    "image_id": original_id, # 图像ID
                    "category_id": labels[i], # 类别ID
                    "bbox": boxes[i], # 边界框
                    "segmentation": rle, # 掩码
                    "score": scores[i], # 分数
                }
                for i, rle in enumerate(rles) # 遍历rles
            ]
        )
    return coco_results    # 返回coco_results


    '''
    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))
            
    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for image_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            # convert to coco bbox format: xmin, ymin, w, h
            boxes = prediction["boxes"]
            x1, y1, x2, y2 = boxes.unbind(1)
            boxes = torch.stack((x1, y1, x2 - x1, y2 - y1), dim=1)
            
            boxes = boxes.tolist()
            
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            labels = [self.ann_labels[l] for l in labels]

            coco_results.extend(
                [
                    {
                        "image_id": image_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results
    
    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            labels = [self.ann_labels[l] for l in labels]

            rles = [
                mask_util.encode(np.array(mask[:, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results
    '''
    
