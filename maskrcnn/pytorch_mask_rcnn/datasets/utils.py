from .voc_dataset import VOCDataset
from .coco_dataset import COCODataset

__all__ = ["datasets", "collate_wrapper"]


def datasets(ds, *args, **kwargs): # 返回数据集
    ds = ds.lower() # 将数据集名称转换为小写。
    choice = ["voc", "coco"] # 数据集名称列表。
    if ds == choice[0]: # 如果数据集名称为voc。
        return VOCDataset(*args, **kwargs) # 返回VOCDataset数据集。
    if ds == choice[1]: # 如果数据集名称为coco。
        return COCODataset(*args, **kwargs) # 返回COCODataset数据集。
    else: # 如果数据集名称不在列表中。
        raise ValueError("'ds' must be in '{}', but got '{}'".format(choice, ds)) # 抛出异常。
    
    
def collate_wrapper(batch):
    return CustomBatch(batch)

    
class CustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.images = transposed_data[0]
        self.targets = transposed_data[1]

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.images = [img.pin_memory() for img in self.images]
        self.targets = [{k: v.pin_memory() for k, v in tgt.items()} for tgt in self.targets]
        return self


