import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from torchvision import transforms


class GeneralizedDataset:
    #定义一个名为GeneralizedDataset的类，这是一个通用数据集类
    """
    Main class for Generalized Dataset.
    """
    #初始化方法，设置最大工作线程数max_workers和是否输出详细信息verbose。
    def __init__(self, max_workers=2, verbose=False):
        self.max_workers = max_workers
        self.verbose = verbose
    #获取数据集中的第i个元素，返回图像和目标（标签）。
    #img_id是图像ID，get_image方法获取图像，transforms.ToTensor将图像转换为张量，get_target方法获取目标标签。
    def __getitem__(self, i):
        img_id = self.ids[i]
        image = self.get_image(img_id)
        image = transforms.ToTensor()(image)
        target = self.get_target(img_id) if self.train else {} # Get目标方法在子类中定义
        return image, target   
    #返回数据集的长度，即图像ID的数量
    def __len__(self):
        return len(self.ids)
    #定义check_dataset方法，使用多线程加速检查数据集，避免_check方法中列出的问题。
    def check_dataset(self, checked_id_file):
        """
        use multithreads to accelerate the process.
        check the dataset to avoid some problems listed in method `_check`.
        """
        #果checked_id_file文件存在，读取文件内容并解析，设置self.ids和self.aspect_ratios，然后返回。
        if os.path.exists(checked_id_file):
            info = [line.strip().split(", ") for line in open(checked_id_file)]
            self.ids, self.aspect_ratios = zip(*info)
            return
        #记录开始时间并打印“Checking the dataset...”信息。
        since = time.time()
        print("Checking the dataset...")
        #创建线程池执行器executor，将数据集分成max_workers块，并为每块数据提交一个线程任务。
        executor = ThreadPoolExecutor(max_workers=self.max_workers)
        seqs = torch.arange(len(self)).chunk(self.max_workers)
        tasks = [executor.submit(self._check, seq.tolist()) for seq in seqs]
        #收集所有线程任务的结果并合并到outs列表中。
        outs = []
        for future in as_completed(tasks):
            outs.extend(future.result())
        if not hasattr(self, "id_compare_fn"):
            self.id_compare_fn = lambda x: int(x)
        outs.sort(key=lambda x: self.id_compare_fn(x[0]))
        # 如果没有定义id_compare_fn，则设置为将ID转换为整数的函数，并按ID对结果进行排序。
        with open(checked_id_file, "w") as f:
            for img_id, aspect_ratio in outs:
                f.write("{}, {:.4f}\n".format(img_id, aspect_ratio))
        #将结果写入checked_id_file文件中。
        info = [line.strip().split(", ") for line in open(checked_id_file)]
        self.ids, self.aspect_ratios = zip(*info)
        print("checked id file: {}".format(checked_id_file))
        print("{} samples are OK; {:.1f} seconds".format(len(self), time.time() - since))
        
    def _check(self, seq):
        out = [] # 初始化一个空列表out，用于存储检查结果。
        for i in seq:
            # 获取图像ID。
            img_id = self.ids[i]
            # 获取目标标签。
            target = self.get_target(img_id)
            # 获取目标标签中的边界框、标签和掩码。
            boxes = target["boxes"]
            labels = target["labels"]
            masks = target["masks"]

            try:
                # 检查边界框的数量是否大于0。
                assert len(boxes) > 0, "{}: len(boxes) = 0".format(i)
                # 检查边界框的数量是否等于标签的数量。
                assert len(boxes) == len(labels), "{}: len(boxes) != len(labels)".format(i)
                # 检查边界框的数量是否等于掩码的数量。
                assert len(boxes) == len(masks), "{}: len(boxes) != len(masks)".format(i)

                # 将图像ID和对应的宽高比添加到out列表中。
                out.append((img_id, self._aspect_ratios[i]))
            except AssertionError as e:
                # 如果检查失败且verbose为True，打印错误信息。
                if self.verbose:
                    print(img_id, e)
        # 返回检查结果。
        return out

                    