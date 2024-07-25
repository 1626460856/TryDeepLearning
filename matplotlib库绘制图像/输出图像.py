import matplotlib.pyplot as plt
def show_it(name, tensor):
    """
    输入一个像素张量，返回一个图像
    :param name: 图像的名称
    :param tensor: 输入的像素张量，形状为 (H, W) 或 (C, H, W) 或 (N, C, H, W)
    """
    tensor = tensor.cpu().numpy()

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