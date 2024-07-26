import pyvista as pv
import numpy as np


def create_cube(size, color):
    """创建一个指定尺寸和颜色的立方体"""
    cube = pv.Cube(center=(0, 0, 0), x_length=size, y_length=size, z_length=size)
    return cube, color


def create_line(start, end, color):
    """创建一条从 start 到 end 的直线"""
    line = pv.Line(start, end)
    return line, color


def show_net():
    # 创建一个PyVista的Plotter对象
    plotter = pv.Plotter()

    # 定义模型层的参数和坐标
    layers = [
        ("ResBackbone", (2, 2, 2), 1.5),
        ("conv1", (3, 2, 2), 1.2),
        ("bn1", (4, 2, 2), 1.2),
        ("relu", (5, 2, 2), 1.0),
        ("maxpool", (6, 2, 2), 1.2),
        ("layer1", (7, 2, 2), 1.5),
        ("Bottleneck 1 (layer1)", (7, 3, 2), 1.0),
        ("Bottleneck 2 (layer1)", (7, 4, 2), 1.0),
        ("Bottleneck 3 (layer1)", (7, 5, 2), 1.0),
        ("layer2", (8, 2, 2), 1.5),
        ("Bottleneck 1 (layer2)", (8, 3, 2), 1.0),
        ("Bottleneck 2 (layer2)", (8, 4, 2), 1.0),
        ("Bottleneck 3 (layer2)", (8, 5, 2), 1.0),
        ("Bottleneck 4 (layer2)", (8, 6, 2), 1.0),
        ("layer3", (9, 2, 2), 1.5),
        ("Bottleneck 1 (layer3)", (9, 3, 2), 1.0),
        ("Bottleneck 2 (layer3)", (9, 4, 2), 1.0),
        ("Bottleneck 3 (layer3)", (9, 5, 2), 1.0),
        ("Bottleneck 4 (layer3)", (9, 6, 2), 1.0),
        ("Bottleneck 5 (layer3)", (9, 7, 2), 1.0),
        ("Bottleneck 6 (layer3)", (9, 8, 2), 1.0),
        ("layer4", (10, 2, 2), 1.5),
        ("Bottleneck 1 (layer4)", (10, 3, 2), 1.0),
        ("Bottleneck 2 (layer4)", (10, 4, 2), 1.0),
        ("Bottleneck 3 (layer4)", (10, 5, 2), 1.0),
        ("inner_block_module", (11, 2, 2), 1.2),
        ("layer_block_module", (12, 2, 2), 1.2),
        ("RegionProposalNetwork", (2, 2, 4), 2.0),
        ("AnchorGenerator", (3, 2, 4), 1.5),
        ("RPNHead", (4, 2, 4), 1.5),
        ("conv (RPNHead)", (4, 3, 4), 1.2),
        ("cls_logits", (4, 4, 4), 1.2),
        ("bbox_pred", (4, 5, 4), 1.2),
        ("RoIHeads", (2, 2, 6), 2.0),
        ("RoIAlign", (3, 2, 6), 1.5),
        ("FastRCNNPredictor", (4, 2, 6), 2.0),
        ("fc1", (4, 3, 6), 1.0),
        ("fc2", (4, 4, 6), 1.0),
        ("cls_score", (4, 5, 6), 1.0),
        ("bbox_pred", (4, 6, 6), 1.0),
        ("MaskRCNNPredictor", (4, 3, 6), 2.0),
        ("mask_fcn1", (5, 3, 6), 1.2),
        ("mask_fcn2", (6, 3, 6), 1.2),
        ("mask_fcn3", (7, 3, 6), 1.2),
        ("mask_fcn4", (8, 3, 6), 1.2),
        ("relu1", (9, 3, 6), 0.8),
        ("relu2", (10, 3, 6), 0.8),
        ("relu3", (11, 3, 6), 0.8),
        ("relu4", (12, 3, 6), 0.8),
        ("mask_conv5", (13, 3, 6), 1.2),
        ("relu5", (14, 3, 6), 0.8),
        ("mask_fcn_logits", (15, 3, 6), 1.2),
        ("Transformer", (2, 2, 8), 2.0),
        ("resize", (3, 2, 8), 1.5),
        ("normalize", (4, 2, 8), 1.5),
        ("first",(2,2,10),2.0),
        ("second", (16, 2, 2), 1.5),
        ("third", (2, 10, 2), 1.5),
    ]

    # 创建立方体并添加到Plotter对象中
    for layer_name, (x, y, z), size in layers:
        cube, color = create_cube(size, 'lightblue')
        cube.translate((x, y, z))
        plotter.add_mesh(cube, color=color, show_edges=True)
        plotter.add_point_labels([[x, y, z]], [layer_name], font_size=10, text_color='black', point_color=color)

    # 创建直角线条
    lines = [
        ((2, 2, 2), (2, 2, 9)),  # x=2 && y=2
        ((2, 2, 2), (2, 9, 2)),  # y=2 && z=2
        ((2, 2, 2), (16, 2, 2))  # x=2 && z=2
    ]

    for start, end in lines:
        line, color = create_line(start, end, 'red')
        plotter.add_mesh(line, color=color, line_width=3)

    # 设置视图和背景
    plotter.set_background('white')
    plotter.view_isometric()

    # 显示3D图像
    plotter.show()


if __name__ == "__main__":
    show_net()
