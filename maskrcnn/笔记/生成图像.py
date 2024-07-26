"""
安装指令
pip install pyvista
"""
import re
import pyvista as pv

#没写出来，未来有时间写
def parse_model_description(file_path):
    """解析模型描述文件"""
    layers = []
    level = 1

    def add_layer(name, size, level):
        """添加层到列表"""
        layers.append([len(layers) + 1, name, size, level])

    with open(file_path, 'r') as file:
        content = file.read()

    # 正则表达式来匹配括号及内容
    pattern = re.compile(r'\(([^)]+):\s*([^()]+)\(', re.DOTALL)
    matches = pattern.findall(content)

    stack = []

    for match in matches:
        layer_name, layer_type = match
        layer_name = layer_name.strip()
        layer_type = layer_type.strip()

        if layer_type.startswith('Conv') or layer_type.startswith('Frozen') or layer_type.startswith('ReLU') or layer_type.startswith('MaxPool') or layer_type.startswith('Linear') or layer_type.startswith('ConvTranspose'):
            # 提取尺寸
            size = re.search(r'\(([^)]+)\)', content[content.find(layer_type):])
            size = size.group(1) if size else '非最小层'
            add_layer(layer_type, size, len(stack) + 1)

        # 处理层次结构
        while stack and stack[-1][1] >= level:
            stack.pop()
        if layer_type not in ('Conv2d', 'FrozenBatchNorm2d', 'ReLU', 'MaxPool2d', 'Linear', 'ConvTranspose2d'):
            level += 1
        stack.append((layer_name, level))
    print(stack)

    return layers


def create_cube(size, color):
    """创建一个指定尺寸和颜色的立方体"""
    cube = pv.Cube(center=(0, 0, 0))
    # Scale cube to the size specified
    cube = cube.scale((size, size, size))
    return cube, color


def show_net(file_path):
    # 读取模型描述文件
    layers = parse_model_description(file_path)

    # 创建一个PyVista的Plotter对象
    plotter = pv.Plotter()

    # 初始化坐标
    x, y, z = 0, 0, 0
    spacing = 3  # 用于分隔层的空间

    # 颜色字典，根据层类型设置颜色
    color_map = {
        'Conv2d': 'blue',
        'FrozenBatchNorm2d': 'green',
        'ReLU': 'red',
        'MaxPool2d': 'yellow',
        'Linear': 'purple',
        'ConvTranspose2d': 'orange'
    }

    # 创建立方体并添加到Plotter对象中
    for layer_id, layer_name, layer_size, layer_level in layers:
        # 根据层的类型和尺寸创建立方体
        size = 1.0  # 可以根据层的实际尺寸调整
        color = color_map.get(layer_name, 'gray')
        cube, color = create_cube(size, color)
        plotter.add_mesh(cube, color=color, show_edges=True)
        plotter.add_point_labels([[x, y, z]], [f"{layer_id}: {layer_name}\n{layer_size}"], font_size=10, text_color='black',
                                 point_color=color)
        print(f"Layer ID: {layer_id}, Name: {layer_name}, Size: {layer_size}, Level: {layer_level}, Position: ({x}, {y}, {z})")
        # 更新坐标位置
        x += size + spacing

    # 设置视图和背景
    plotter.set_background('white')
    plotter.view_isometric()

    # 显示3D图像
    plotter.show()


if __name__ == "__main__":
    file_path = 'D:/视觉实验室/try/maskrcnn/笔记/maskrcnn-net.txt'
    show_net(file_path)
