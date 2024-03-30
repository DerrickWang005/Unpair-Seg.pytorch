import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_boxes(tensor, figsize=(10, 10), image_size=(1024, 1024)):
    """
    可视化边界框

    参数:
    - tensor: 形状为 [1, 400, 4] 的 numpy 数组，表示边界框坐标
    - figsize: 图像的大小，以英寸为单位
    - image_size: 背景图像的尺寸，用于规范化边界框坐标
    """
    tensor = tensor.cpu()
    fig, ax = plt.subplots(1, figsize=figsize)
    # 设置坐标轴的范围
    ax.set_xlim(0, image_size[0])
    ax.set_ylim(image_size[1], 0)

    # 遍历 tensor 中的每个边界框并绘制
    for box in tensor[0]:
        # 假设坐标是相对于 image_size 的图像的
        x_min, y_min, x_max, y_max = box# * np.array(image_size*2)
        # 创建一个矩形并添加到轴上
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig("debug.png")
