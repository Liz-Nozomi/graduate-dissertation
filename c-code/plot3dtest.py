import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 输入散点和五个通道的颜色信息
points = np.array([(1, 2, 3, 0.5, 0.2, 0.8),
                   (3, 4, 5, 0.2, 0.7, 0.3),
                   (6, 8, 2, 0.9, 0.1, 0.5),
                   # 添加更多散点...
                   ])

# 提取坐标和颜色信息
def paint3d(points):#一个六维的xyzrgb数组
    x, y, z, r, g, b = np.hsplit(points, 6)

# 创建3D图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
    scatter = ax.scatter(x.flatten(), y.flatten(), z.flatten(), c=np.column_stack((r, g, b)), marker='o')

# 设置坐标轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

# 添加颜色条
    fig.colorbar(scatter)

# 显示图像
    plt.show()
    return plt
