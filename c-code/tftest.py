import numpy as np



# 假设点云数据
point_cloud_data = np.random.rand(125000, 3 + 3)

# 转换为3D张量
width, height, depth = 50, 50, 50  # 根据数据分布和模型需求进行调整
channels = 6  # 3D坐标(x, y, z) + 3个特征
input_data = point_cloud_data.reshape((width, height, depth, channels))
print(input_data.shape)