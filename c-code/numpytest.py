import numpy as np

# 创建一个示例数组
arr = np.array([[1, 2, 3, 4, 5, 6],
                [1, 2, 3, 7, 8, 9],
                [1, 2, 3, 10, 11, 12],
                [4, 5, 6, 13, 14, 15],
                [4, 5, 6, 16, 17, 18],
                [1,3,9,0,0,0]])

# 将前三列相等的行合并
unique_rows, indices = np.unique(arr[:, :3], axis=0, return_inverse=True)
summed_rows = np.zeros((unique_rows.shape[0], 3 + 3))
for i in range(unique_rows.shape[0]):
    summed_rows[i, :3] = unique_rows[i]
    summed_rows[i, 3:] = np.sum(arr[indices == i, 3:], axis=0)

# 输出结果
print("原始数组：\n", arr)
print("处理后的数组：\n", summed_rows)