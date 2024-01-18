import csv
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import griddata 
import MDAnalysistest

csv_file='../f-file/datamark.csv'
def findsol(csv_file,currentrow):
    # 打开 CSV 文件
    with open(csv_file, 'r') as file:
        # 创建 CSV 读取器
        csv_reader = csv.reader(file)
        
        for row_num, row in enumerate(csv_reader, 1):
            if row_num == currentrow:
                return [row[0],row[1],row[2],row[3]]
    
print(findsol(csv_file,4))

#把数据点变成高密度的
def dense(start,end,step):
    point=MDAnalysistest.readnp(start,end,step)

    xyz_coordinates=point[:,0:3]
    rgb_channels = point[:,3:6]
    dense_grid = np.mgrid[0:20, 0:20, 0:20].reshape(3, -1).T
    print(np.shape(dense_grid))
    # 使用griddata进行插值，得到稠密的rgb数据
    dense_rgb = griddata(xyz_coordinates, rgb_channels, dense_grid, method='linear', fill_value=0.0)

    # 将插值后的数据调整为(20, 20, 20, 3)形状
    dense_rgb = dense_rgb.reshape(20, 20, 20, 3)
    return dense_rgb




