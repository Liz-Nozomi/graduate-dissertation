import MDAnalysis as mda
import path_config
import numpy as np
import plot3dtest
import csv

csv_file='../f-file/datamark.csv'
def findsol(csv_file,currentrow):
    # 打开 CSV 文件
    with open(csv_file, 'r') as file:
        # 创建 CSV 读取器
        csv_reader = csv.reader(file)
        
        for row_num, row in enumerate(csv_reader, 1):
            if row_num == currentrow:
                return [row[0],row[1],row[2],row[3]]
sol=findsol(csv_file,6)
path="/Volumes/Liz2T/RateNet/dataset/"+sol[0]+"-"+sol[1]+'-'+sol[2]

   
'''
current_env="md_dev"
env_info=path_config.md_paths.get(current_env,None)
if env_info:
    gro_path=env_info.get('gro','')
    xtc_path=env_info.get('xtc','')
'''

gro_file=path+"/200-10"+sol[0].lower()+sol[1].lower()+"-mdnvt-6ns-"+sol[2]+"-ml.gro"

xtc_file=path+"/200-10"+sol[0].lower()+sol[1].lower()+"-mdnvt-6ns-"+sol[2]+"-ml.xtc"

#xtc异常处理不想写了 跳

XTC = xtc_file
GRO = gro_file
u = mda.Universe(GRO, XTC)


def getparam(u):
    

    MIM = np.trunc(u.select_atoms("resname MIM").positions)
    zeros_array = np.zeros((MIM.shape[0], 3), dtype=MIM.dtype)
    MIM=np.hstack((MIM,zeros_array))
    MIM[:, 3:] = np.array([[1, 0, 0]])

    MES = np.trunc(u.select_atoms("resname MES").positions)
    zeros_array = np.zeros((MES.shape[0], 3), dtype=MES.dtype)
    MES=np.hstack((MES,zeros_array))
    MES[:, 3:] = np.array([[0, 1, 0]])

    EMP = np.trunc(u.select_atoms("resname EMP").positions)
    zeros_array = np.zeros((EMP.shape[0], 3), dtype=EMP.dtype)
    EMP=np.hstack((EMP,zeros_array))
    EMP[:, 3:] = np.array([[0, 0, 0]])

    param=np.r_[MIM,MES,EMP]
    return param
#param是一个六位数组，包含了所有的点xyz和相对应的RGB，这里直接屏蔽了第三通道因为一共俩分子


# numpy 数组，用来存储所有的该分子（残基）的坐标信息
# 这里是低精度的像素点信息
def readnp(start,end,delta):
    sum=np.array([[0,0,0,0,0,0]])
    for ts in u.trajectory[start:end:delta]:#这里得改，但是不知道怎么取帧来平均，都得试试
        sum=np.concatenate([getparam(u),sum],axis=0) 
    param=sum


    # 归一化，查重处理矩阵，这些玩意不能动
    unique_rows, indices = np.unique(param[:, :3], axis=0, return_inverse=True)
    summed_rows = np.zeros((unique_rows.shape[0], 3 + 3))
    for i in range(unique_rows.shape[0]):
        summed_rows[i, :3] = unique_rows[i]
        summed_rows[i, 3:] = np.sum(param[indices == i, 3:], axis=0)
    
     # 到这里都别动 我也看不懂了现在
    column4=summed_rows[:,3]
    column5=summed_rows[:,4]
    column6=summed_rows[:,5]
    max_4=np.max(column4)
    max_5=np.max(column5)
    max_6=np.max(column6)
    normalization=max(max_4,max_5,max_6)
    divide=np.divide(summed_rows[:,3:6],normalization)
    summed_rows[:,3:6]=divide
   

    np.savetxt(path+"/"+str(start)+"-"+str(end)+"-"+sol[0]+sol[1]+sol[2]+".txt",summed_rows)
    #plot3dtest.paint3d(summed_rows)
    return summed_rows
    

for i in range(1,18000,3000):
    readnp(i,i+2999,1)
