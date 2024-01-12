import MDAnalysis as mda
import path_config
import numpy as np
import plot3dtest

current_env="md_dev"
env_info=path_config.md_paths.get(current_env,None)
if env_info:
    gro_path=env_info.get('gro','')
    xtc_path=env_info.get('xtc','')
    
if gro_path:
    gro_file=gro_path+"/mixed_solv_prod.gro"
    try:
            # 打开文件并读取内容
        with open(gro_file, 'r') as file:
            content = file.read()
            print(content)  # 输出文件内容（这里仅作示例）
    except FileNotFoundError:
            print("gro文件未找到！")
xtc_file=xtc_path+"/mixed_solv_prod_first_20_ns_centered_with_10ns.xtc"
#xtc异常处理不想写了 跳

XTC = xtc_file
GRO = gro_file
u = mda.Universe(GRO, XTC)

CEL = np.trunc(u.select_atoms("resname CEL").positions)
SOL = np.trunc(u.select_atoms("resname SOL").positions)
DIO = np.trunc(u.select_atoms("resname DIO").positions)
zeros_array = np.zeros((CEL.shape[0], 3), dtype=CEL.dtype)
CEL=np.hstack((CEL,zeros_array))
CEL[:, 3:] = np.array([[1, 0, 0]])
zeros_array = np.zeros((SOL.shape[0], 3), dtype=SOL.dtype)
SOL=np.hstack((SOL,zeros_array))
SOL[:, 3:] = np.array([[0, 1, 0]])
zeros_array = np.zeros((DIO.shape[0], 3), dtype=DIO.dtype)
DIO=np.hstack((DIO,zeros_array))
DIO[:, 3:] = np.array([[0, 0, 1]])

param=np.r_[CEL,SOL,DIO]
print(param)



# numpy 数组，用来存储所有的该分子（残基）的坐标信息
# 这里是低精度的像素点信息

unique_rows, indices = np.unique(param[:, :3], axis=0, return_inverse=True)
summed_rows = np.zeros((unique_rows.shape[0], 3 + 3))
for i in range(unique_rows.shape[0]):
    summed_rows[i, :3] = unique_rows[i]
    summed_rows[i, 3:] = np.sum(param[indices == i, 3:], axis=0)



column4=summed_rows[:,3]
column5=summed_rows[:,4]
column6=summed_rows[:,5]
max_4=np.max(column4)
max_5=np.max(column5)
max_6=np.max(column6)

normalization=max(max_4,max_5,max_6)

divide=np.divide(summed_rows[:,3:6],normalization)
summed_rows[:,3:6]=divide
np.savetxt("param.txt",summed_rows)
print(summed_rows)

plot3dtest.paint3d(summed_rows)
