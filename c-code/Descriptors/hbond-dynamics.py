import MDAnalysis
import MDAnalysis.analysis.align
import sys
import re
import itertools
import os
from MDAnalysis.tests.datafiles import GRO, XTC
import numpy as np
import json
import scipy.integrate as integrate

# 定义分子动力学模拟文件的路径
GRO="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.gro"
XTC="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.xtc"
# 使用MDAnalysis库加载模拟数据
u = MDAnalysis.Universe(GRO, XTC)

# 定义一个函数来计算氢键的数量
def HBonds_MIM(Ha_atom,Oa_atom,Nd_atom):   
    # 定义氢键形成原子的数量
    n_Ha = 1
    n_Nd = len(Nd_atom)
    # 初始化氢键计数
    T_HB = 0                    
    # 遍历所有的氢键形成原子对
    for j in range (n_Ha):       
        for i in range (n_Nd):           
            # 计算原子间的距离
            HN_vector= Ha_atom[j] - Nd_atom[i]  
            HN_distance= np.linalg.norm(HN_vector) 
            ON_vector=Oa_atom[j] - Nd_atom[i]
            ON_distance= np.linalg.norm(ON_vector)  
  
            # 如果原子间的距离在氢键形成范围内
            if HN_distance <= 2.5 and ON_distance <=3.6:
                # 计算原子间的角度
                theta_HNO = np.arccos(np.dot(HN_vector,ON_vector)/(np.linalg.norm(HN_vector)*np.linalg.norm(ON_vector)))
                angle_HNO=np.rad2deg(theta_HNO)                            
                # 如果角度在氢键形成范围内
                if angle_HNO <=30:  
                    # 增加氢键计数
                    T_HB=T_HB + 1                                      
    return T_HB

# 设置中心点坐标
x_center=25
y_center=25
# 计算距离切片的数量
n=int(16/0.1)      # 将半径分为160个切片来计算每个切片中的氢键数量
HBs_density_rs=[]

# 遍历模拟轨迹中的每一帧
for ts in u.trajectory[10000:100000]: 

    # 选择MCM41中的氢和氧原子
    mcm_H=u.select_atoms('resname MCM and name Ho')
    mcm_O=u.select_atoms('resname MCM and name Oh')

    # 选择MIM中的碳和氢原子
    mim_CR=u.select_atoms('resname MIM and name CR5')
    mim_HR=u.select_atoms('resname MIM and name HR9')

    # 选择MIM中的碳和氢原子
    mim_CW1=u.select_atoms('resname MIM and name CW2')  
    mim_HW1=u.select_atoms('resname MIM and name HW7')
    mim_CW2=u.select_atoms('resname MIM and name CW3')
    mim_HW2=u.select_atoms('resname MIM and name HW8')

    # 选择MIM中的氮原子
    mim_N2=u.select_atoms('resname MIM and name NA4')
   
    # 计算MIM的质心位置
    mim=u.select_atoms('resname MIM')
    mim_center=mim.center(mim.masses,compound='residues')

    # 初始化氢键计数数组
    T_HB_CR_HN=np.zeros(n)
    T_HB_CW1_HN=np.zeros(n)
    T_HB_CW2_HN=np.zeros(n)
    T_HB_OH_HN=np.zeros(n)
    T_HBs_Total=np.zeros(n)
    # 遍历MIM的每个原子
    for i in range (len(mim_center)):
        # 计算MIM质心到中心点的距离
        r_distance=np.sqrt(np.square(mim_center[i][0]-x_center)+np.square(mim_center[i][1]-y_center))        
        r_distance=round(r_distance/0.1)    
        # 根据距离计算氢键数量
        for j in range (n):
             if r_distance == j:
                 # 计算CR-HR-N之间的氢键
                 T_HB_CR_HN[j] =HBonds_MIM(mim_HR.positions[i],mim_CR.positions[i],mim_N2.positions)
                 # 计算CW1-HW1-N之间的氢键
                 T_HB_CW1_HN[j] =HBonds_MIM(mim_HW1.positions[i],mim_CW1.positions[i],mim_N2.positions)

                 # 计算CW2-HW2-N之间的氢键
                 T_HB_CW2_HN[j] =HBonds_MIM(mim_HW2.positions[i],mim_CW2.positions[i],mim_N2.positions)

                 # 计算MCM41的OH与MeSCN的OH-HO-N之间的氢键
                 T_HB_OH_HN[j] =HBonds_MIM(mcm_H.positions[i],mcm_O.positions[i],mim_N2.positions)

    # 计算每个切片的总氢键数量
    for m in range(n):
        T_HBs_Total[m]=T_HB_CR_HN[m] + T_HB_CW1_HN[m] + T_HB_CW2_HN[m] + T_HB_OH_HN[m]   

    # 计算每个切片的氢键密度
    HBs_density_r=[]
    for m in range (n):
        HBs_density=T_HBs_Total[m]*10000 / ((np.pi*440*(m+1)**2-np.pi*440*m**2))
        HBs_density_r.append(HBs_density)
    HBs_density_rs.append(HBs_density_r)

# 计算氢键密度的平均值
HBs_density_rs=np.mean(HBs_density_rs,axis=0)

# 将结果保存到文件
with open('mcm2nm-hbonds-density-r-mim.txt','w') as f:
    for item in range(len(HBs_density_rs)):
        f.write(str(item))
        f.write("  ")
        f.write(str(HBs_density_rs[item]))
        f.write("\n")