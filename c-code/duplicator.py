import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import MDAnalysistest
import plot3dtest
# 定义初始点位置

angle=90
# 定义旋转角度（单位为弧度）
anglerad = angle*math.pi/180

axis = [np.random.rand(), np.random.rand(), np.random.rand()]  # 选取Z轴作为旋转轴

axisnorm=axis/np.linalg.norm(axis)
print(axisnorm)
point=MDAnalysistest.readnp(0,5,1)

rbase=point[:,0:3]

quar=[math.cos(anglerad/2),axisnorm[0],axisnorm[1],axisnorm[2]]
print(quar)
r=R.from_quat(quar)
rout=r.apply(rbase)
rfinal=np.concatenate((rout,point[:,3:6]),axis=1)


unique_rows, indices = np.unique(rfinal[:, :3], axis=0, return_inverse=True)
summed_rows = np.zeros((unique_rows.shape[0], 3 + 3))
for i in range(unique_rows.shape[0]):
    summed_rows[i, :3] = unique_rows[i]
    summed_rows[i, 3:] = np.sum(rfinal[indices == i, 3:], axis=0)
    
    

positive=min(np.min(summed_rows[:,0]),np.min(summed_rows[:,1]),np.min(summed_rows[:,2]))
add=np.add(summed_rows[:,0:3],-positive)
summed_rows[:,0:3]=add

normalization=max(np.max(summed_rows[:,3]),np.max(summed_rows[:,4]),np.max(summed_rows[:,5]))
divide=np.divide(summed_rows[:,3:6],normalization)
summed_rows[:,3:6]=divide

plot3dtest.paint3d(summed_rows)

