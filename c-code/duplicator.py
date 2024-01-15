import numpy as np
from scipy import spatial
import math
import MDAnalysistest
# 定义初始点位置

angle=45
# 定义旋转角度（单位为弧度）
anglerad = angle*math.pi/180

axis = [np.random.rand(), np.random.rand(), np.random.rand()]  # 选取Z轴作为旋转轴
axisnorm=axis/np.linalg.norm(axis)

point=MDAnalysistest.readnp(0,1,1)
quar=[math.cos(anglerad),axisnorm[0],axisnorm[1],axisnorm[2]]
print("point here",point)
print(quar)

