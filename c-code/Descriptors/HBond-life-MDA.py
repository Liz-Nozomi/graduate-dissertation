import MDAnalysis as mda
from MDAnalysis.analysis import hbonds

# 读取坐标文件和拓扑文件

gro="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.gro"
xtc="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.xtc"

u = mda.Universe(gro,xtc)

# 定义氢键分析器
hbonds_analysis = hbonds(u, 'resname BMI', 'resname BUS', distance=3.5, angle=120.0)

# 执行氢键分析
hbonds_analysis.run()

# 获取氢键结果
hbonds_results = hbonds_analysis.count_by_time()

# 打印氢键结果
print(hbonds_results)

