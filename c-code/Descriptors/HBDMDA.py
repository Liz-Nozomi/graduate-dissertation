import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as hba
import matplotlib.pyplot as plt

# 读取GROMACS轨迹文件和拓扑文件
gro="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.gro"
xtc="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.xtc"

# 初始化Universe对象
universe = mda.Universe(gro, xtc)

hbonds=hba(universe=universe)
hbonds.run(verbose=True)
