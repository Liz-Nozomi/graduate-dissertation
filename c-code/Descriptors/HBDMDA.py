import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as HBA
import numpy as np

# 读取GROMACS轨迹文件和拓扑文件
# gro="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.gro"
tpr="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.tpr"
xtc="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.xtc"
#%% 
# 初始化Universe对象
u = mda.Universe(tpr, xtc)
# 选择氢键供体和受体
bmi_selection = {
    'donors': 'resname BMI',
    'hydrogens': 'resname BMI and name H*',
    'acceptors': 'resname BMI and name N*'
}
bus_selection = {
    'donors': 'resname BUS',
    'hydrogens': 'resname BUS and name H*',
    'acceptors': 'resname BUS and name N*'
}

def analyze_hydrogen_bond_lifetime(universe, selection):
    hba = HBA(universe=universe, update_selections=True)
    hba.donors_sel = selection['donors']
    hba.hydrogens_sel = selection['hydrogens']
    hba.acceptors_sel = selection['acceptors']
    hba.run()
    return hba.hbonds

bmi_hbonds = analyze_hydrogen_bond_lifetime(u, bmi_selection)
bus_hbonds = analyze_hydrogen_bond_lifetime(u, bus_selection)
print(bmi_hbonds[1,:])
#%% 
def calculate_lifetime(hbonds):
    lifetimes = {}
    for i, hbond in enumerate(hbonds):
        donor, hydrogen, acceptor, _, _ ,_= hbond
        # 这里使用一个简单的计数器来跟踪氢键的持续时间
        if (donor, hydrogen, acceptor) not in lifetimes:
            lifetimes[(donor, hydrogen, acceptor)] = 1
        else:
            lifetimes[(donor, hydrogen, acceptor)] += 1
    return lifetimes


bmi_lifetimes = calculate_lifetime(bmi_hbonds)
bus_lifetimes = calculate_lifetime(bus_hbonds)
# 打印BMI组分的氢键寿命
print("BMI Hydrogen Bond Lifetimes:")
for hbond, lifetime in bmi_lifetimes.items():
    print(f"Hydrogen Bond {hbond}: {lifetime} frames")

# 打印BUS组分的氢键寿命
print("\nBUS Hydrogen Bond Lifetimes:")
for hbond, lifetime in bus_lifetimes.items():
    print(f"Hydrogen Bond {hbond}: {lifetime} frames")