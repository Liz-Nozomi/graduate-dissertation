import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as HBA

# 加载体系的MD轨迹文件
tpr="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.tpr"
xtc="/Volumes/exfat/RateNet/dataset/BIm-BuSCN-320/200-10bimbuscn-mdnvt-6ns-320-ml.xtc"
universe = mda.Universe(tpr, xtc)

# 定义BMI和BUS组分的氢键选择字符串
# 这些选择字符串基于您提供的GRO文件中的原子信息
sys_selection = {
    'donors_sel': "(resname BMI and (name N*))",
    'hydrogens_sel': "(resname BMI and name H*) or (resname BUS and name H*)",
    'acceptors_sel': "(resname BMI and (name N*)) or (resname BUS and (name N* or name S*))"
}

# 创建HydrogenBondAnalysis对象并设置参数
hba = HBA(universe=universe)
#donors_guessed = hba.guess_donors(select='all',max_charge=-0.5)


hba.donors_sel = sys_selection['donors_sel']
hba.hydrogens_sel=sys_selection['hydrogens_sel']
hba.acceptors_sel=sys_selection['acceptors_sel']

hba.d_h_cutoff=1.2 
hba.d_a_cutoff=3.0 
hba.d_h_a_angle_cutoff=150
hba.update_selections=True

# 运行氢键分析
hba.run()

# 获取氢键数据
hbond_data = hba.results.hbonds

# 计算氢键寿命
tau_timeseries, timeseries = hba.lifetime()

# 打印氢键寿命信息
print("Hydrogen Bond Lifetimes:")
for tau, count in zip(tau_timeseries, timeseries):
    print(f"Lifetime: {tau} frames, Count: {count}")