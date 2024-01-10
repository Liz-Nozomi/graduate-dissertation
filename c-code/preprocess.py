# this script is used to transfer md simulation results into 3d plots, with n channels(n=types of compounds)
import MDAnalysis as mda
import path_config
current_env="md_dev"
env_info=path_config.md_paths.get(current_env,None)
if env_info:
    gro_path=env_info.get('gro','')
    
if gro_path:
    gro_file=gro_path+"/mixed_solv_prod.gro"
    try:
            # 打开文件并读取内容
        with open(gro_file, 'r') as file:
            content = file.read()
            print(content)  # 输出文件内容（这里仅作示例）
    except FileNotFoundError:
            print("gro文件未找到！")
            
# 以上代码用于读取gro文件。

u=mda.Universe(gro_file)
print(u)
print(len(u.trajectory))