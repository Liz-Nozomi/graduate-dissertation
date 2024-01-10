关于python程序：/Volumes/Liz2T/SolventNetData/2020_SolventNet_Chem_Sci/python_scripts/generate_grid_interpolation.py

第一个函数Normalize_3d_rdf

**输入：**

- `num_dist`：[np.array, shape=(time, x, y, z)] 数值分布随时间变化的数据。
- `bin_volume`：[float] 体积元的体积。
- `total`：[float] 在您的盒子中可能的总数，例如总溶剂原子数。
- `volume`：[np.array, shape=(time,1)] 每帧的体积。

**输出：**

- ```
  normalized_num_dist
  ```

  ：[np.array, shape=(time, x, y, z)] 规范化的数值分布。规范化基于以下计算：

  - （数值分布 / 体积元体积）/（总可能数 / 体积） **注意：** 需要交换轴以正确地将数值分布除以体积。