Descriptors

1 可及表面积

![image-20240401144837747](/Users/liz/Library/Application Support/typora-user-images/image-20240401144837747.png)

gmx sasa可完成计算。

这里用Chemicalize的MarvinSketch工具计算

1-甲基咪唑：255.79 Å2

1-乙基咪唑：274.53 Å2

1-丁基咪唑：338.78 Å2

硫氰酸甲酯：232.29 Å2

硫氰酸乙酯：267.49 Å2

硫氰酸丁酯：318.74 Å2

2 氢键寿命

![image-20240411145720771](/Users/liz/Library/Application Support/typora-user-images/image-20240411145720771.png)

// 不是，哥们，gromacs和mdanalysis都算不出啊

MDA程序写出来了，手动选择供体和受体

// 关于氢键的Donor和Acceptor，均最好通过量化的方法完成计算。

计算完成后，可以用hbond-dynamics.py进行计算动力学，也可以用HBDMDA2.py计算寿命。

Imidazole and 1‑Methylimidazole Hydrogen Bonding and Nonhydrogen Bonding Liquid Dynamics: Ultrafast IR Experiments中描述了烷基咪唑的氢键供体和受体。

不是，寄了啊？这篇文献说1-甲级咪唑根本没供体，只有受体性质，硫氰酸甲酯的N和S也是很强的受体，所以D-H-A结构根本没有D



3 优先排除系数

![image-20240412124212754](/Users/liz/Library/Application Support/typora-user-images/image-20240412124212754.png)

个人认为没有计算的价值。这里描述的是共溶剂的聚集效应，而我们的体系中毫无与类似相关的效应。

4  文献中找到的描述符

| **Mol.Wt (amu)** | **E.HOMO (eV)** | **E.LUMO (eV)** | **Dipole (debye)** | **CPK Area(Å)**    | **CPK Ovality** | **Polarizability** | **HBD Count** | **HBA Count** | **ZPE (kJ.mol)** |
| ---------------- | --------------- | --------------- | ------------------ | ------------------ | --------------- | ------------------ | ------------- | ------------- | ---------------- |
| 摩尔质量         | HOMO轨道能量    | LUMO轨道能量    | 电偶极矩           | CPK模型 原子表面积 | CPK椭圆度       | 极性               | 氢键提供者    | 氢键接收者    | 零点能           |

谢谢https://pubchem.ncbi.nlm.nih.gov/ 大爹

4.1 摩尔质量

1-甲基咪唑：82.10380

1-乙基咪唑：96.13042

1-丁基咪唑：124.184

硫氰酸甲酯：73.117

硫氰酸乙酯：87.143

硫氰酸丁酯：115.19

4.2-4.3 HOMO-LUMO轨道能量

使用Gaussian进行计算。

4.4 电偶极矩（Dipole Moment）

用Gaussian计算。

4.5-4.6 CPK面积 原文用

4.7 极化度

1-甲基咪唑：7.37 Å3

1-乙基咪唑：9.2 Å3

1-丁基咪唑：12.88 Å3

硫氰酸甲酯：8.9 Å3

硫氰酸乙酯：10.74 Å3

硫氰酸丁酯：14.42 Å3

4.8 零点能

Gaussian结果

5 电荷离域度（charge delocalization）
