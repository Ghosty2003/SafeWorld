# 固定低速→高速：函数组合检查

固定8个已训练函数、40种预先列出的组合；80fit/20复用internal。
所有基本函数在训练集P1/P2全过；正加权平均、逐点min/max/median不引入时间输入或改变detector。
没有在internal上拟合权重；从固定网格选组合仍属于调参，不能当独立验证。

| 组合 | train整路径 | internal整路径 | internal P1违反 | internal P2违反 | 状态 |
|---|---|---|---|---|---|
| original | 80/80 | 18/20 | 13 | 14 | BORDERLINE |
| blend1_0.25 | 80/80 | 14/20 | 16 | 18 | BORDERLINE |
| blend1_0.5 | 80/80 | 17/20 | 12 | 15 | BORDERLINE |
| blend1_0.75 | 80/80 | 19/20 | 12 | 13 | BORDERLINE |
| min1 | 80/80 | 14/20 | 19 | 19 | BORDERLINE |
| max1 | 80/80 | 18/20 | 13 | 14 | BORDERLINE |
| blend2_0.25 | 80/80 | 17/20 | 12 | 14 | BORDERLINE |
| blend2_0.5 | 80/80 | 18/20 | 11 | 13 | BORDERLINE |
| blend2_0.75 | 80/80 | 19/20 | 11 | 13 | BORDERLINE |
| min2 | 80/80 | 17/20 | 14 | 15 | BORDERLINE |
| max2 | 80/80 | 19/20 | 12 | 13 | BORDERLINE |
| blend3_0.25 | 80/80 | 16/20 | 13 | 14 | BORDERLINE |
| blend3_0.5 | 80/80 | 18/20 | 9 | 10 | BORDERLINE |
| blend3_0.75 | 80/80 | 19/20 | 11 | 12 | BORDERLINE |
| min3 | 80/80 | 15/20 | 15 | 16 | BORDERLINE |
| max3 | 80/80 | 19/20 | 12 | 13 | BORDERLINE |
| blend4_0.25 | 80/80 | 9/20 | 25 | 26 | BORDERLINE |
| blend4_0.5 | 80/80 | 12/20 | 22 | 22 | BORDERLINE |
| blend4_0.75 | 80/80 | 13/20 | 20 | 21 | BORDERLINE |
| min4 | 80/80 | 8/20 | 29 | 31 | BORDERLINE |
| max4 | 80/80 | 18/20 | 13 | 13 | BORDERLINE |
| blend5_0.25 | 80/80 | 15/20 | 15 | 17 | BORDERLINE |
| blend5_0.5 | 80/80 | 19/20 | 12 | 12 | BORDERLINE |
| blend5_0.75 | 80/80 | 19/20 | 10 | 12 | BORDERLINE |
| min5 | 80/80 | 15/20 | 18 | 19 | BORDERLINE |
| max5 | 80/80 | 18/20 | 13 | 14 | BORDERLINE |
| blend6_0.25 | 80/80 | 12/20 | 22 | 24 | BORDERLINE |
| blend6_0.5 | 80/80 | 14/20 | 20 | 20 | BORDERLINE |
| blend6_0.75 | 80/80 | 17/20 | 12 | 14 | BORDERLINE |
| min6 | 80/80 | 11/20 | 21 | 21 | BORDERLINE |
| max6 | 80/80 | 17/20 | 18 | 19 | BORDERLINE |
| blend7_0.25 | 80/80 | 16/20 | 13 | 17 | BORDERLINE |
| blend7_0.5 | 80/80 | 18/20 | 11 | 13 | BORDERLINE |
| blend7_0.75 | 80/80 | 18/20 | 11 | 12 | BORDERLINE |
| min7 | 80/80 | 16/20 | 14 | 16 | BORDERLINE |
| max7 | 80/80 | 17/20 | 15 | 16 | BORDERLINE |
| all_mean | 80/80 | 18/20 | 10 | 10 | BORDERLINE |
| all_median | 80/80 | 16/20 | 16 | 18 | BORDERLINE |
| all_min | 80/80 | 12/20 | 21 | 21 | BORDERLINE |
| all_max | 80/80 | 18/20 | 12 | 12 | BORDERLINE |

选中：{"id": "blend5_0.75", "op": "weighted", "members": [0, 5], "weights": [0.75, 0.25]}
对照整路径通过：{'constant': 0, 'mode_constant': 0, 'shuffle0': 1, 'random0': 3, 'shuffle1': 2, 'random1': 4, 'shuffle2': 1, 'random2': 8}
未启动正式校准或Test1。完整AND独立重算4000条候选路径。
