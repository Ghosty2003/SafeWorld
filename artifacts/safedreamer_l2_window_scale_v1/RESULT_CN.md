# L2 窗口候选的尺度检查

规范：F[0,277] G[0,23](clearance>=0.2)，H=300，eta=0.01。
当前输入为完整latent/planner、解码信号、因果safe-run计数，不含未来时间。
基函数来自80条fit训练；20条旧internal用于选择家族和输出尺度，未用于梯度训练。
不会改变旧48-state SAFE结果，也未使用其校准/Test1。

| V输出倍数 | train P1/P2整路径 | internal P1 | internal P2 | internal整路径 | 状态 |
|---|---|---|---|---|---|
| 1.0 | 80/80 | 6000/6000 | 474/476 | 18/20 | BORDERLINE |
| 1.5 | 80/80 | 6000/6000 | 475/476 | 19/20 | BORDERLINE |
| 2.0 | 80/80 | 6000/6000 | 476/476 | 20/20 | KEEP_DEV |

开发集选中输出倍数2.0。没有放松eta或容差，而是整体放大同一个非恒定V的下降幅度。
正尺度不能修复V增加的转移；本候选原本已P1全过，两个失败仅是下降不足eta。
internal值范围：[0, 1.52774]。
常数/按mode常数/打乱/随机对照的整路径通过数：{'constant': 0, 'mode_constant': 0, 'monitor_rank': 15, 'shuffle0': 3, 'random0': 0, 'shuffle1': 3, 'random1': 0, 'shuffle2': 2, 'random2': 7}
20/20完成规范、无初始已完成；完整AND与逐路径独立实现精确一致。
**KEEP_DEV不是SAFE：同一内部集经过反复选择，需要新独立确认；没有正式CP下界。**
