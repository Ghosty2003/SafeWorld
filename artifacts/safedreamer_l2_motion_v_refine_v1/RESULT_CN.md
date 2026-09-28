# 固定低速→高速规范：V优化结果

固定detector：先speed<=0.2245011670，随后speed>=0.7390221069；H=300、eta=0.01、零容差。
同一SafeDreamer checkpoint + CCEPlanner.policy；仅80fit+20复用internal，未使用正式calibration/Test1。
V训练加入fit-only长等待加权、pending状态归一化、完成剩余步数辅助标签；未来标签不作为模型输入。
8个函数、每个400步辅助预训练+4000步优化；原18/20候选保留。

| 候选 | 最佳输出倍数 | train整路径P1/P2 | internal整路径P1/P2 | internal P1违反 | internal P2违反 | 状态 |
|---|---|---|---|---|---|---|
| full128 | 4.0 | 80/80 | 14/20 | 19/6000 | 19/148 | BORDERLINE |
| full256 | 4.0 | 80/80 | 17/20 | 14/6000 | 15/148 | BORDERLINE |
| deter128 | 4.0 | 80/80 | 15/20 | 16/6000 | 17/148 | BORDERLINE |
| kinematic64 | 4.0 | 20/80 | 7/20 | 32/6000 | 33/148 | BORDERLINE |
| full_max4 | 4.0 | 80/80 | 7/20 | 30/6000 | 30/148 | BORDERLINE |
| full_residual | 4.0 | 80/80 | 15/20 | 18/6000 | 19/148 | BORDERLINE |
| full128_rank | 2.0 | 80/80 | 10/20 | 28/6000 | 28/148 | BORDERLINE |
| deter128_rank | 4.0 | 80/80 | 15/20 | 16/6000 | 18/148 | BORDERLINE |
| original_motion_burst_elu128 | 4.0 | 80/80 | 18/20 | 13/6000 | 14/148 | BORDERLINE |

选中：original_motion_burst_elu128，输出倍数4.0，BORDERLINE。
选中函数：train P1 24000/24000，P2 750/750；internal P1 5987/6000，P2 134/148。
internal完整AND通过18/20，任务完成20/20；区域失败0、closure失败0、坏模式进入Zfree状态0。

**即使达到KEEP_DEV，也只是反复用于调参的内部数据全部通过，不是新独立验证，更不是SAFE。**
没有降低eta、放宽容差、改阈值、过滤失败路径或重设计时预算。接受态V=0来自因果完成状态；未完成态V>=eta。
独立重算2400条候选路径与全部对照；权重/数据hash、完整AND和first-hit monitor检查通过。
