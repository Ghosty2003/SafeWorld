# SafeDreamer L2 specification screening 审计结果

结论：ABSTAIN / no-qualified-spec。这不是规范为假、模型必然不安全或有效V不存在的证明。

本轮固定已审计checkpoint和CCEPlanner.policy()，H=300。80条既有fit路径，仅使用其train_fit分区；
另采20条新internal-validation，seed从141000000起，与以往internal-check及正式cal/test分离。
18个权重均在新internal-validation采样前冻结。候选/阈值/架构/训练规则事先写入plan.json。
这20条用于本轮选择，之后不能再当最终校准或独立测试。未读取旧calibration/test样本或结果。

## 固定定义

LOW阈值=0.4787735829，HIGH阈值=0.8225157078，来自80条fit路径的clearance状态分布25%/75%分位数。
clearance=min(norm(decoded hazard vectors))，无hazard定义为clearance>=0.2，goal半径为严格<0.3。
q_t消费到t的观测；初始标签在t=0消费。P2使用源状态q_t。每条301个状态恰好300条真实转移，无人工末尾自环。
eta=0.01。所有评价用float64、P1/P2容差均为0。P1检查全部转移，P2检查q_t∈B。
V输入为完整RSSM、planner carry、当前解码clearance/goal距离和因果自动机mode；没有剩余时间、路径ID或未来信息。
训练辅助标签使用有限图rank；未完成路径仅施加下界，不把H截断当作成功。
Z_free={V<eta}；bad-mode输出结构性>=eta。Guarantee接受态V=0且自动机吸收；这不证明物理安全区域不变。
Persistence的good状态不是吸收态，有限安全后缀不等于已经永久完成。

## 候选与结果

| Spec | MP class | B / accepting | train P1违反 | train P2违反 | internal P1违反 | internal P2违反 | 整路径P1P2 train/internal | preview完整事件 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| F(clearance >= high) | Guarantee | [0] / 1 | 2163/24000 | 4576/5552 | 545/6000 | 1126/1350 | 40/80；10/20 | 10/20 | BORDERLINE |
| F(LOW & F HIGH) | Guarantee | [0, 1] / 2 | 3511/24000 | 7553/8954 | 1069/6000 | 2206/2544 | 11/80；4/20 | 4/20 | BORDERLINE |
| F(goal_distance < 0.3) | Guarantee | [0] / 1 | 3428/24000 | 7000/9163 | 619/6000 | 1309/1833 | 19/80；6/20 | 6/20 | BORDERLINE |
| F(LOW & F G[0,7](clearance >= high)) | Guarantee | [0, 1, 2, 3, 4, 5, 6, 7, 8] / 9 | 6218/24000 | 11031/14537 | 1712/6000 | 3117/3983 | 3/80；0/20 | 0/20 | BORDERLINE |
| F G(clearance >= 0.2) | Persistence / co-Buchi | [0] / 1 | 11306/24000 | 0/36 | 2862/6000 | 0/14 | 0/80；0/20 | 0/20 | BORDERLINE |
| F G(clearance >= high) | Persistence / co-Buchi | [0] / 1 | 11192/24000 | 13640/18007 | 2818/6000 | 3584/4718 | 0/80；0/20 | 0/20 | BORDERLINE |

以上每个spec选一个最佳候选；全部18个函数及其常数/按mode打乱/随机函数对照见report.json。
P1/P2比例分母不同；不能用大量接受态零差值掩盖pending状态的P2违反。

## 有限图可行性的限制

以完整product state的精确字节身份建图，正权边表示要求下降eta。SCC内有正权边即不可行。
没有正权环时用凝聚DAG最长加权路径构造非负可行见证。没有合并近似latent，也没有虚构末尾转移。
本轮图由有限路径构成；可行仅说明采到的约束有解，不说明某个神经函数可泛化，亦不说明无限支持集可证。

## 完整certificate AND和非退化检查

preview = 全路径P1 AND P2 AND endpoint AND Zfree/bad-exclusion AND sampled closure AND region AND finite/nonnegative。
region在fit的标准化完整物理状态半径上固定；不利用validation扩大区域。
真正常数V、仅按mode的常数、5次mode内打乱V、5个随机网络均检查；不以仅测全局常数替代其余对照。
Persistence另缺无限closure依据，不把sampled predicate持续为真写成无限warrant。

## 模型选择与停止

每个spec尝试线性softplus、128×128 ELU、256×256 ELU，各1200更新；train每200更新选checkpoint。
在新internal-validation上优先整路径P1/P2通过、再平衡P1/P2单步通过率；完全没有使用CP下界选模型。
未达到要求时只记BORDERLINE；不能发SAFE。也不能由P1/P2失败推出spec violation。
正式calibration和Test1均未运行，p_hat_gamma不填0而标未计算。
Persistence候选既没有有限完成证明，也没有无限closure证明；与有限终止Guarantee区别见plan.json。

## 复核

1800条候选路径和4320条对照路径独立重算完全一致；数据、checkpoint、代码、权重哈希已复核。
审计文件：audit.json；逐模型细节：report.json；对比表：screening_summary.csv；总表：final_table.md。

## 起点已完成与最接近候选

| Spec | train 起点已完成 | internal 起点已完成 | internal 单纯完成 | internal 全路径P1/P2 |
|---|---|---|---|---|
| high | 15 | 5 | 18 | 10/20 |
| return | 0 | 0 | 15 | 4/20 |
| goal | 0 | 0 | 17 | 6/20 |
| recovery8 | 0 | 0 | 11 | 0/20 |
| persist_safe | None | None | None | 0/20 |
| persist_high | None | None | None | 0/20 |

按预先规定的全路径P1/P2优先规则，最接近BORDERLINE：F(clearance >= high)，width=256。
train整路径通过40/80；internal通过10/20。internal P1违反545/6000，P2违反1126/1350。
这只是本轮开发候选排序，不是KEEP，也不是统计置信声明。
所有候选已经在用户要求的数值100%条件处失败；Persistence的额外语义保留条件没有改变本轮结论。
