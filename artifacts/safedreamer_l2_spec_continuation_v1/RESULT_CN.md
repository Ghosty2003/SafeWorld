# L2 继续筛选：fit/internal-only

固定当前checkpoint、CCEPlanner、H=300、eta=0.01；80条fit、20条复用内部调参路径。
8个规范×2类函数（线性softplus、128×128 ELU），各2000次更新；只用fit挑训练checkpoint。
未读取正式校准或Test1，未改变已保存的48-state SAFE结果。本轮内部集已反复用于选择，不是新独立确认集。

| spec | 原轮内部整路径P1/P2 | 本轮train | 本轮internal | internal P1违反 | internal P2违反 | 完整事件preview | 状态 |
|---|---|---|---|---|---|---|---|
| F(clearance >= high) | 10/20 | 38/80 | 10/20 | 530/6000 | 1112/1350 | 10/20 | BORDERLINE |
| F(LOW & F HIGH) | 4/20 | 10/80 | 5/20 | 1029/6000 | 2163/2544 | 5/20 | BORDERLINE |
| F(goal_distance < 0.3) | 6/20 | 15/80 | 6/20 | 644/6000 | 1388/1833 | 6/20 | BORDERLINE |
| F(LOW & F G[0,7](clearance >= high)) | 0/20 | 1/80 | 0/20 | 1729/6000 | 3539/3983 | 0/20 | BORDERLINE |
| F G(clearance >= 0.2) | 0/20 | 0/80 | 0/20 | 2801/6000 | 0/14 | 0/20 | BORDERLINE |
| F G(clearance >= high) | 0/20 | 0/80 | 0/20 | 3021/6000 | 2435/4718 | 0/20 | BORDERLINE |
| F[0,277] G[0,23](clearance>=0.2) | 新增 | 80/80 | 11/20 | 8/6000 | 13/476 | 11/20 | BORDERLINE |
| F[0,253] G[0,47](clearance>=0.2) | 新增 | 78/80 | 7/20 | 23/6000 | 30/956 | 7/20 | BORDERLINE |

## 非退化及语义检查

| spec | 初始已完成 | 初始未完成路径P1/P2 | mode内打乱最高通过 | 随机函数最高通过 | 仅monitor rank通过 |
|---|---|---|---|---|---|
| high | 5 | 5/15 | 6/20 | 6/20 | 不适用 |
| return | 0 | 5/20 | 0/20 | 0/20 | 不适用 |
| goal | 0 | 6/20 | 0/20 | 0/20 | 不适用 |
| recovery8 | 0 | 0/20 | 0/20 | 0/20 | 不适用 |
| persist_safe | None | 0/20 | 0/20 | 0/20 | 不适用 |
| persist_high | None | 0/20 | 0/20 | 0/20 | 不适用 |
| safe_window24 | 0 | 11/20 | 0/20 | 0/20 | 15/20 |
| safe_window48 | 0 | 7/20 | 0/20 | 0/20 | 15/20 |

KEEP要求：train/internal所有P1/P2严格通过、P2非空，常数/按mode常数/打乱/随机对照不能全通过。
线性与ELU结果均保存在screening_summary.csv；不能把单步接近100%当作整路径100%。
新增窗口从t=0读取，允许在H=300内任意完整窗口完成；不同于原F[1,17]G[0,47]的64步规范。
Persistence仍是无限公式，有限样本不提供永久完成或全支持集closure证明；未发任何新warrant。
有限图可行仅表示已采样转移约束有解，不表示神经函数必能泛化，也不表示无限性质已证。

本轮KEEP函数数：0。
精确重算1600条候选路径；monitor独立窗口枚举、权重hash、数据来源及严格AND已核验。
