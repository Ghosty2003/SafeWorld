# SafeDreamer：非连续安全窗口的 L2 specification screening

本轮仅80条train-fit、20条复用internal-development；固定已审计checkpoint和CCEPlanner，H=300、eta=0.01、零数值容差。
8个不同事件任务×3函数家族×3预设尺度，共24个训练网络、72个尺度候选；每网络3000次优化更新。
未读取正式校准/Test1。内部集已多次参与开发选择，不能据此声明独立泛化置信保证。
没有重复“连续N步无hazard”窗口，也没有更改先前结果。

所有谓词顺序必须在0..300内完成，阶段之间至少间隔一个采样状态。q是因果阶段计数；接受态吸收；B是所有未接受模式。
safe_recover在完成前发生hazard即永久拒绝；完成之后不再要求无hazard。其余规范本身不保证避障。
Zfree={V<0.01}，坏模式V>=0.01，接受模式V=0。有限图可行只覆盖已有转移，不是无限时域证明。

## 阈值（全部来自fit）

{
  "clear_low": 0.47877358293460015,
  "clear_high": 0.7040214672720281,
  "speed_low": 0.22450116704240514,
  "speed_high": 0.7390221068978473,
  "goal_near": 0.8195899491365692,
  "goal_far": 1.699064482656371,
  "away": 0.9036861192804462
}

净空/目标距离/速度/位移都是decoder输出派生量。Goal-relative distance不是环境goal_met；位移相对decoded t0位置，不是实际里程。

## 每个任务最佳候选，包括没有成功的任务

| 任务 | fit任务完成 | internal任务完成 | fit整路径P1/P2 | internal整路径P1/P2 | internal P1违反 | internal P2违反 | 完整证书preview | 状态 |
|---|---|---|---|---|---|---|---|---|
| LOW后恢复到HIGH | 78/80 | 18/20 | 33/80 | 9/20 | 443/6000 | 807/1387 | 9/20 | BORDERLINE |
| 低速后进入高速 | 80/80 | 20/20 | 80/80 | 18/20 | 13/6000 | 14/148 | 18/20 | BORDERLINE |
| 高速后减速到低速 | 80/80 | 20/20 | 80/80 | 11/20 | 53/6000 | 56/444 | 11/20 | BORDERLINE |
| 远离目标后接近目标 | 43/80 | 9/20 | 13/80 | 4/20 | 1801/6000 | 3669/4107 | 4/20 | BORDERLINE |
| 接近、远离、再接近目标 | 22/80 | 7/20 | 0/80 | 0/20 | 2601/6000 | 2621/5197 | 0/20 | BORDERLINE |
| 离开起始位置 | 80/80 | 20/20 | 52/80 | 10/20 | 296/6000 | 564/1064 | 10/20 | BORDERLINE |
| 离开起始区域后返回 | 56/80 | 16/20 | 0/80 | 0/20 | 1582/6000 | 1596/3098 | 0/20 | BORDERLINE |
| 恢复到HIGH前不进入hazard | 66/80 | 12/20 | 25/80 | 5/20 | 1270/6000 | 1917/3087 | 5/20 | BORDERLINE |

P1检查全部300条转移；P2只在源模式属于B时要求下降eta。整路径比例不能由单步比例替代。
KEEP_DEV必须train/internal严格100%且常数、打乱、随机对照不全通过；否则BORDERLINE。没有内部完成的任务单独REJECT。

## 具体函数和失败解释

### LOW后恢复到HIGH

- 定义：[{"signal": "clearance", "op": "le", "threshold": 0.47877358293460015}, {"signal": "clearance", "op": "ge", "threshold": 0.7040214672720281}]
- 函数：recover_linear，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败11条；终点失败2条；区域gate失败0条；closure失败0条；安全拒绝0条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 0, 'shuffle0': 0, 'random0': 2, 'shuffle1': 0, 'random1': 0, 'shuffle2': 1, 'random2': 0}
- 前5个P2失败转移：[{"path": 0, "source_t": 22, "mode_from": 1, "mode_to": 1, "delta_V": -0.005515264447157442}, {"path": 4, "source_t": 7, "mode_from": 1, "mode_to": 1, "delta_V": 20.271456890462122}, {"path": 5, "source_t": 28, "mode_from": 0, "mode_to": 0, "delta_V": 0.0019185189170556338}, {"path": 5, "source_t": 29, "mode_from": 0, "mode_to": 0, "delta_V": -0.008333873275728543}, {"path": 5, "source_t": 32, "mode_from": 0, "mode_to": 0, "delta_V": 0.0008766129856687677}]
- 完成时刻（未完成记null）：[29, 17, 16, 8, 17, 183, 13, 3, 251, 67, 7, 16, 14, null, 33, null, 31, 4, 48, 30]
- 解读：任务未完成与证书函数失败均需单独检查；P1/P2不通过本身不是规范为假的证明。

### 低速后进入高速

- 定义：[{"signal": "speed", "op": "le", "threshold": 0.22450116704240514}, {"signal": "speed", "op": "ge", "threshold": 0.7390221068978473}]
- 函数：motion_burst_elu128，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败2条；终点失败0条；区域gate失败0条；closure失败0条；安全拒绝0条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 8, 'shuffle0': 2, 'random0': 2, 'shuffle1': 2, 'random1': 4, 'shuffle2': 1, 'random2': 4}
- 前5个P2失败转移：[{"path": 11, "source_t": 4, "mode_from": 1, "mode_to": 1, "delta_V": 0.012415582922968404}, {"path": 13, "source_t": 17, "mode_from": 1, "mode_to": 1, "delta_V": 0.03852216172464473}, {"path": 13, "source_t": 18, "mode_from": 1, "mode_to": 1, "delta_V": 0.02802023004681553}, {"path": 13, "source_t": 20, "mode_from": 1, "mode_to": 1, "delta_V": 0.07965011692511004}, {"path": 13, "source_t": 22, "mode_from": 1, "mode_to": 1, "delta_V": 0.07615923933783542}]
- 完成时刻（未完成记null）：[7, 6, 6, 4, 4, 7, 10, 5, 7, 3, 3, 6, 3, 45, 8, 3, 2, 4, 8, 7]
- 解读：20/20已完成行为，当前失败来自V/证书检查而不是任务未完成；放大V不能修复正的delta_V。

### 高速后减速到低速

- 定义：[{"signal": "speed", "op": "ge", "threshold": 0.7390221068978473}, {"signal": "speed", "op": "le", "threshold": 0.22450116704240514}]
- 函数：slow_down_elu128，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败9条；终点失败0条；区域gate失败0条；closure失败0条；安全拒绝0条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 0, 'shuffle0': 0, 'random0': 0, 'shuffle1': 0, 'random1': 0, 'shuffle2': 0, 'random2': 0}
- 前5个P2失败转移：[{"path": 1, "source_t": 14, "mode_from": 1, "mode_to": 1, "delta_V": 0.17799877535243125}, {"path": 1, "source_t": 15, "mode_from": 1, "mode_to": 1, "delta_V": 0.33324544668836475}, {"path": 1, "source_t": 20, "mode_from": 1, "mode_to": 1, "delta_V": 0.2347954547652058}, {"path": 1, "source_t": 22, "mode_from": 1, "mode_to": 1, "delta_V": 0.16372956627515012}, {"path": 1, "source_t": 23, "mode_from": 1, "mode_to": 1, "delta_V": 0.5857102332546622}]
- 完成时刻（未完成记null）：[12, 36, 17, 31, 7, 10, 17, 9, 36, 15, 40, 19, 20, 7, 48, 21, 27, 26, 22, 24]
- 解读：20/20已完成行为，当前失败来自V/证书检查而不是任务未完成；放大V不能修复正的delta_V。

### 远离目标后接近目标

- 定义：[{"signal": "goal", "op": "ge", "threshold": 1.699064482656371}, {"signal": "goal", "op": "le", "threshold": 0.8195899491365692}]
- 函数：goal_approach_elu128，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败16条；终点失败11条；区域gate失败0条；closure失败0条；安全拒绝0条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 0, 'shuffle0': 0, 'random0': 0, 'shuffle1': 0, 'random1': 0, 'shuffle2': 0, 'random2': 0}
- 前5个P2失败转移：[{"path": 0, "source_t": 23, "mode_from": 1, "mode_to": 1, "delta_V": -0.005421538379956958}, {"path": 0, "source_t": 24, "mode_from": 1, "mode_to": 1, "delta_V": -0.008437627282070759}, {"path": 0, "source_t": 25, "mode_from": 1, "mode_to": 1, "delta_V": -0.00025820126531561627}, {"path": 0, "source_t": 26, "mode_from": 1, "mode_to": 1, "delta_V": -0.0062574625860236655}, {"path": 0, "source_t": 27, "mode_from": 1, "mode_to": 1, "delta_V": -0.009335128938966042}]
- 完成时刻（未完成记null）：[37, 17, null, null, 215, null, 40, null, null, 265, null, null, 184, null, 14, null, null, 18, 17, null]
- 解读：任务未完成与证书函数失败均需单独检查；P1/P2不通过本身不是规范为假的证明。

### 接近、远离、再接近目标

- 定义：[{"signal": "goal", "op": "le", "threshold": 0.8195899491365692}, {"signal": "goal", "op": "ge", "threshold": 1.699064482656371}, {"signal": "goal", "op": "le", "threshold": 0.8195899491365692}]
- 函数：goal_revisit_linear，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败20条；终点失败13条；区域gate失败0条；closure失败0条；安全拒绝0条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 0, 'shuffle0': 0, 'random0': 0, 'shuffle1': 0, 'random1': 0, 'shuffle2': 0, 'random2': 0}
- 前5个P2失败转移：[{"path": 0, "source_t": 1, "mode_from": 0, "mode_to": 0, "delta_V": 1.0192424445830515}, {"path": 0, "source_t": 2, "mode_from": 0, "mode_to": 0, "delta_V": 0.2767231464107507}, {"path": 0, "source_t": 4, "mode_from": 0, "mode_to": 0, "delta_V": 0.36980390330029334}, {"path": 0, "source_t": 5, "mode_from": 0, "mode_to": 0, "delta_V": 0.1910046807464183}, {"path": 0, "source_t": 7, "mode_from": 0, "mode_to": 0, "delta_V": 0.37108269022925255}]
- 完成时刻（未完成记null）：[null, 96, null, null, 215, null, 62, null, null, 265, null, null, 184, null, null, null, null, 258, 217, null]
- 解读：任务未完成与证书函数失败均需单独检查；P1/P2不通过本身不是规范为假的证明。

### 离开起始位置

- 定义：[{"signal": "displacement", "op": "ge", "threshold": 0.9036861192804462}]
- 函数：leave_start_linear，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败10条；终点失败0条；区域gate失败0条；closure失败0条；安全拒绝0条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 0, 'shuffle0': 0, 'random0': 1, 'shuffle1': 0, 'random1': 0, 'shuffle2': 0, 'random2': 0}
- 前5个P2失败转移：[{"path": 0, "source_t": 25, "mode_from": 0, "mode_to": 0, "delta_V": -0.009876042216855896}, {"path": 0, "source_t": 32, "mode_from": 0, "mode_to": 0, "delta_V": 0.017713612451223826}, {"path": 0, "source_t": 33, "mode_from": 0, "mode_to": 0, "delta_V": 0.00312980124073603}, {"path": 0, "source_t": 34, "mode_from": 0, "mode_to": 0, "delta_V": 0.010346857435729051}, {"path": 0, "source_t": 35, "mode_from": 0, "mode_to": 0, "delta_V": -0.0019359806400893742}]
- 完成时刻（未完成记null）：[41, 16, 80, 75, 219, 282, 36, 75, 14, 7, 78, 13, 14, 6, 14, 4, 14, 16, 17, 43]
- 解读：20/20已完成行为，当前失败来自V/证书检查而不是任务未完成；放大V不能修复正的delta_V。

### 离开起始区域后返回

- 定义：[{"signal": "displacement", "op": "ge", "threshold": 0.9036861192804462}, {"signal": "displacement", "op": "le", "threshold": 0.4518430596402231}]
- 函数：return_start_linear，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败20条；终点失败4条；区域gate失败0条；closure失败0条；安全拒绝0条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 0, 'shuffle0': 0, 'random0': 0, 'shuffle1': 0, 'random1': 0, 'shuffle2': 0, 'random2': 0}
- 前5个P2失败转移：[{"path": 0, "source_t": 0, "mode_from": 0, "mode_to": 0, "delta_V": 2.3902033871591746}, {"path": 0, "source_t": 2, "mode_from": 0, "mode_to": 0, "delta_V": 1.2937713636865524}, {"path": 0, "source_t": 4, "mode_from": 0, "mode_to": 0, "delta_V": 3.077503650201079}, {"path": 0, "source_t": 7, "mode_from": 0, "mode_to": 0, "delta_V": 2.087989557258744}, {"path": 0, "source_t": 9, "mode_from": 0, "mode_to": 0, "delta_V": 0.19749936786662659}]
- 完成时刻（未完成记null）：[159, 38, 237, 118, 229, null, 260, 207, 26, null, 92, 37, 62, null, 174, null, 64, 41, 43, 111]
- 解读：任务未完成与证书函数失败均需单独检查；P1/P2不通过本身不是规范为假的证明。

### 恢复到HIGH前不进入hazard

- 定义：[{"signal": "clearance", "op": "le", "threshold": 0.47877358293460015}, {"signal": "clearance", "op": "ge", "threshold": 0.7040214672720281}]
- 函数：safe_recover_linear，输出乘4.0；初始已完成0/20。
- 失败项（可能重叠）：P1/P2失败15条；终点失败8条；区域gate失败0条；closure失败0条；安全拒绝6条。
- 对照整路径通过：{'constant': 0, 'mode_constant': 0, 'signal_prior_only': 0, 'shuffle0': 1, 'random0': 2, 'shuffle1': 0, 'random1': 0, 'shuffle2': 0, 'random2': 0}
- 前5个P2失败转移：[{"path": 0, "source_t": 24, "mode_from": 3, "mode_to": 3, "delta_V": -0.005816481608234891}, {"path": 0, "source_t": 25, "mode_from": 3, "mode_to": 3, "delta_V": 0.008902233025512185}, {"path": 0, "source_t": 28, "mode_from": 3, "mode_to": 3, "delta_V": 0.007155874517215111}, {"path": 0, "source_t": 30, "mode_from": 3, "mode_to": 3, "delta_V": 0.0378621307689635}, {"path": 0, "source_t": 32, "mode_from": 3, "mode_to": 3, "delta_V": 0.010819947484569159}]
- 完成时刻（未完成记null）：[null, 17, null, null, null, 183, 13, 3, 251, 67, 7, null, null, null, 33, null, 31, 4, 48, 30]
- 解读：任务未完成与证书函数失败均需单独检查；P1/P2不通过本身不是规范为假的证明。

## 此前还试过但未达到双100%的非窗口规范

| 规范 | 当轮最佳内部整路径P1/P2 | 说明 |
|---|---|---|
| F(clearance >= high) | 10/20 | 当前函数未达到双100%，不等于数学上无解 |
| F(LOW & F HIGH) | 4/20 | 当前函数未达到双100%，不等于数学上无解 |
| F(goal_distance < 0.3) | 6/20 | 当前函数未达到双100%，不等于数学上无解 |
| F(LOW & F G[0,7](clearance >= high)) | 0/20 | 当前函数未达到双100%，不等于数学上无解 |
| F G(clearance >= 0.2) | 0/20 | 当前函数未达到双100%，不等于数学上无解 |
| F G(clearance >= high) | 0/20 | 当前函数未达到双100%，不等于数学上无解 |

上表是早期6规范筛选；高净空任务后来单独换函数曾提高到13/20，仍未全通过。
## 结论

本轮KEEP_DEV尺度候选数：0。
低速后进入高速最接近双100%，但多数路径在2–10步完成（另有45步）；它衡量启动/运动能力，不代表避障、导航或长期recurrence。
本轮只筛选，不发SAFE，不计算CP下界，不启动正式calibration/Test1。
所有未成功项保留在all_candidates.csv；failure_breakdown.csv按gate给出失败原因，report.json保留每条路径结果。
复核：7200条尺度候选路径、全部随机/打乱对照、独立first-hit事件检测、完整AND、数据/权重哈希重算通过。
