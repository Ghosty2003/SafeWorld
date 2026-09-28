# SafeDreamer finite-window recurrence：fit-only pilot

已完成。新增独立入口，没有修改旧 TD-MPC2/L3 实验，也没有运行 κ-successor 期望漂移。
采样和拟合共约150.7秒。**M_MIN 未固定；正式 Δ、CP 下界和 C_rec 尚未产生。**

## 固定配置与四路隔离

- checkpoint：`20240307-010600_osrp_vector_safetygymcoor_SafetyPointGoal1-v0_0.ckpt`。
- 当前 CCEPlanner 配置逐项匹配旧 SafeDreamer policy manifest；不是 actor-only。
- H=300，301个状态、300条真实模型转移，eta=0.01。
- 动作为随机规划结果，RSSM随机性保留；采样在JAX GPU，小预算网络在PyTorch CPU训练。
- 起点：独立seed的环境reset，再编码为latent。沿用解码初始goal距离>=1.0条件。
  27次draw中保留16条，另11条只因起点条件不满足而排除，所有原始路径均保存。
- 16条新fit路径：前12条拟合g和输入范围，后4条为fit内部检查；不调参、不挑选epoch，训练400次更新。
- 第一条路径额外进行同seed重放，初始观测、latent、decoded、action、RSSM、planner状态逐元素相同。
- 旧reset未完全受imagination seed控制的问题，通过独立入口的实例级seed记录器处理；没有改旧wrapper文件。

| 数据角色 | 实际采样数（保留） | 状态 |
| --- | ---: | --- |
| fit | 16 | 12拟合 + 4内部检查 |
| cal_delta | 0 | LOCKED_NOT_COLLECTED |
| cal_CP | 0 | LOCKED_NOT_COLLECTED |
| test | 0 | LOCKED_NOT_COLLECTED，未运行1000条Test1 |

四路seed命名空间独立。入口若请求cal_delta/cal_CP/test，当前会拒绝运行，等待M_MIN等定义冻结。
本轮没有confidence或warrant判断。

## 事件与reset的精确定义

只检查后继状态1..300；状态0只作初始化。
每个状态的8个decoded hazard中心距离均>=0.2，则该状态安全。
连续48个安全状态完成一次事件，然后从下一状态开始下一段计数，不重叠计数。
hazard将未完成的安全计数清零，但**不能重置证书预算**；只有合法完成事件才允许预算reset。

一直安全会在48、96、144、192、240、288完成6次事件。这是真实完成多个安全窗口，
不要求中间主动进入hazard，也不宣称发生了物理区域的反复退出/返回。

hazard interruption count只统计hazard打断一个已有正长度的未完成safe run。
因此起始连续hazard、尚未开始safe run时，该数可以为0，但hazard state count不为0。

## 每条fit rollout

以下gate结果使用 **Delta_fit_preview=0.1316909003**，它仅来自前12条训练路径的最大整路径残差，
不是conformal校准结果。失败转移指预算在执行一步下降后、合法reset前已变负的转移。

| fit ID（0起） | 用途 | event count | hazard interruption | hazard states | 首/末事件时刻 | 不健全转移数 | 完整预览 |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 0 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 1 | 拟合 | 6 | 0 | 3 | 51 / 291 | 0 | PASS |
| 2 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 3 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 4 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 5 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 6 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 7 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 8 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 9 | 拟合 | 6 | 0 | 2 | 50 / 290 | 0 | PASS |
| 10 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 11 | 拟合 | 6 | 0 | 0 | 48 / 288 | 0 | PASS |
| 12 | fit内部检查 | 6 | 0 | 0 | 48 / 288 | 8 | FAIL |
| 13 | fit内部检查 | 6 | 0 | 0 | 48 / 288 | 40 | FAIL |
| 14 | fit内部检查 | 6 | 0 | 0 | 48 / 288 | 47 | FAIL |
| 15 | fit内部检查 | 6 | 0 | 1 | 49 / 289 | 44 | FAIL |

事件数分布：6次=16/16；未发现正在进行的safe run被hazard打断。
3条路径的hazard都发生在初始尚未开始安全段时。

## M_MIN=3/4/5/6 对照

| 候选M_MIN | 仅事件数量通过 | Delta=0完整预览 | Delta_fit：拟合12条 | Delta_fit：内部检查4条 | Delta_fit：合计 |
| --- | --- | --- | --- | --- | --- |
| 3 | 16/16 | 0/16 | 12/12 | 0/4 | 12/16，75% |
| 4 | 16/16 | 0/16 | 12/12 | 0/4 | 12/16，75% |
| 5 | 16/16 | 0/16 | 12/12 | 0/4 | 12/16，75% |
| 6 | 16/16 | 0/16 | 12/12 | 0/4 | 12/16，75% |

**不能用75%作为未来成功概率预测。** 其中12条参与了g、region和修正值的拟合。
后4条只是fit内部检查，也不是正式test。16条的小pilot没有区分四个M_MIN的能力；本轮没有自动选择它们。

## 全部gate的AND与失败统计

每条路径的预览通过条件是启用gate的完整AND，而不是仅计event：
drift AND delta margin AND region/Z_free-source gate AND transition soundness AND event-source consistency AND count。
CV/sliding尚未定义阈值，本pilot明确禁用，字段为null/NOT_APPLICABLE，不伪装成通过。
以后若启用，必须同样进入AND，并在独立calibration前冻结。

| failure reason | Delta=0失败路径数 | Delta_fit失败路径数 |
| --- | ---: | ---: |
| DRIFT_FAIL | 0 | 0 |
| DELTA_MARGIN_FAIL | 16 | 4 |
| REGION_GATE_FAIL | 16 | 4 |
| TRANSITION_SOUNDNESS_FAIL | 16 | 4 |
| EVENT_SOURCE_FAIL | 0 | 0 |
| INSUFFICIENT_EVENTS（所有候选M） | 0 | 0 |
| CV_FAIL | 不启用 | 不启用 |
| SLIDING_WINDOW_FAIL | 不启用 | 不启用 |

失败原因是多标签，同一路径可同时命中三项，不应相加当作路径数量。
所有16条路径均在本轮fit输入包络内（outside states=0）。这里的REGION_GATE_FAIL来自
低预算Z_free来源转移不健全，而不是几何输入包络越界。四条内部检查路径的最低到达预算分别约
-0.03233、-0.17524、-0.15331、-0.10739。

输入包络采用仅由12条拟合路径确定的标准化全状态L2球，半径55.51498；它是有限样本输入范围检查，
**不是已经验证的不变区域**。低预算区也不是旧L2的吸收完成集合。

## 预算证书的限制与固定基线

本轮沿用reset-budget形式：V_t=g_init(x_s)+Delta-eta*(t-s)。
输入含完整RSSM、planner状态和计数，不含未来信息；预算elapsed/段起点构成额外的监测状态。
训练可使用完成段的后验时长标签；右删失尾段只提供已观察时长下界。
尾段和每个事件到达的最后一条转移都参加验证，先检查下降后的预算，再执行合法reset。

**drift全部通过是构造属性，不代表网络学会了逐状态下降。** 12条拟合样本通过也受fit最大残差修正影响。
保留一个相同event/region gate的固定g_init=H*eta=3基线：它在16/16条路径上通过各候选M的预览。
这说明当前有限窗口预算证书可以由宽松预算满足，不能仅凭高通过率宣称学到了有预测价值的g或无限recurrence证明。
当前网络在4条内部检查路径上低估等待预算，这是后续需要考虑的拟合问题；本轮没有看结果后重训或放宽gate。

## 审计与文件

- 独立入口：experiments/safedreamer_finite_recurrence_pilot.py。
- 纯事件/预算逻辑：core/finite_recurrence.py。
- 复核：experiments/safedreamer_finite_recurrence_audit.py。
- 12项新单元测试通过，32项已有L3相关测试通过。
- audit.json精确重放全部16条路径的事件时间戳、300条转移检查、全部M_MIN预览及failure breakdown；
  核对模型/代码/原始数据哈希和四路数据未使用状态。
- fit_rows.csv：逐路径、逐Delta预览、逐M_MIN的所有gate与failure reason。
- report.json：完整预算轨迹、分段结果、事件时间戳、基线、统计汇总。
- g_init.pt、fit_model.json：pilot网络、仅fit修正值、输入包络；不是最终冻结模型。
- seed_replay.json、plan.json、fit/path_*.json与npz：来源、随机种子、起点筛选和原始轨迹。

首次v1在采样前因环境封装层识别失败而停止，保留STATUS.md；本v2为修复后的完成记录。

下一步等待用户根据fit-only结果确定M_MIN，并决定是否继续改进fit候选。
**cal_delta/cal_CP/test均未开始；没有发warrant，也没有计算CP下界。**
