# SafeDreamer LOW→HIGH recurrence：fit-only pilot

已完成。新事件有明显计数差异，但预算泛化和“大预算退化”仍存在。
未选正式阈值或 M_MIN；未启动 cal_delta、cal_CP、test；无 CP 下界、无正式 warrant。
本结果不是无限 recurrence 证明，也不是原论文 Theorem 5.6。

## 数据与固定条件

- 复用已审计的 `safedreamer_finite_recurrence_pilot_v2` 的 16 条模型 imagination 路径；本轮没有新采样。
- 12 条 train-fit、4 条 internal-fit-check，全部属于 D_fit。内部检查曾用于开发，不是未触碰的正式测试。
- 每条 H=300，含初始状态共301个状态；完整检查300条转移，包含末尾转移和事件到达边。
- 固定原 SafeDreamer checkpoint、CCEPlanner.policy()、RSSM 随机动力学和实际 planner 配置，未改成 actor-only。
- 保留原始起点分布：独立环境 reset 后编码，初始 decoded goal distance>=1.0。没有按未来事件成功与否筛路径。
- 环境仅提供初始 reset 状态；路径来自 world model，不是实环境 rollout。
- 四个候选各自重新训练同结构 g_init，400次更新、相同训练种子。输入为完整 latent、planner carry、当前 ARMED 位，无未来信息和时间标签。
- eta=0.01。Delta_fit 是12条训练路径的最大整路径残差，仅供有偏 fit 预览，未做 conformal calibration。
- CV/sliding 延续原 pilot 的明确禁用状态，不将其写成已验证。

## 事件定义与审计

`d_hazard = min(norm(decoded[9:25].reshape(8,2)))`。

UNARMED 遇 d<=d_low 后 ARMED；之后遇 d>=d_high 才发 event 并退回 UNARMED。
初始状态 t=0 可以 ARM，但不能发 event；return 必须发生在严格更晚的状态。
一直 HIGH 不会发事件；中间带内抖动不会重复发事件；LOW 不重置预算，只有合法 return 才重置。

记录两种 LOW 数：

- N_low_entries：真正由 UNARMED→ARMED 的次数，与每次 return 配对，末尾可有一次未完成返回。
- N_low_excursions：原信号每次进入 d<=d_low 区间的次数。已 ARMED 时再次进 LOW 不算新的 arming。

因果 detector 与独立区间搜索逐时间戳核对。核对内容含阈值、LOW/return/event/reset 时间、clearance 和 ARMED 状态，不只核对计数。
`traces.json` 保存每次 reset 的 pre/post budget、event id、LOW/return clearance，以及完整逐步预算。
事件发生时先检查到达边上的 pre-reset budget，不能用重置掩盖负预算。
右删失尾段也检查已观察的消耗，但不臆造未来返回。

## Clearance 分布

分位数按相关的采样状态池统计，仅为描述统计，不把4816个状态当独立样本。
四组阈值仅来自12条 train-fit 的3612个状态；没有用内部检查、calibration 或 test 选择阈值数值。

| 数据 | minimum | Q25 | median | Q75 | Q90 | mean |
| --- | --- | --- | --- | --- | --- | --- |
| 12条 train-fit | 0.0574 | 0.4736 | 0.6196 | 0.7910 | 1.0272 | 0.6632 |
| 4条 internal-fit-check | 0.1501 | 0.5231 | 0.6590 | 0.8307 | 1.0138 | 0.6884 |
| 合计16条 | 0.0574 | 0.4851 | 0.6290 | 0.8031 | 1.0224 | 0.6695 |

每条路径的 minimum、mean、Q25/50/75/90 等见 `clearance_distribution.json`。
LOW/HIGH 是相对 decoded clearance 概念，不等同于“进入 hazard/离开 hazard”，也不证明物理真实距离准确。

## 四组候选，不冻结选择

| 候选 | 分位数 low/high | d_low | d_high | 每条return范围 | 中位数 | 平均数 | 至少2次return |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Q25/Q50 | 0.473598 | 0.619625 | 1–8 | 5 | 4.5000 | 15/16 |
| B | Q25/Q75 | 0.473598 | 0.791038 | 1–4 | 2 | 2.1250 | 12/16 |
| C | Q10/Q90 | 0.382907 | 1.027161 | 0–2 | 1 | 0.6250 | 1/16 |
| D | Q50/Q75 | 0.619625 | 0.791038 | 2–6 | 3 | 3.0625 | 16/16 |

本批数据确实出现多次 LOW→HIGH，而不是持续 HIGH 被自动计数。
A 的返回多，C 的 gap 最大且返回稀少；B、D 在这批数据上同时有重复返回和计数差异。
这只说明候选信号非恒定，不证明跨种子/新数据的稳定性；没有据此自动选择 B 或 D。

## 每条路径：LOW arming 次数 / return 次数

| fit index | imagination seed | 分组 | A | B | C | D |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 121000001 | train | 2/2 | 1/1 | 1/0 | 2/2 |
| 1 | 121000005 | train | 6/5 | 3/2 | 2/1 | 3/2 |
| 2 | 121000009 | train | 3/3 | 2/2 | 1/1 | 2/2 |
| 3 | 121000015 | train | 5/5 | 1/1 | 1/1 | 3/2 |
| 4 | 121000017 | train | 5/5 | 2/1 | 2/1 | 3/2 |
| 5 | 121000019 | train | 1/1 | 1/1 | 0/0 | 2/2 |
| 6 | 121000025 | train | 6/5 | 3/2 | 2/1 | 4/3 |
| 7 | 121000029 | train | 6/6 | 4/4 | 2/2 | 4/4 |
| 8 | 121000031 | train | 3/3 | 2/2 | 1/0 | 5/5 |
| 9 | 121000035 | train | 9/8 | 4/3 | 1/0 | 4/3 |
| 10 | 121000039 | train | 4/4 | 2/2 | 1/1 | 2/2 |
| 11 | 121000043 | train | 7/7 | 3/3 | 1/0 | 3/3 |
| 12 | 121000045 | internal_check | 5/5 | 4/3 | 1/0 | 7/6 |
| 13 | 121000047 | internal_check | 5/4 | 3/2 | 1/0 | 4/3 |
| 14 | 121000051 | internal_check | 5/4 | 3/2 | 2/1 | 4/3 |
| 15 | 121000053 | internal_check | 5/5 | 3/3 | 1/1 | 6/5 |

所有 LOW/return/event 时间、inter-event intervals、未完成 LOW 和单独 excursion 次数在 `rollout_event_counts.csv`。

## M_MIN 扫描：全部仅为 fit-only 预览

每格为“计数条件通过数 / 完整 gate AND 通过数”；两者分母都是16。
完整预览用各候选自己的 g_init + Delta_fit，不是正式证书。

| M_MIN候选 | A | B | C | D |
| --- | --- | --- | --- | --- |
| 1 | 16 / 12 | 16 / 14 | 9 / 9 | 16 / 14 |
| 2 | 15 / 11 | 12 / 10 | 1 / 1 | 16 / 14 |
| 3 | 14 / 10 | 5 / 4 | 0 / 0 | 9 / 7 |
| 4 | 12 / 8 | 1 / 1 | 0 / 0 | 4 / 3 |
| 5 | 9 / 7 | 0 / 0 | 0 / 0 | 3 / 2 |
| 6 | 3 / 3 | 0 / 0 | 0 / 0 | 1 / 1 |
| 7 | 2 / 2 | 0 / 0 | 0 / 0 | 0 / 0 |
| 8 | 1 / 1 | 0 / 0 | 0 / 0 | 0 / 0 |
| 9 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

超过某候选最大计数的格均为必然零；程序为每组至少扫1–4，并扫到其最大计数+1。
例如 M=2 的计数满足率 A/B/C/D 为93.75%/75%/6.25%/100%；
完整预览率为68.75%/62.5%/6.25%/87.5%。这些不是总体概率下界。
M=1 只要求一次返回，不能据此称为已展示多次 recurrence。

## Gate failure breakdown

下面以 M=1 展示；其他 M 的全部明细在 `failure_breakdown.csv`。

| 候选 | Delta_fit | event不足 | drift失败 | Delta margin失败 | region gate失败 | soundness失败 | source失败 | 内部检查完整通过 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.370511 | 0 | 0 | 4 | 4 | 4 | 0 | 0/4 |
| B | 0.242766 | 0 | 0 | 2 | 2 | 2 | 0 | 2/4 |
| C | 0.209113 | 7 | 0 | 1 | 1 | 1 | 0 | 2/4 |
| D | 0.214503 | 0 | 0 | 2 | 2 | 2 | 0 | 2/4 |

这些失败计数允许重叠，不能相加当失败路径总数。
全部路径均在经验几何 envelope 内，region gate 的失败来自低预算源状态的转移检查，
与 Delta margin/soundness 是同一预算不足的不同表现，不是三种独立故障。
C 中一条预算失败路径同时没有 event，因此补预算并不改善该组 M=1 的完整成功数。

drift 全过也不能视为网络学好了：这里是 realized reset budget，每步直接扣 eta，
其下降由构造保证。网络学习的是 reset/init 时给多少预算，而非论文的 conditional expected drift。
训练路径的 Delta_fit 取训练最大残差，本来就会消掉它们的预算失败；这不是泛化证据。

## 大预算退化仍存在，但不能伪造返回

另算常数 g_init=H*eta=3 的基线，沿用同一个 event、region 和完整 gate AND。
本批数据上它使所有路径的预算检查通过；最后通过数恰好等于上表的“计数条件通过数”。
例如 D 在 M=2 时常数基线16/16，而学习预算14/16；C 在 M=2 时二者都只有1/16。

所以新 detector 修复了“持续 HIGH 就自动产事件”的问题，却没有自动解决证书预算的非退化性。
大预算不能增加 event count，也不能证明未来会返回；它仅保证这300步内预算不耗尽。
当前 fit pilot 不能据此宣称学到了有预测判别力的证书，也不能升级为无限时域 warrant。

## 核验与文件

- 新增10项 detector/预算测试，连同原 pilot 和 budget ablation 共26项测试通过。
- 单独从已保存的四组权重重放192组逐路径诊断、1200个 gate-AND 决策，精确一致。
- checkpoint、原代码、原路径哈希核对；旧实验和原始结果未修改。
- `audit.json` 与 `audit_replay.json` 保存核验记录。
- 正式阈值=null，M_MIN=null，cal_delta=0，cal_CP=0，test=0；到此停止。

所要求的六个输出均已保存：`clearance_distribution.json`、`threshold_candidates.csv`、
`rollout_event_counts.csv`、`rollout_certificate_results.csv`、`failure_breakdown.csv`、本文件。
此外保存四组 g_init 权重、plan/report 和逐步预算/reset记录 `traces.json`。

入口：`experiments/safedreamer_high_clearance_pilot.py`。
复核：`experiments/safedreamer_high_clearance_audit.py`。
