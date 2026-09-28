# SafeDreamer L3 论文 H1 路线：首次候选训练与独立点验证

## 结果与完成范围

**ABSTAIN / NOT_ESTABLISHED。没有发出论文正式无限时域 warrant。**

已实现并跑通：完整 product-state 采样、非吸收 Büchi 计数器、W/U 训练、
冻结后新路径和独立一步后继验证、Hoeffding 修正、逐项论文前提报告。
尚未实现：SafeDreamer RSSM + CCEPlanner 的全局认证覆盖/包含/保留证明后端。
因此本程序是候选验证与证明缺口审计入口，不能宣称完整无限证明器已实现。

规范：`GF G[0,47](decoded hazard margin >= 0)`。
接受计数器到 48 后若遇 hazard 仍清零。采用论文 H1，U 在 C 中的接受状态
也要检查，不假设“连续安全计数 >=24”是前向不变核心。

## 本次运行

- 原 checkpoint 与本地 CCEPlanner 均未改变；JAX GPU imagination。
- 12 条全新训练路径，20 条全新独立验证路径，每条 192 步 / 193 状态。
- 共生成 48 条原始路径；16 条因起点 decoded goal distance <1.0 被排除。
  排除只看 t=0，不按未来成功率筛选。
- 每条保留路径均匀随机选择一个 t∈[1,191] 的锚点，采 32 个独立随机后继。
  每次恢复完整 RSSM 与 planner carry，不把连续 32 步冒充独立后继。
- 共 32 个路径锚点、1024 个一步后继；训练 W/U 500 epochs 后冻结，再采验证集。
- 主试验约 3.29 分钟；小样本接口测试和程序开发不计入。

## 证书条件结果

| 检查 | 训练集经验通过 | 独立验证经验通过 | 独立验证加采样修正后 |
| --- | ---: | ---: | ---: |
| 非接受状态 W 期望下降至少 0.01 | 3/3 | 1/4 | 0/4 |
| C 中 U 期望不增加（包括接受状态） | 12/12 | 9/20 | 0/20 |

验证锚点中有 16 个已处于接受状态：这些点不要求 W 严格下降，但仍检查 U。
这些比例以状态锚点为单位，不是整条路径满足率。

每函数/锚点分配 delta_i=0.05/(2×20)=0.00125，K=32、函数结构上界 B=1，
Hoeffding 加项约 0.32318。它比目标 0.01 漂移大得多；本轮查询预算十分保守，
不能把经验均值直接称作已认证的期望界。训练点通过也未转化成独立验证泛化。

## 同批新验证路径的有限行为

- 至少两个不重叠的 48 状态安全窗口：20/20。
- 后半段内仍有完整 48 状态安全窗口：20/20。
- 全程无 hazard：14/20。
- 每条最长连续安全段最少 186 状态，中位数 193 状态。

因此“有限路径表现很好”和“候选 W/U 未能认证”可以同时成立。
这些是 learned-model decoder 的行为，不是对真实环境精度的评价，
也不是无限 recurrence 满足概率为 100%。

## 正式证明仍缺什么

RSSM/planner 闭环有效的单元误差界、相关区域覆盖、C 子水平集包含关系、
整个 C 上 H1 保留条件、初始分布支撑、初始 U 期望上界和 collar 进入概率界。
程序明确列为 NOT_ESTABLISHED，不接受手填通过布尔值来发正式证书。
即使将来采样点全部通过，没有上述证明后端也不能得到全局 warrant。

## 验证与复算

17 项相关单元测试通过。另审计 32 个源状态、1024 个后继的完整 latent、
planner carry 和自动机计数，均与保存数据重建一致；各路径指纹无重复。
证书/计划/源码哈希检查与报告复算通过。

注意：首次使用默认 CPU 线程数复算，因浮点差异触发严格报告比对失败；
恢复原训练/评价线程配置后精确复算通过。原始报告未替换，运行如下：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
/home/sunyhg/miniconda3/envs/safedreamer/bin/python \
  experiments/l3_safedreamer_paper.py --evaluate-only \
  --output artifacts/safedreamer_l3_paper_pilot192
```

程序：experiments/l3_safedreamer_paper.py
模型：certificates.pt（W/U 的 state_dict 与冻结的训练归一化）
完整结果：report.json；冻结记录：frozen.json；实验计划：plan.json。
L2 原有结果未改动。试验已结束，无后台训练在继续。
