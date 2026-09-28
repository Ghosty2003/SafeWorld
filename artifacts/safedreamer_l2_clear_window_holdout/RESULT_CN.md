# 当前结果存档：连续 48 个无 hazard 状态

本文件整理已经完成的冻结实验，不重新训练、采样或更改原判定。

## 固定的实验

- 规范：`F[1,17] G[0,47](decoded hazard_margin >= 0)`。
- 含义：从第 1–17 步中选择一个起点，连续 48 个采样状态无 hazard，最迟第 64 步完成。
- 不要求到达原 goal，也不要求整个 64 步始终安全；选定区间之外允许 hazard。
- 数据：SafeDreamer world model imagination，动作来自同一本地 CCEPlanner。
- 初始分布保持解码 goal 距离 >=1.0；这只是采样条件，不是本规范的完成条件。
- V：`linear_max4`，第 4800 次更新；eta=0.01；没有使用新校准/测试数据调参。
- 输入：latent、planner 状态及连续 clear 计数；没有未来结果或剩余时间倒计时。

## 结果

| 指标 | 新校准集 | 单侧 95% 校准置信下界 | 独立测试集 |
|---|---:|---:|---:|
| 完成当前规范 | 100/100 | 97.05% | 100/100 |
| 整条路径同时满足 P1/P2 | 98/100 | 93.84% | 97/100 |
| 完整证书事件 | 98/100 | 93.84% | 97/100 |
| 进入候选 Z_free | 100/100 | 97.05% | 100/100 |

完整证书：P1/P2 在整条路径成立、终点 V<eta，而且规范已完成。
校准点估计 98%，测试实际 97%，差 1 个百分点。测试中有 3 条路径虽然
完成了规范，但未全部满足当前 V 的 P1/P2，因此没有通过完整证书。
100/100 是观测结果，不是未来所有路径必然完成的保证。

## 正式判定与事后解读分开保存

- 原预设门槛 theta=95%，统计置信度=95%。
- 正式结果：**NO_WARRANT**，因为完整证书事件下界 93.84% <95%。
- 区域检查未发现反例；NO_WARRANT 不表示已经观察到规范违反。
- 事后按 theta=90% 解读：93.84% >=90%，满足该较低概率要求。
- 90% 不是本次采样前预设的门槛；仅作讨论备注，不覆盖 `plan.json`、
  `prediction.json`、`report.json` 中的原始 95% 门槛和正式结果。
- 规范完成事件的 97.05% 下界与完整证书事件的 93.84% 下界不是同一指标。

以上统计解释依赖相同初始分布、相同策略和独立同分布采样等假设；不作
真实环境、无限时域或 support-wide 演绎证明。各下界是单独置信区间。

## Z_free 的具体意义

当前 `Z_free={V<0.01}` 在这些路径上与“已完成规范”的监测状态一致。
完成分支的学习值约 0.002502，接受状态吸收。因此它是 model/monitor
乘积状态中的完成集合，不是“以后永远不会遇到 hazard”的物理安全区域。

校准/测试均未发现未完成就进入的状态。区域内来源转移分别为 1569/1585，
没有观察到 P1/P2 违反或退出。首次进入范围分别为第 48–54 /48–55 步。
这些结果不能证明所有未见过的低值状态也有效。

## 文件位置

- `selected.pt`：冻结的 V 权重和训练归一化参数。
- `frozen.json`、`plan.json`、`policy_config.json`：模型身份、规范、门槛和采样配置。
- `prediction.json`：在测试开始之前保存的校准结果和判断。
- `report.json`：校准、测试指标及比较。
- `calibration.npz`、`test.npz`：各 100 条保留路径。
- `calibration_raw/`、`test_raw/`：原始采样批次，包括起点筛除与配额外样本。
- `*_provenance.json`、`prior_identities.json`、`test_started.json`：来源、哈希与时序。
- `SUMMARY.md`：更完整的英文报告。
- `../safedreamer_l2_clear_window_holdout.log`：本轮日志，不是 IDE 中的旧 vrefine 日志。

原始训练数据和候选拟合仍保留在相邻的 `safedreamer_l2_approach1_more_data/`
及 `safedreamer_l2_clear_window_v_linear6000/` 中。

重新核对（不重新采样、不修改 V）：

```bash
cd /home/sunyhg/Documents/SafeWorld
/home/sunyhg/miniconda3/envs/safedreamer/bin/python experiments/l2_clear_window_holdout.py --evaluate-only
```

本次为本地文件保存；未执行 git commit 或 push。
