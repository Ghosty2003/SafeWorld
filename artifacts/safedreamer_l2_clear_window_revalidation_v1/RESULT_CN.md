# 冻结48-state规范：新校准与Test1

Layer | Specification | MP class | Backend / carrier | H | Verdict | p_hat_gamma (CP lower, 95%) | Test 1 (N=1000)
--- | --- | --- | --- | --- | --- | --- | ---
L2 | F[1,17] G[0,47](¬hazard) | Guarantee (bounded) | SafeDreamer / SafetyPointGoal1-v0 | 64 | SAFE | 0.9739 | 970/1000, 0.9595


固定原linear_max4（step4800）V、eta=.01、48-state detector、H=64、checkpoint和CCEPlanner。
500条新校准先固定结论，再生成1000条独立Test1。无训练、调参、根据结果追加样本或未来成功筛选。
与旧实验一致，仅按初始解码goal距离>=1.0筛起点；现在显式记录实际gym reset及imagination独立seed。
这里SAFE仅指固定有限时域model-scope完整证书事件概率下界过门槛，不是无限时域安全或支持集全局证明。

| 指标 | calibration500 | Test1 1000 |
|---|---|---|
| 完整证书 | 493/500 | 970/1000 |
| 完成48-state窗口 | 500/500 | 1000/1000 |
| 单侧95%证书CP下界 | 0.973865 | 0.959528 |
| P1违反转移 | 1 | 20 |
| P2违反转移 | 7 | 30 |
| 未完成却通过完整证书 | 0 | 0 |

校准判定原因：{"required_probability": 0.95, "confidence": 0.95, "lower_bound": 0.9738652881320734, "probability_threshold_met": true, "no_observed_region_counterevidence": true, "result": "FINITE_EVENT_THRESHOLD_MET", "reasons": [], "scope": "Finite model-only certificate AND completion probability under fixed initial distribution and policy. Not infinite-horizon safety or support-wide proof."}
Test1 sublevel检查：NO_VIOLATION_ON_OBSERVED_SOURCES
Test1自身下界独立报告，不替代校准下界、不用于重选V；这些95%置信界是各自的界，不是联合95%声明。
没有观察到错误发证也不能证明错误概率为零。有限样本预测成立依赖固定模型、策略、起点分布与独立同分布抽样假设。
审计：所有raw（含起点被排除者）均留存；seed/哈希/时序/完整AND/独立窗口检测及指标重算核验通过。
