# SafeDreamer L3 接入准备

依据本地 `origin/main` 提交 `055ab637d35114e7f81099ba17eabadd407996ad`。
未切换分支、合并代码或修改已冻结 L2 文件。

## 已完成

核对 `core/lbsm`：当前分支与该 main 引用中的核心文件无差异。
新增 `experiments/l3_safedreamer_screen.py`，对已有 100 条、100 步
SafeDreamer CCEPlanner imagination 路径做探索性筛选，结果见 report.json。
源分布是初始 decoded goal distance >= 1.0；没有按未来成功筛选。
两个单元测试通过。没有训练 L3 证书、生成新 rollout 或发出 warrant。

| 候选 p（研究 GF p） | 曾满足 p | 至少两段分开的 p 区间 | 前后半段各出现 p |
| --- | ---: | ---: | ---: |
| goal distance < 0.3 | 54/100 | 40/100 | 29/100 |
| goal distance < 1.0 | 84/100 | 57/100 | 57/100 |
| hazard margin >= 0 | 100/100 | 10/100 | 100/100 |

87 条路径始终 hazard-free，因而不发生重新进入。这不是 GF 失败：
永远保持 p 为真也满足 GF p。以上所有数字都不是无限时域满足率。
goal 指 decoder 中的相对 goal vector；尚未证明长期 imagination 中
goal 重置与真实任务一致，不能解释成完成多个真实环境目标。

## 下一步需固定规范及结论范围

main 有不同入口，不能混为一谈：

- `tdmpc2/l3_generic_lbsm_lib.py`：峰值/区间统计性实验；包含离线峰值分段和
  区间内人工倒计时，不直接复制为 SafeDreamer 状态函数证书。
- `core/lbsm/distributional_pipeline.py`：训练 W/U、期望算子，再进行独立
  残差校准与 warrant-anchor 验证。输出仅是分布内漂移条件质量下界，
  不是整条路径 recurrence 概率，也不是无限时域全局 L3 证明。

拟采用第二条作为独立接入口，需要：

1. 固定 GF p 的 p、保留集合 I、初始状态分布与 anchor 时间采样分布。
2. 保持 checkpoint 和本地 CCEPlanner 配置；完整输入包含 RSSM 和 planner carry。
3. 每条独立路径预先随机选择一个 anchor，再从相同完整状态独立采样多个一步后继。
   不能把同一路径所有相关时间点当成独立二项样本。
4. 分开 W/U 训练、算子训练、残差校准、warrant 验证；若调参再增加独立开发集。
5. 最后另采测试路径，报告有限时域 recurrence 诊断，禁止将其称作无限 GF 真值。

预算注意：main 默认 alpha_W=alpha_U=0.05、delta_W=delta_U=0.01，
总算子误差预算是 0.12。因此 drift_valid_mass_lower 的上限不超过 0.88，
不能沿用这些默认值同时要求 theta=0.95。若仍要求 95% 门槛，必须事先
重新设计误差预算和样本量，而不是跑完再降低门槛。L2 的 95% 路径事件门槛
与此处的漂移条件质量门槛也不是同一个事件。
