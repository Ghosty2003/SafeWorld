# SafeDreamer L3 W/U 候选筛选

规范保持 `GF G[0,47](decoded_hazard_margin>=0)`，checkpoint、CCEPlanner、
初始条件、192 步采样时域、epsilon_W=0.01、ell=0.8 不变。
始终安全也满足规范；不要求安全窗口之间必须发生 hazard。

## 数据与实验

复用 pilot 的 12 条训练路径、20 条开发验证路径。旧 validation 已看过结果，
本次明确改作开发数据，不再是独立最终验证。每个锚点保存 32 个独立一步后继。
12 条训练路径的 2304 个实际想象转移也用于训练辅助损失，不添加人工末尾转移。

W: plain / wide / automaton-mode heads / next-acceptance auxiliary supervision。
U: plain / sublevel shaping / automaton-mode heads / positive diagonal quadratic。
各运行种子 41、73，400 epochs，共 16 次拟合；再加原始 W/U 对照，比较 81 个配对。
右删失的 return-time 标签不当作永不返回；所有 U 检查包含接受状态，不启用 H2 豁免。
sublevel shaping 使用人工训练壳层，仅为启发式，不是认证边界。

## 结果：原始一步采样均值漂移检查

| 指标 | 原 pilot | 选中候选 |
| --- | --- | --- |
| 训练 W | 3/3 | 3/3 |
| 开发 W | 1/4 | 2/4 |
| 训练 U | 12/12 | 12/12 |
| 开发 U | 9/20 | 11/20 |

选中 `W_wide_73.pt` + `U_sublevel_41.pt`。W 是 128 宽、3 层隐藏层的 tanh MLP；
U 是 64 宽、2 层隐藏层的 MLP，使用训练集内部/人工壳层塑形。
W_mode_41 配相同 U 的主要排名分数并列，并非存在唯一最优解。
完整候选和配对排名见 report.json。

20 个开发锚点中 16 个已处于接受状态，因此 W 仅有 4 个需要检查；U 则检查全部 20 个。
选中 U 在 20 个初始状态和全部开发路径状态上均满足 U<=0.8，仍只是采样成员关系，
并不证明该集合包含所有可达状态或具有保留性质。

两者逐条路径、逐步的单后继漂移诊断均为 0/20 全程通过；这比期望漂移条件更强，
不能把它当作期望条件必然失败的证明，但说明不能声称所有采样转移都已满足。

## 结论与限制

换函数带来有限改善，尚未解决开发数据上的漂移失败。没有重新采最终独立后继数据，
所以本轮不输出置信下界；这些开发通过率不是无限 recurrence 的概率。
正式 L3 仍为 ABSTAIN / NOT_ESTABLISHED；全区域覆盖、包含、保留等证明条件也尚未建立。
不能据此断言规范为假、有效 W/U 不存在，或选中模型已经获得 warrant。

选中权重重新加载后，开发指标逐项精确复现；新旧相关 9 项自动测试通过。
旧 pilot 权重、报告、程序和 L2 结果均未修改。

复现命令（输出必须使用新目录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 \
/home/sunyhg/miniconda3/envs/safedreamer/bin/python experiments/l3_wu_search.py \
  --epochs 400 --seeds 41 73 --output artifacts/safedreamer_l3_wu_matrix_repeat
```

后续有价值的工作是增加独立训练路径、尤其是未接受区域中的训练锚点及其后继，
再用开发集挑选。最终选定后必须用全新独立后继验证，不能继续在本次开发集声称置信保证。
