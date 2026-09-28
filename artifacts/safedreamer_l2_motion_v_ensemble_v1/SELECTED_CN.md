# 低速→高速：本轮更好的V

固定阈值speed<=0.2245011670后再speed>=0.7390221069；H=300，eta=0.01。
当前结果：BORDERLINE，未达到双100%，无正式SAFE结论。

| 指标 | 原V | 新组合V |
|---|---|---|
| train整路径P1/P2 | 80/80 | 80/80 |
| internal整路径P1/P2 | 18/20 | 19/20 |
| internal P1 | 5987/6000 | 5990/6000 |
| internal P2 | 134/148 | 136/148 |
| internal任务完成 | 20/20 | 20/20 |
| internal完整certificate AND | 18/20 | 19/20 |

新V=0.75×(4×原ELU128函数)+0.25×(4×新full_residual函数)。两个输入处理分别随模型冻结；不是重训policy。
selected_manifest.json给出精确组合及路径/hash；selected_bundle.pt保留所有所需权重备份。
常数/打乱/随机对照整路径通过：{'constant': 0, 'mode_constant': 0, 'shuffle0': 1, 'random0': 3, 'shuffle1': 2, 'random1': 4, 'shuffle2': 1, 'random2': 8}
剩余失败路径：[{"path": 13, "completion_t": 45, "p1_failure_times": [8, 17, 20, 24, 28, 31, 32, 39, 40, 41], "p2_failure_times": [8, 17, 20, 22, 24, 28, 31, 32, 36, 39, 40, 41], "max_delta": 0.2858376285397648}]

训练仅80fit；20internal用于选择家族/尺度/组合，未加入梯度训练，但已反复调参，因此19/20不是独立统计证据。
完整AND、权重hash、选中值数组及独立加权公式已重算一致。旧48-state SAFE和24-state KEEP_DEV结果不变。
本轮没有正式calibration、CP下界或Test1。
