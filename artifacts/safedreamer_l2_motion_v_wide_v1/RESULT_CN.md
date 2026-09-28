# 固定低速→高速：48函数扩大搜索

48个新网络：12种架构/输入模板，各4个初始化与训练设置；每个4000步优化，部分另有400步辅助预训练。
固定阈值0.2245011670→0.7390221069、H=300、eta=.01、checkpoint、CCEPlanner及gate。
80条fit用于梯度与checkpoint选择，20条复用internal用于家族/尺度/组合选择。无时间输入、无未来标签输入。
每个网络4个输出尺度、3种固定混合权重及min/max；连同原19/20基线共433个候选。
没有读取正式calibration/Test1；结果只属开发阶段。

| 模板 | 该模板最佳候选 | train整路径 | internal整路径 | internal P1违反 | internal P2违反 | 状态 |
|---|---|---|---|---|---|---|
| elu_original | elu_original_s1_g8 | 80/80 | 19/20 | 10/6000 | 10/148 | BORDERLINE |
| elu256 | elu256_s0_mix0.5 | 80/80 | 19/20 | 9/6000 | 11/148 | BORDERLINE |
| elu_deter | elu_deter_s3_mix0.5 | 80/80 | 19/20 | 8/6000 | 9/148 | BORDERLINE |
| tanh | tanh_s0_mix0.5 | 80/80 | 19/20 | 9/6000 | 9/148 | BORDERLINE |
| silu | silu_s0_mix0.75 | 80/80 | 19/20 | 9/6000 | 10/148 | BORDERLINE |
| softplus | softplus_s3_mix0.5 | 80/80 | 19/20 | 7/6000 | 8/148 | BORDERLINE |
| residual | residual_s3_min | 80/80 | 19/20 | 11/6000 | 11/148 | BORDERLINE |
| mode_heads | mode_heads_s0_mix0.75 | 80/80 | 19/20 | 9/6000 | 10/148 | BORDERLINE |
| max8 | max8_s0_mix0.75 | 80/80 | 19/20 | 8/6000 | 9/148 | BORDERLINE |
| quadratic | quadratic_s1_mix0.75 | 80/80 | 19/20 | 10/6000 | 10/148 | BORDERLINE |
| fourier | fourier_s3_mix0.5 | 80/80 | 19/20 | 9/6000 | 10/148 | BORDERLINE |
| latent | latent_s0_mix0.75 | 80/80 | 19/20 | 9/6000 | 9/148 | BORDERLINE |

选中：{"id": "softplus_s3_mix0.5", "kind": "blend", "weight": 0.5}
train：P1 24000/24000，P2 750/750；internal：P1 5993/6000，P2 140/148。
完整AND：19/20；任务完成20/20；区域失败0、closure失败0。
对照整路径：{'constant': 0, 'mode_constant': 0, 'shuffle0': 1, 'random0': 0, 'shuffle1': 1, 'random1': 0, 'shuffle2': 1, 'random2': 3}
剩余失败：[{"path": 13, "p1_times": [20, 24, 28, 32, 34, 40, 41], "p2_times": [20, 23, 24, 28, 32, 34, 40, 41], "first_completion": 45}]
**内部集已反复用于选择，不能把其100%（若出现）当作新独立泛化保证。没有发SAFE，没有正式CP。**
全部433个候选含失败项保存在comparison.csv/report.json；selected_bundle.pt保留选中权重和组合规则。
