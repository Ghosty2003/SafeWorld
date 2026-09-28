# 窗口连续进度函数追加对照

仅80 fit/20复用internal，H=300；无正式校准或Test1访问。
| 函数 | train整路径 | internal整路径 | internal P1违反 | internal P2违反 | 状态 |
|---|---|---|---|---|---|
| safe_window24_coefficient | 80/80 | 18/20 | 0/6000 | 2/476 | BORDERLINE |
| safe_window24_additive | 80/80 | 12/20 | 6/6000 | 16/476 | BORDERLINE |
| safe_window24_soft_progress | 80/80 | 6/20 | 16/6000 | 31/476 | BORDERLINE |
| safe_window48_coefficient | 80/80 | 15/20 | 2/6000 | 5/956 | BORDERLINE |
| safe_window48_additive | 80/80 | 6/20 | 1/6000 | 20/956 | BORDERLINE |
| safe_window48_soft_progress | 76/80 | 2/20 | 32/6000 | 48/956 | BORDERLINE |

remaining指还需多少个连续safe状态，由当前因果monitor决定，不是未来return time或剩余rollout时间。
三种结构都允许latent影响V；当前mode提供progress先验。接受态吸收且V=0，不代表物理区域永久安全。
每个模型均重算全部真实转移，并用独立逐路径AND复核；对照详见report.json。
