# SafeDreamer L3 双区域候选筛选

候选规范：GF(d_goal < 0.5) AND GF(d_goal > 1.0)。
距离来自 observation decoder 的 [7:9] 向量范数。
这是探索性有限前缀筛选，没有训练 W/U、计算置信下界或发证。

数据：已有 safedreamer_goal03_policy100/rollouts.npz 的全部 100 条路径；
初始 decoded goal distance >= 1.0，固定原 checkpoint 和本地 CCEPlanner。
没有重新生成数据，没有按未来成功过滤，没有更改任何 L2 结果。
源文件哈希及逐路径 A/B/A 时刻在 report.json。

| 指标 | 同批路径前 64 步 | 完整 100 步 |
| --- | ---: | ---: |
| 曾进入 A | 71/100 | 76/100 |
| 至少一次 A→B→A | 38/100 | 46/100 |
| 至少两次 A→B→A | 16/100 | 26/100 |
| 至少三次 A→B→A | 0/100 | 6/100 |
| 总完成往返次数 | 54 | 82 |

100 步中，31 条在第 50 步之后完成过往返；12 条有整次往返
（起始 A、B、最终 A）都在第 50 步之后。相邻往返允许共享末端 A，
起点 B 不算一次往返；中间区域保持监控进度，严格使用 <0.5 与 >1.0。
这些统计不是无限 recurrence 的成功/失败判定。

## Goal 重置审计

本地 safety-gymnasium/bases/base_task.py 的 continue_goal 默认 True。
builder.py 在 goal_achieved 时调用 task.update_world()；
goal_base.py 的 update_world() 调用 build_goal_position() 重建目标。
SafeDreamer 的 safetygymcoor.py 读取当前 task.goal.pos 构造相对 goal 向量。

因此，目标距离变大不一定代表离开同一个目标，也可能对应目标重置。
本次数据是在 reset 编码之后纯 imagination，未来并未调用真实环境的
重置逻辑；也没有每一步可靠的物理 goal ID。模型可能学到相关重置模式，
但本次审计无法把某一次距离跳变归因为 goal 重置、运动或模型误差。

结论：作为“decoded 当前目标距离近/远交替”的候选具有有限行为证据；
不能解释为固定物理区域往返，也不能宣称高概率或无限 L3 WARRANT。
若要求固定地标巡逻，需要稳定坐标/目标身份的额外状态及策略行为依据。
若接受动态当前目标语义，可先冻结该语义再采更长的独立开发路径筛选。
目前不应直接启动正式校准发证。

验证：test_l3_dual_region_screen 与 test_l3_safedreamer_screen 共 6 项通过。
