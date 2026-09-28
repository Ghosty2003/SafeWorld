# 低速后进入高速：冻结校准与 Test1

| Layer | Specification | MP class | Backend / carrier | H | Verdict | p_hat_gamma (CP lower, 95%) | Test 1 (N=1000) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | F[0,300](LOW speed then strictly later HIGH speed) | Guarantee (bounded) | SafeDreamer / SafetyPointGoal1-v0 | 300 | ABSTAIN | 0.8664 | 898/1000, 0.8809 |

LOW <=0.2245011670；严格之后 HIGH >=0.7390221069；H=300，eta=.01。
内部开发仍为19/20，不是双100%合格；SAFE若出现，仅指预注册阈值下的统计有限模型结论。
无真实环境安全或无限时域保证；各下界为分别95%，不是联合95%。

| 指标 | 新校准500 | 校准CP下界 | 独立Test1000 | Test自身CP下界 |
|---|---|---|---|---|
| task | 500/500 | 0.9940 | 1000/1000 | 0.9970 |
| p1 | 450/500 | 0.8751 | 903/1000 | 0.8862 |
| p2 | 448/500 | 0.8708 | 899/1000 | 0.8819 |
| p1p2 | 448/500 | 0.8708 | 899/1000 | 0.8819 |
| certificate | 446/500 | 0.8664 | 898/1000 | 0.8809 |

calibration failure breakdown: {"p1": 50, "p2": 52, "endpoint": 0, "exclusion": 0, "closure": 0, "region": 2, "finite_nonnegative": 0}

test failure breakdown: {"p1": 97, "p2": 101, "endpoint": 0, "exclusion": 0, "closure": 0, "region": 1, "finite_nonnegative": 0}

两批未混用；全部raw、逐路径失败时间和V值保留；无测试后调参。
