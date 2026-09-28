# SafeDreamer recurrence specification screening

## 结论

已完成6类规范、每类2组阈值，共12组有限筛选。
**主选：clearance_q25_75；备选：approach_q25_75。两者均为 BORDERLINE。没有候选达到 KEEP。**
这是选择下一步研究对象，不是认定证书已可用。预算退化和泛化问题没有通过换规范自动解决。
主选的定义、阈值、权重和规则已写入 `frozen_primary.json`，不自动运行 calibration。

## 数据边界与实验公平性

- 仅复用16条已审计的 D_fit imagination 路径：12条训练、4条fit内部验证。H=300，无新采样。
- 固定 SafeDreamer checkpoint、CCEPlanner.policy() 和原起点分布；未改为随机动作或actor-only。
- 阈值只使用12条train-fit的状态分布。每种信号最多Q25/Q75和Q10/Q90两组，计划先写入plan.json再训练。
- M_MIN统一为2；dwell=1个采样状态，refractory=0。没有为追求高通过率事后降低门槛或改detector。
- 每候选同种64×64 MLP、400次更新、同训练种子、eta=0.01、同gate规则。
- 4条内部验证已经用于此前开发，不能当作独立最终测试。段落/状态相关，不把它们当作大量独立样本。
- D_cal_delta=0、D_cal_CP=0、D_test=0；未计算CP下界，未运行Test1。

本轮输入统一为完整RSSM、planner carry、当前因果标量signal、ARMED位（579维）。
相比之前B的578维预算函数，多了当前signal，故本轮预算权重、误差不能冒充上一轮同一模型的结果。
goal progress需要过去的goal距离；其当前差分被显式放入输入，不借用未来状态。
decoded向量统一以float64计算范数，本轮clearance阈值与前轮float32阈值有约1e-8的差别，已按全精度冻结，事件时间不因该微差改变。

## 可用信号和语义

从当前wrapper已确认的29维observation布局读取：velocity[0:2]、goal[7:9]、hazards[9:25]。
robot位置本轮未用：已有三类信号足以组成6类候选，不额外人为指定位置区块或坐标方向。
“可读取”不等于解码信号已对真实环境校准，所有结论仅为model-scope。

所有事件均因果地先LOW→ARMED，再于严格更晚时刻HIGH→发一次event并UNARMED。
t=0可ARM但不发event，长时间HIGH不会持续发event。

| family | 标量signal及事件 | 语义限制 |
| --- | --- | --- |
| clearance | 最近hazard解码距离，LOW→HIGH | 反复从较近hazard的位置回到较大clearance；LOW不一定已经违规 |
| motion | decoded速度模长，LOW→HIGH | 低速后再次运动；不是到达goal，也不要求撞过hazard |
| approach | 1/(1+decoded goal距离)，LOW→HIGH | 先远离再接近goal；并不是原任务d<0.3的成功 |
| progress | (1+tanh((d_goal(t-1)-d_goal(t))/0.1))/2，LOW→HIGH | 先退步后恢复正进展；可能受到goal变化或decoded抖动影响 |
| clearance_safe | clearance返回 AND G(no-hazard) | 全部状态0..300距离>=0.2才满足safety conjunct |
| motion_safe | motion返回 AND G(no-hazard) | 运动恢复之外再要求全程解码无hazard |

progress在t=0固定为中性0.5；所有候选低阈值<0.5、高阈值>0.5，因此要求真正由退步转向进展。
安全组合使用同一返回detector，额外safety条件是完整AND，不能覆盖其他gate失败。

## 候选阈值与计数分布

括号内为min/median/max；数值仅显示到4位，CSV/JSON保存全精度。

| 候选 | low/high（signal单位） | train计数 | internal计数 | 完整预览train/internal | 状态 |
| --- | --- | --- | --- | --- | --- |
| clearance Q25/75 | .4736/.7910 | 1/2/4 | 2/2.5/3 | 8/12；2/4 | BORDERLINE（主选） |
| clearance Q10/90 | .3829/1.0272 | 0/1/2 | 0/.5/1 | 1/12；0/4 | REJECT |
| motion Q25/75 | .2315/.7597 | 8/13.5/20 | 8/11.5/13 | 12/12；0/4 | BORDERLINE |
| motion Q10/90 | .1062/1.0250 | 0/3.5/7 | 2/3/6 | 10/12；0/4 | BORDERLINE |
| approach Q25/75 | .3984/.5931 | 0/1.5/3 | 0/1/2 | 6/12；0/4 | BORDERLINE（备选） |
| approach Q10/90 | .3486/.6999 | 0/.5/2 | 0/1/1 | 1/12；0/4 | REJECT |
| progress Q25/75 | .3495/.7461 | 9/18/32 | 11/21/30 | 12/12；1/4 | BORDERLINE |
| progress Q10/90 | .2042/.8706 | 2/8/23 | 7/13/27 | 12/12；1/4 | BORDERLINE |
| clearance_safe Q25/75 | .4736/.7910 | 1/2/4 | 2/2.5/3 | 5/12；2/4 | BORDERLINE |
| clearance_safe Q10/90 | .3829/1.0272 | 0/1/2 | 0/.5/1 | 1/12；0/4 | REJECT |
| motion_safe Q25/75 | .2315/.7597 | 8/13.5/20 | 8/11.5/13 | 9/12；0/4 | BORDERLINE |
| motion_safe Q10/90 | .1062/1.0250 | 0/3.5/7 | 2/3/6 | 8/12；0/4 | BORDERLINE |

### 非退化性：零事件、计数门槛满足比例和方差

下列比例均为train/internal；安全组合返回计数和对应基础family相同，但完整certificate额外受安全条件限制。
“count pass fraction”指路径中N>=2的比例，不是每一步的通过率。

| 基础候选 | 零事件比例 | N>=2比例 | event-count方差 |
| --- | --- | --- | --- |
| clearance Q25/75 | 0% / 0% | 66.7% / 100% | .833 / .250 |
| clearance Q10/90 | 41.7% / 50% | 8.3% / 0% | .389 / .250 |
| motion Q25/75 | 0% / 0% | 100% / 100% | 12.854 / 3.500 |
| motion Q10/90 | 8.3% / 0% | 83.3% / 100% | 3.889 / 2.750 |
| approach Q25/75 | 8.3% / 25% | 50% / 25% | .743 / .500 |
| approach Q10/90 | 50% / 25% | 8.3% / 0% | .410 / .188 |
| progress Q25/75 | 0% / 0% | 100% / 100% | 38.076 / 63.188 |
| progress Q10/90 | 0% / 0% | 100% / 100% | 33.910 / 66.500 |

motion/progress计数不恒定，但M=2裸通过率已经饱和；不能因其variance大就认为它有良好区分度。
clearance主选在全部fit上12/16达标，既非全0也非全满；内部4/4不能证明未来一定满足。

## Certificate compatibility和泛化

等待误差只在已完成的return间隔计算，单位是等效步数（raw g MAE / eta）；右删失尾段不虚构下一次return时间。
回归常数为train完成间隔的平均预算，不使用内部验证拟合。

| 基础候选 | 等待MAE train/internal（步） | internal回归常数MAE（步） | 完整通过率差train−internal |
| --- | --- | --- | --- |
| clearance Q25/75 | 7.82 / 68.33 | 63.97 | +16.7百分点 |
| clearance Q10/90 | 4.83 / 90.09 | 54.00 | +8.3百分点 |
| motion Q25/75 | 4.16 / 17.15 | 14.19 | +100百分点 |
| motion Q10/90 | 6.07 / 75.29 | 52.03 | +83.3百分点 |
| approach Q25/75 | 7.83 / 94.11 | 82.75 | +50百分点 |
| approach Q10/90 | 4.57 / 172.51 | 87.57 | +8.3百分点 |
| progress Q25/75 | 4.32 / 10.59 | 11.23 | +75百分点 |
| progress Q10/90 | 7.04 / 27.36 | 20.86 | +75百分点 |

组合family的回归误差与基础family相同。通过率差另受安全条件影响，全部数值在summary.csv。
只有progress Q25/75的内部MAE略优于回归常数，但事件过密，内部完整通过也低，不能据单一MAE选它。

内部4条各候选预算失败数：clearance Q25/75=2，Q10/90=1；motion两组均4；
approach两组均3；progress两组均3。组合规范对应预算失败数不变，另有1条违反全程安全conjunct。
所有候选的构造drift和event-source检查均无失败。预算不足可同时触发margin、region低预算源检查和soundness，不能相加当独立失败条数。
几何region为经验训练envelope，不是证明过的不变区域；完整明细见spec_failure_breakdown.csv。

## 退化对照

每组使用相同事件、gate和region，比较：学习预算、匹配训练平均分配的常数、H*eta=3大常数、以及10个固定seed的预算打乱。
打乱只在同一split的初始/reset位置间重排预算，保持该split预算分配的边际分布；这是回顾性对照，不是可部署策略。
未利用打乱结果改变学习detector或预算参数。

内部4条完整通过比例：

| 候选 | 学习 | 匹配常数 | 大常数 | 打乱均值 |
| --- | --- | --- | --- | --- |
| clearance Q25/75 | 50% | 0% | 100% | 35% |
| clearance Q10/90 | 0% | 0% | 0% | 0% |
| motion Q25/75 | 0% | 0% | 100% | 0% |
| motion Q10/90 | 0% | 0% | 100% | 25% |
| approach Q25/75 | 0% | 0% | 25% | 0% |
| approach Q10/90 | 0% | 0% | 0% | 0% |
| progress Q25/75 | 25% | 50% | 100% | 35% |
| progress Q10/90 | 25% | 25% | 100% | 27.5% |

主选打乱后的通过数10次分别为2、1、2、2、1、0、1、1、2、2（每次分母4）。
学习预算2/4虽高于打乱均值，多个打乱seed也达到2/4，不能据此宣称可靠判别力。
progress甚至打乱/匹配常数不差于学习函数，这是重要警告。
安全组合的大常数还会被安全conjunct淘汰，完整对照见JSON。

trivial detector另外测试了：

- 每个HIGH状态直接发event：主选内部4条计数为157、21、131、43，而合法LOW→HIGH只有3、2、2、3。
- 每48步机械发event：300步内每条都是6，与绝大部分合法事件分布不同。

主选内部这两个退化detector在M=2上也能给出4/4计数通过，说明只看裸pass会隐藏退化。
HIGH-only完整时间戳与合法事件源不一致（全部候选的逐路径核对数在details中），不能冒充通过event-source gate。
机械事件对照只比较计数，不把它算成合法证书。没有利用trivial detector替换正式定义。

## 为什么这样推荐，而不是按成功率排序

筛选规则先检查事件分布是否有意义，再看预算泛化；随后按事先写入plan的语义优先级
clearance→goal approach→motion→短期progress选择。不是以完整通过率或CP下界排名。
KEEP要求有适量重复事件、低零事件率、内部预算MAE优于简单回归常数且通过率差不超过35个百分点。
本轮没有候选同时满足，所以没有人为强行给一个KEEP。

**主选 clearance_q25_75（BORDERLINE）：**

- 有真正LOW→HIGH机制，全部fit次数1–4，训练8/12、内部4/4至少返回2次。
- 无需相邻goal差分，语义直接：从相对低clearance回到较高clearance。
- M=2完整预览10/16，失败能分解为4条次数不足、2条预算不足。
- 但内部预算MAE不优于简单回归常数，打乱对照也没有给出强证据；因此只是继续研究的最佳候选。

**备选 approach_q25_75（BORDERLINE）：**

- 任务语义清楚、与主选使用不同signal；先距离>=约1.5098，再回到<=约0.6861。
- 不等于d<0.3到达goal，不把向目标靠近冒充任务完成。
- 全部fit中有零次、一次、多次分布，不是一直维持条件。
- 内部仅1/4达到两次返回，预算也差；选择它是语义明确的备选研究对象，不是发证成功率更高的备选。

motion Q10/90可以保留为诊断对照，但预算内部0/4、打乱反而25%，目前不优先。
安全conjunct若是科学问题所必需，不能为了成功率把它移除；本次推荐的是recurrence本身，不声称全程安全。

## 主选冻结配置与停止点

主选文件 `frozen_primary.json` 固定了：

- event：decoded最近hazard距离先<=0.4735981774226446，后>=0.7910375755438274；比较均含等号。
- M_MIN=2，H=300；dwell=1，refractory=0；状态0可ARM但不能return。
- 冻结本轮权重、特征处理和region参数；不在calibration后重训或改detector。
- 预算仅在初始及合法event后重置为g(x)+Delta；每步扣0.01，重置前检查到达边。
- 完整gate保持300条转移都检查，含尾段；主选没有额外全程无hazardconjunct；CV/sliding禁用。
- Delta规则：后续全新cal_delta每条路径取max(0,max段[eta*观察段长−g(start)])，
  alpha=.05，取ceil((n+1)*.95)顺序统计量；所需序号大于n时无有效有限Delta，禁止发证，不能用fit最大值替代。
- 本轮Delta_fit仅有偏预览；正式Delta值仍null。此规则校准的是固定有限路径残差，不自动证明无限recurrence。

冻结表示结果可复现、将来不能看cal结果后再挑，不表示所有接受条件已建立。
如继续优化预算，应视为新开发版本，重新冻结，并只用尚未使用的校准/测试路径做后续评估。
没有创建或启动任何正式校准/测试任务。

## 输出与核验

请求的四个文件均已生成：

- spec_screening_summary.csv
- spec_candidate_details.json
- spec_failure_breakdown.csv
- spec_recommendation_CN.md

另有plan、12组权重、逐路径诊断、frozen_primary和audit/audit_replay。
35项测试通过；独立从保存权重重放2496个路径/对照case，事件、reset、完整gate、打乱seed和推荐结果一致。
旧TD-MPC2/L3及已有pilot文件未修改。
脚本：experiments/safedreamer_recurrence_screening.py；复核：experiments/safedreamer_recurrence_screening_audit.py。
