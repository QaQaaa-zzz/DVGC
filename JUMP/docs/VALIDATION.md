# 当前验证

日期：2026-09-14。新任务：前向行驶下窄障碍完整腾越。旧 JIT 和 STTW_CONTROL 未修改。

**最新更正：找到更早的 `transition_4988928` 单 Actor 自然起点轨迹，包含跳上旧平台、驶离平台后沿、再落回地面继续前进的片段。当前 pi_up_star 也有平台上落地行驶片段。此前仅凭最终 pitch/roll 失败否定此前落地能力的判断过强。以下新增逐步审计区分“存在落地行驶片段”与“达到新任务完整恢复标准”；旧终止标签不改。**

- 基线 JIT action/observation：15 passed。
- 新项目70项行为测试＋继承通知3项：`PYTHONPATH=JUMP/src:JIT/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest JUMP/tests JIT/tests/test_error_notifications.py -q` → **73 passed in 4.41s**。覆盖预算预留/错误计费/启动错误封口、oracle事件链、接触辅助函数、接地资格/触发辅助类及图件。runtime完整FIFO/reset不是这些辅助类测试的独立证明。
- 两个原始checkpoint身份/载入/76D推理及场景编译已通过；上述检查积分0步。
- 首批真实动态资格实验：12个声明情况全部完成、仅2条不同控制轨迹，0接地资格、0完整成功、12任务失败、0未知。实际216控制步/828物理步。没有实际触发，不能据此比较六个delay。
- 单变量近起点追加：只把初始x从1.8m改为2.5m，两个null-trigger情况完成；仍0接地资格。实际54控制步/213物理步。
- 模型检查：nq12/nv11/nu4，总质量5.091kg，payload2kg，原执行器及摩擦保留，仅显式改变障碍几何。

| 情况 | 失败终点 | 原因 | 控制步/物理步 |
|---|---:|---|---:|
| 初始x1.8，pi0，每个触发设置都未到触发门 | .405s | yaw_limit | 每条21/81；六条126/486 |
| 初始x1.8，repair0000，每个触发设置都未到触发门 | .285s | yaw_limit | 每条15/57；六条90/342 |
| 初始x2.5，pi0，null trigger | .675s | roll_limit | 34/135 |
| 初始x2.5，repair0000，null trigger | .390s | yaw_limit | 20/78 |

总实际270控制步/1041物理步；部分终止控制按ceil(substeps/4)计费，未执行子步不称实际积分。PPO和监督优化器均0步。两批分别预留4800和800，余量已释放；历史账本不删除。

首批完整轨迹中pi0在.080s首次轮触地、.240s双轮净空均>2mm；repair0000在.075s首次轮触地、.190s双轮净空均>2mm，signal始终0。最大连续双轮接触观测跨度均仅.005s，未达.1s资格。可描述为触地后无指令腾空，不能把初始化间隙或该未资格运动称为正式liftoff，更不能在本批区分关节动作与反弹的因果贡献。

实际动态轨迹逐项核对76D历史：初始全零，mask依次000→001→011→111；动作/速度/障碍距离/高度来自上一控制端点，未发现FIFO错位。这与原JIT15项基础测试、两个真实checkpoint/reset零步检查是三类独立证据。

CPU每子步后mj_forward使接触与传感值对齐当前状态，旧Warp RK4采样契约不同；尚未完成同后端/采样消融。因此结论限于“两个冻结Actor不能直接通过当前新资格设置”，不是旧策略能力消失、物理不可行或新主线失败。

- [首批全覆盖索引](../runs/qualification/initial_20260914T114808Z/analysis/INDEX.md)：12组PNG/PDF与12条JSONL，原图不覆盖。
- [近起点追加索引](../runs/qualification/near_start_20260914T115254Z/analysis/INDEX.md)：2组PNG/PDF与2条JSONL，包含yaw/wz。
- [可版本管理的紧凑证据与原件哈希](evidence/qualification_initial.json)。完整原始结果留在服务器。
- 两批watcher心跳、终态退出及completion_notified=true已核对，delivery_errors均为空；这是通知服务送达证据，不宣称用户已看到弹窗。
- 独立代码审查无阻断项；启动错误回执缺口已修复并复审通过，旧结果不重跑。

## 2026-09-14 自然起点来源专家核对

用户提出先查找用于Phase U/D的自然起点奖励引导策略，因此先核验来源并比较现有冻结Actor，尚未启动新PPO。

- [冻结专家清单](/home/qy/DVGC/JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json) 指定pi_up_star为`phase_u_v4_pitch15penalty_9977856_seed820901_20260826/checkpoints/transition_9977856`，payload哈希`965b2f141d5cff17f7cef943004fe4e4f62f0fecfe88b47353dc78b14ced723a`。两位专家payload均与清单一致。
- [U原始训练声明](/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826/run_manifest.json) 为fresh、parent_checkpoint=null、起始transition0。[D训练来源](/home/qy/DVGC/JIT/runs/phase_d/phase_d_new_reward_smoke_25600_seed820001_20260827/phase_d_provenance.json) 则明确从该U Actor初始化，critic/optimizer新建。不能将U自身称为更早完整Actor的参数继承者。
- 旧DVGC另有[reference_jump.csv](/home/qy/DVGC/data/reference_jump.csv)，含完整参考状态/动作/奖励分项，[来源声明](/home/qy/DVGC/dvgc/two_phase_guideline.py:571) 将它用于开环guideline。这可能与用户记忆有关，生成它的原始Actor checkpoint尚未定位；当前仓库Git仅显示baseline导入。该轨迹的2ms记录周期与大扭矩列不能直接解释为当前5ms、髋膝±30Nm模型的能力。
- U原8条已用自然起点评估从x1.5/vx2出发，signal约.56s才开启，信号前侧倾最大.279°、偏航.918°；全部约1.04s到达Apex。随后继续U全部最终pitch_limit；该终止汇总不能回答此前是否已落地行驶，详见下方逐步更正。D旧42状态评价32成功/10侧倾失败，其中该U来源14/14成功；分段快照接续不能代替一次在线自然起点U→D新任务。
- [当前pi0旧GPU自然评估](/home/qy/DVGC/JIT/runs/pi_unified_evaluation/pi_0_round1_canonical_natural_current_head_20260904/report.json) 已记录同payload从x1.5出发16控制步/.32s即yaw_limit。当前pi0的自然接近问题不是只在新CPU发生；旧样本已使用，不是新final TEST。

新声明`configs/qualification_source_expert.json`固定同x1.5起点，比较pi_up_star/pi0，各执行null及资格后.3s触发两种设置。最多1600控制步/900s；实际130控制步/514物理步，8.33s墙钟。4情况仅2条不同轨迹，0资格、0触发、0完整成功、4失败、0未知。

| 冻结Actor | 实际终点及原因 | 前进距离 | vx范围m/s | 最大侧倾/偏航 | 每设置控制/物理步 |
|---|---|---:|---|---|---:|
| pi_up_star | 1.000s，approach_not_qualified | 1.841m | 1.648–2.136 | .435°/.865° | 50/200 |
| pi0 | .285s，yaw_limit | .611m | 1.858–2.410 | 6.699°/46.420° | 15/57 |

U的资格阻断来自接触持续性：前/后轮首次实际接地.080/.085s，双轮同时接触最长仅.085–.095s，计时跨度10ms；原判据要求100ms。首次接地后，没有一帧同时满足“双轮无接触且各自净空>2mm”；.130s后前/后轮最大净空仅.698/.704mm。前/后最长无接触段分别6/7帧，按5ms样本覆盖计约30/35ms；离散数据不能确定帧间精确时刻。

每轮在首次接地后的178个完整40ms窗口中均至少接触一次，但这只是诊断，不替代原资格标准。原`approach_not_qualified`仍保留。姿态幅度小也不等于低角速度：末端wy约.958rad/s。末端车前缘距障碍仅24.48mm，不能再任意等待。

实际contactforce全有限，依冻结OR规则重算接触布尔量无不一致；前轮5个零法向力记录因穿透超过容差仍依法计接触。力是每次mj_forward重算后的瞬时求解结果，不是前5ms的平均轮载，接触振荡的物理真实性仍需单独诊断。

- [本批完整4情况PNG/PDF及逐步数据](../runs/qualification/source_expert_20260914T120639Z/analysis/INDEX.md)
- [来源和新运行哈希、逐情况紧凑证据](evidence/source_expert_audit.json)
- runner源码无修改；新声明校验与4情况真实执行通过。原73项CPU测试是上一提交的回归证据，本次未重复训练或扩大其证明范围。完成通知送达记录无错误。
- 当前选择：优先以pi_up_star作为后续接近控制的来源候选，先诊断接触资格，再做在线触发/专家切换。尚未承诺它可完整通过新窄障碍任务。累计400控制步/1555物理步，0PPO/监督优化更新。

## 2026-09-14 更早单策略与完整旧轨迹审计

找到了一个与用户描述吻合的更早候选：[transition_4988928](/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928)。它来自 fresh 奖励训练的中间检查点，累计4,988,928训练transition；同一Actor连续执行自然接近、跳跃和落地行驶，没有U→D切换，也不是空中RSI开始。它与当前JIT同为76D Actor、106D critic、256×3隐藏层、4通道动作及12+12a后轮/关键帧中心绝对关节映射。

原场景是前沿x=3.6m、后沿x=7.6m、高0.16m的长平台。这里观察到的是“跳上平台、随后驶下后沿”的行为，不能作为一次腾空跨过新窄障碍的证据。两份checkpoint的payload/identity均通过载入和有限76D零观测推理检查；这只证明文件完整和接口可用，新增积分0步。

| 连续轨迹的节点 | 更早候选，seed980001 | 当前pi_up_star，seed1000001 |
|---|---|---|
| 自然起点 | x=1.5m，vx=2m/s，非空中RSI | 同左 |
| 首次jump signal / Apex | 0.52 / 1.00s | 0.56 / 1.04s |
| 平台上描述性窗口 | 1.30–3.06s，跨度1.76s | 1.36–4.08s，跨度2.72s |
| 平台后低地面描述性窗口 | 3.40–5.92s，跨度2.52s | 未观测到 |
| 真实最终失败 | 6.22s，roll_limit，x=14.882m | 4.68s，pitch_limit，x=7.725m |

“描述性窗口”是在旧数据上事后计算的最长连续片段：Apex之后、前后轮相对地形几何净空绝对值均≤1cm、|roll|≤5°、|pitch|≤10°、vx≥1m/s；平台分区用root x∈[3.6,7.6]，平台后低地面分区用root x>8.0。持续时间是最后采样时刻减第一采样时刻，采样间隔20ms。它不是原训练成功标签，也不验证实际轮载、无碰撞、全车通过或合法落地点。图中保留完整记录和真正失败终点，不在窗口末尾截断。

更早候选的8条已用旧评估记录全部包含上述两类片段：平台上1.76–1.82s，平台后低地面2.44–2.70s；最终6.20–6.38s全部侧翻。8条都使用相同名义自然初态，不能称为8种初态泛化或8个独立训练种子。当前pi_up_star的8条平台描述性窗口为1.44–3.00s，最终4.62–4.78s全部俯仰失败。

**新恢复标准仍未满足，不能把这些片段写成完整任务成功。** 更早候选seed980001的低地面片段中，vx=2.204–3.258m/s，最大|roll|=4.724°、|pitch|=0.359°，但最大|yaw|=8.640°、|y|=1.132m、|wy|=3.825rad/s。当前U首条平台片段最大|yaw|=20.208°、|wy|=7.005rad/s。小侧倾/俯仰幅度不代表航向、角速度和走廊跟踪合格。

直接将当前冻结恢复数值阈值（roll5°/pitch10°/yaw5°、各角速度0.5rad/s、|y|0.15m、|vx−2|0.4m/s）用于Apex后的原始20ms采样：更早候选8条中最长连续数值达标跨度为0.28s，当前U8条没有达标采样，均不足新协议0.5s保持要求。此项不附加上述1cm净空描述条件，也不重建接触/碰撞/完整事件链；它是旧轨迹数值差距诊断，不是新场景复评。

其他来源搜索已收束：非JIT旧共享策略的证据来自phase-RSI，连续100条旧多专家试验没有恢复成功；`reference_jump.csv`来源只能追到Git baseline导入，生成checkpoint未找到。CSV的0.522s“Recovery”按落地近似锚点和14°侧倾截断，末帧侧倾约−45.1°，不能凭阶段名确认稳定恢复。另外旧`ccd512_guard`候选8条timeout实际伴随平台前沿滞留和大俯仰，不能把timeout当成功。本次没有找到可确认达到新恢复标准的更早单Actor；这不等于整个文件系统不存在用户记忆中的其他策略。

- [16条完整PNG/PDF及逐步CSV索引](../runs/source_search/20260914/INDEX.md)
- [更早候选首条完整轨迹图](../runs/source_search/20260914/figures/earlier_4988928__980001.png)
- [当前U原始完整视频](/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826/evaluations/transition_9977856/representative.mp4)
- [可版本管理的逐条数值、checkpoint检查与工件哈希](evidence/historical_landing_audit.json)
- 分析脚本与完整原始路径在索引同目录；本次核对16份NPZ哈希和16份CSV数据，保留32份PNG/PDF。没有新仿真或优化更新，累计仍为400控制步/1555物理步。

后续将更早候选与当前pi_up_star并列为底层来源候选，在相同新场景/触发/接触协议下再比较；尚未改变冻结策略配置。已有证据支持继续从这些Actor验证和适配完整技能，而不是仅因最终失败就假设需要从头训练跳跃。接触资格诊断与新任务完整恢复仍待执行。
