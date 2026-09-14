# 当前验证

日期：2026-09-14。新任务：前向行驶下窄障碍完整腾越。旧 JIT 和 STTW_CONTROL 未修改。

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
- U原8条已用自然起点评估从x1.5/vx2出发，signal约.56s才开启，信号前侧倾最大.279°、偏航.918°；全部约1.04s到达Apex。随后继续U全部最终pitch_limit，不是完整恢复成功。D旧42状态评价32成功/10侧倾失败，其中该U来源14/14成功；分段快照接续不能代替一次在线自然起点U→D新任务。
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
