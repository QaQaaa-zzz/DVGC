# Jump landing V2：P0 / P1 诊断报告（2026-10-09）

> 最新用户目标：同一模型两边稳定落地，不要求保持前进速度。已找到并配对验证 `transition_01966080`；见[新增双引擎报告](runs/no_speed_dual_landing/REPORT.md)。下文保留此前P0/P1旧协议与未放行门槛，不回写历史结论。

先看 [XY、速度、高度、侧倾、真实平台支撑及逐步奖励对照图](runs/p1_analysis/paired_diagnostics.png)，以及 [PhysX 中间 checkpoint 对照](runs/p1_physx/checkpoint_candidates.png)。图中的2m/s直线仅为几何参考，不代表可行性保证；实际轨迹保留真实终点。

## 状态与结论

P0 配置溯源与诊断基础设施已实现验证；P1 已执行本轮有界对照，但**放行门槛未通过**：环境布局、普通 reset 与 hard reset 的接触后差异尚未完成因果隔离。按[任务书](docs/TASK.md)停止扩大 PPO 预算。E1 已准备且完成离线 learner 测试，正式训练未运行；E2、E3 未运行。没有启动1000万步，没有覆盖历史 docs/evidence。

## 1. P0：配置与身份

[配置审计报告](runs/p0_config/REPORT.md)、[机器审计](runs/p0_config/config_audit.json)、[解析配置](runs/p0_config/resolved_config.json)记录原始来源与身份。独立 `ResolvedTaskConfig` 供 task/observation 共用；重复声明冲突拒绝，增量 knee 模式不再静默按旧映射执行。legacy 回放保持原始语义，未改历史环境。

确切9904128来源含 sim2sim 物理随机化及低速失败包装，后者要求前进速度>0.2m/s并使用失败回报−1000；历史 PhysX 没有完整复刻这两个包装。核心 reward/reset/action/event 配置一致并不等于完整训练环境一致。历史 kd5 warm 的冻结奖励声明与实际加载配置匹配，不能把配置溯源隐患直接说成已证实的历史失败原因。

本次两端均 sim_dt=0.005s、ctrl_dt=0.020s；PhysX 后轮kd=5。MJX复核使用确切9904128 Actor及其源配置，显式关闭训练物理DR后调用自然初态 reset，保留低速终止；因此它是名义物理自然初态开发测试，不是随机训练分布成功率。身份、例外与运行时版本见[声明与配置目录](runs/p1_mjx_9904128_retry02)。

## 2. 同一个9904128是否严格落地

[原始MJX落地判定](runs/p1_mjx_9904128_retry02/landing.json)、[轨迹](runs/p1_mjx_9904128_retry02/trace.npz)、[原生接触](runs/p1_mjx_9904128_retry02/contacts.json)。单个名义自然初态：起跳0.66s，apex0.98s，确认触地1.34s，最长严格保持0.42s；0.5s和1s均不通过。8s timeout，不是侧倾摔倒，也不是完整稳定落地成功。触地后侧向速度和部分角速度/支撑条件违规，见[逐约束统计](runs/p1_analysis/source_landing_constraints.json)。不能由此推断所有初态都失败。

判据来自[独立 evaluator](landing_evaluation.py)：连续真实地面接触后起跳、真实平台顶面轮支撑、非法接触限制、roll<10°、pitch<15°、body omega x/y<1rad/s、世界侧向速度绝对值<0.25m/s、前向速度>0.2m/s，连续保持0.5/1s。body angular rate不是Euler导数；这不是2m/s跟踪门槛。接触无效时禁止认证成功，成功后继续运行。

录制器最初遇到 MuJoCo3.6 host转换形状异常及一次SIGSEGV，失败日志保留。安装版本的 Warp→host 转换明确跳过 contact/efc，不能对该不完整host调用mj_contactForce。修复为每5ms直接读取真实Warp contact/efc solver数组；没有用几何距离冒充接触，没有修改安装包或原训练环境。

## 3. PhysX：不是所有训练 checkpoint 都失败

| 模型 / 动作模式 | 开发结果 | 边界 |
|---|---|---|
| 源9904128 / deterministic |1.52s侧倾终止|重复进程结果一致|
| warm final / deterministic |1.78s侧倾终止|重复进程结果一致|
| warm final / sampled，3个噪声种子 |均8s timeout；严格保持0.38–0.48s|后2–8s平均vx仅0.149–0.178m/s|
| warm 02457600 / deterministic |严格1s通过，最长5.80s|一个开发自然初态；平均vx约0.629m/s|
| warm 04915200 / deterministic |严格1s通过，最长6.26s|一个开发自然初态；平均vx约0.317m/s|

[详细报告](runs/p1_physx/REPORT.md)、[完整因素表](runs/p1_physx/eval_factorial.csv)、[2.46M判定](runs/p1_physx/warm_mid_02457600_certified/landing_lane_0.json)、[4.92M判定](runs/p1_physx/warm_mid_04915200_certified/landing_lane_0.json)。每个目录保留完整轨迹、接触、初态、配置和预算。两个中间候选都先标定接触报告，再hard reset回自然初态；不是 airborne RSI 成功。2.46M后段通过平台后缘落回地面，4.92M至8s仍在平台上。

因此“整轮续训从未具备稳定落地”不成立；**部分中间 checkpoint 好于 final**。尚不能推广为多初态鲁棒性、正式测试成功率或2m/s跟踪成功。未自动替换源初始化模型。

## 4. P1 分歧定位与剩余门槛

- 动作导出：三个模型、普通/极端观测共9组检验，最大差6.184e-6，小于3e-5；见[动作一致性](runs/p1_learner/action_parity.json)。目前没有导出错误证据。
- 动作采样：warm final 随机动作存活、确定性动作摔倒，能解释部分 train/eval 差异；timeout慢速爬行不能认证稳定落地。
- 布局：同一lane、同一origin、逐步完全相同physical target，384→1首次接触80ms出现小的冲量/关节差，100ms root差超过1e-5，1.28s root最大差0.0935m；远端lane达0.206m。[首次接触](runs/p1_physx/layout_first_contact.json)、[重放](runs/p1_physx/layout_replay.json)。支持接触数值敏感性，尚未隔离求解顺序/缓存/布局机制，不能直接断言引擎bug。
- reset：完整第一回合复现后只执行task.reset，显式初始状态完全相同，但source第二回合1.66s、warm1.80s终止；source状态80ms、动作100ms首先分歧，warm状态820ms、动作840ms首先分歧。[reset对照](runs/p1_physx/reset_repeatability.json)。旧的单步前缀错误探针保留，不能冒充完整回合reset验证。
- 原生physics heartbeat在新增的两次修正reset探针中逐子步验证，1352子步均一次event、每control四次；此前探针仅验证四次world.step及接触快照，不倒填heartbeat。逐实例solver迭代属性读回仍是审计缺口。

## 5. Learner离线诊断及E1准备

[learner报告](runs/p1_learner/REPORT.md)；两个真实384×64 batch保存behavior learner、raw action、logprob，离线更新全部丢弃。历史训练未保留等价batch，以下不是历史单轮KL的精确重建。

| checkpoint | A不更新 | B仅normalizer | C仅SGD | D两者 |
|---|---:|---:|---:|---:|
| warm final，KL(old\|\|new) |0|0.00005251|0.01602|0.01679|
| fresh，KL(old\|\|new) |0|216.70|41617.94|85.91|

fresh的统计变化与无门控SGD均能造成巨大分布偏移；只冻结normalizer不足。门控用另一个明确方向KL(new||behavior)，不混用表中数值。真实batch验证fresh候选硬拒绝后参数、完整Adam和normalizer逐位回滚；warm软阈值停止。梯度裁剪后两者范数均0.5，大裁剪前范数不等于裁剪失效。

[E1配置与预算](docs/E1_PREPARED.md)：源9904128 Actor/Critic/normalizer，fresh Adam，lr3e-5、2epochs、clip0.1、entropy0.002、冻结normalizer、KL软0.02/硬0.05；20×384×64=491520训练交互，10/20 block各做开发回放，非自动扩到40。learner_overrides和任务配置分别保存哈希；物理/摩擦/映射/观测不混改。配置launch_ready=false，正式串联训练/评估尚未运行，TensorBoard未启动。E2落地奖励和E3真实前缀后缀/起跳保持均未运行。

## 6. 预算、验证与下一步

PhysX实际53363控制交互、213452物理子步，含49152真实采样交互；[预算](runs/p1_physx/total_budget.json)。MJX成功测试400控制/1600物理子步；首失败已执行4物理步，SIGSEGV尝试确认至少4、声明上限1600、精确值UNKNOWN。因此总已知物理下界215060，上界216656；未知失败不能按零计。所有新采样仅用于诊断，接受的持久训练更新0。

26项配置、落地、接触、learner与E1控制测试通过。原生训练与子进程评估的完整串联未测试；完整逐项奖励重建尚未交付，当前MJX CSV仅总奖励/事件。摩擦、动作映射、输入扩展消融、独立测试集和多初态鲁棒性均未运行。

唯一下一步：保持learner冻结，先隔离PhysX布局与reset的接触后差异，补实际solver参数读回和最小固定动作对照；明确可接受重复性范围后重新判断P1 gate，再按E1→E2→E3推进。此处停止扩大训练符合任务书停止条件，不将准备代码称为实验完成。
