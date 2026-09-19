# JIT 当前状态与证据

## 2026-09-19 100轮完成后的0098配对诊断

完成100轮逐轮审计与0010/0070/0098四时刻各256回合诊断。复用七策略前两模型同seed首批2048回合，仅新增0098共1024回合，计费409600控制步、0训练步；逐元素随机draw及共同存活请求配对通过。
起扰0/5/10/15步：0010成功66/84/239/254，0070为34/93/234/252，0098为43/90/241/249（各分母256）。0098合计623/1024，0070为613/1024；四列配对差值区间均跨零，没有明确全面改善证据。
100次补训采用34个，56个固定起点失败；3653个新增成功中的2557来自未采用候选，不可解释为现策略保留能力。第100轮起步171个训练候选的513个转向及hip请求均为正，表明同起扰时刻内仍有方向覆盖差异，不能称扰动只是变小。
完整报告含逐轮/逐时刻CSV、3072逐回合数据、失败组成、配对区间及四场景全部轨迹PNG/PDF；仅开发诊断，正式TEST未开启。
[全面分析与建议](../runs/monitoring/lineage100_diagnosis_20260919/REPORT.md)。

## 2026-09-18 GPU空闲自动接续100轮

用户授权GPU空闲后自动续训、再次占用时让出并继续等待。修复最终启动检查的资源竞争：wait模式下重新等待，不返回blocked错误。显式wait_until_idle允许资源无限等待；CPU外层与分片监督不因等待超时，实际GPU子任务仍保留计算超时和有限交互预算。
新脚本`cli/resume_pulse_when_idle.py`冻结代码并复用锁定的完成阶段。52项相关CPU测试通过，离线核对第72轮56个pending训练支持与已完成128000步PPO。
监督PID1982304，冻结代码ae098bd；当前GPU占用，执行状态waiting。累计71→100轮，保守继承31919967，总预算115308000。
暂停发生在完成的子任务边界并退出释放GPU；不是即时无损抢占，不使用保留显存的SIGSTOP。单子任务中途外部抢占显存仍有OOM风险；真实错误不隐藏。
[状态、冻结配置及启动证据](../runs/experiments/neighborhood_reward005_safe256_20260916/idle_resume_100/INDEX.md)。

## 2026-09-18 七策略 Phase U 重训与四固定起扰窗口已启动

2026-09-18 17:11 CST 现场核验：用户批准的三臂顺序 Phase U 重训已经启动，当前第一臂 `training_baseline_phase_u`。实验入口：[INDEX](../runs/experiments/seven_policy_phase_u_15m_pulse_20260918/INDEX.md)、[冻结声明](../runs/experiments/seven_policy_phase_u_15m_pulse_20260918/spec.json)、[实时状态](../runs/experiments/seven_policy_phase_u_15m_pulse_20260918/status.json)、[启动核验](../runs/experiments/seven_policy_phase_u_15m_pulse_20260918/launch_verification.json)。结果尚未产生，本条不宣称训练完成或成功率提升。

执行代码为已提交 `876118d94202703a1f2891a693139d23b85a96ff` 的独立 `git archive` 快照：`JIT/runs/code_snapshots/seven_policy_phase_u_15m_pulse_20260918`。来源严格复用已完成 `four_policy_initial_pulse_10k_20260918/spec.json` 的四个方法：repair10 来自 `neighborhood_mixed_rsl_20260916/lineage/round_0010` development manifest；fresh RSI 为 8,000,000 步；repair70 为 `neighborhood_reward005_safe256_20260916/nonfinite_recovery/attempt_0002/lineage/round_0070`；历史 Phase U 为 `phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826` 的 4,988,928 checkpoint。只有前三个非历史 Phase U 来源创建后代，恢复 Actor/normalizer，Critic/optimizer 全新、训练计数从0开始；历史 Phase U 只作固定比较来源，未作为warm start。

三臂按 baseline/fresh_rsi/repair_0070 顺序执行，训练种子 9183101/9183102/9183103；各新增 **14,991,360** 步，共 **44,974,080**。保留指定历史 Phase U 物理、奖励、reset、动作及观测。四个独立起扰条件 `[0]`、`[5]`、`[10]`、`[15]` 使用种子 9183200/9183205/9183210/9183215，各条件七策略×1,000回合，batch256、尾批232；四通道±0.25，持续3控制步，正式25步恢复成功口径不变。最终评价上限 **11,200,000**，训练内固定面板及诊断上限 **96,000**，合计交互上限 **56,270,080**。开发证据，未开启最终TEST/JCE/JEL。

启动验证：311个来源/配置/代码文件锁全部通过；`test_seven_policy_phase_u.py` 与 `test_phase_u_warm_start.py` 共 **27 passed**（CPU回归）。生产CUDA烟测成功加载全部三个initializer，编译一次 Phase U reset，完成三个Actor推理，动作有限；额外训练/物理转移均0，不能据此宣称生产rollout或吞吐已验证。烟测0.5s采样GPU峰值436MiB，期间现有STTW PID1600571保持存活、显存1918MiB。

后台supervisor **1638053**，第一训练子进程 **1638090**，通知watcher **1638054**；核验时JIT子进程GPU846MiB、STTW1918MiB，未等待GPU空闲，也未停止STTW。`JAX_PLATFORMS=cuda,cpu`、预分配false；完整nohup命令与环境见[launch.json](../runs/experiments/seven_policy_phase_u_15m_pulse_20260918/launch.json)。第一臂`backend.json`为gpu/cuda:0，`actor_initialization.json`确认Actor/normalizer恢复和fresh Critic/optimizer。

watcher使用本实验`ACTIVE_RUN.json`与`notifications/`，保留`DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus`、`DISPLAY=:0`；心跳`checked_unix=1789722676.2084248`，无delivery_errors，GNOME通知服务可访问。当前没有错误或整体完成事件，不宣称已显示弹窗；只在顶层整体完成时通知。生产训练与四条件评价由后台继续执行，任何错误保留且不自动重试；最终成功率、实际交互数和报告验收待全部完成后核实。


## 2026-09-18 四策略初始化后三步扰动对比已启动

用户纠正扰动协议：跳跃策略从初始化起全程控制，只在控制步0、1、2叠加四通道±0.25随机动作残差，控制步3以后扰动归零。此前三策略10000回合使用0/5/10/15/20/25混合起扰，其中大量脉冲与离地重叠或发生在离地后，因此旧结果不能回答该问题，保持为独立历史证据。

新对比加入`lineage_repair_0070`，来源为`neighborhood_reward005_safe256_20260916/nonfinite_recovery/attempt_0002/lineage/round_0070`的已冻结晋升策略；连同`lineage_repair_0010`、fresh RSI 800万步和Phase U 4,988,928步，共四策略各10000回合。零训练，40个批次，扰动回合上限16000000步；0070新增1个无扰动回合400步，总上限16000400。2回合/策略GPU冒烟已完成3600步，四者pulse mask均为[0,1,2]且后续请求扰动非零数为0。

正式运行已从冻结提交`7ae4a10`启动，监督PID1075334、桌面监视PID1075335。0070名义回合已完成；首个正式批次四策略各256回合并通过时序、配对、分母和成本校验，成功数暂为56/56/242/34，仅是1/40批中间值。运行已进入第2批，实时入口为`runs/experiments/four_policy_initial_pulse_10k_20260918/`。

## 2026-09-18 邻域谱系OOM恢复：用户改为累计100轮

原谱系完成71轮，第72轮接续评估的第二个256候选分片在恢复状态、创建CUDA图时OOM。
新恢复保留完整采样及第一个已完成分片，校验配置、候选身份、RNG lane与SHA锁；复用不增加计费，支持重复恢复。
增加每子进程GPU空闲与至少20000MiB空闲显存门控；外循环显式CPU，仿真仍cuda,cpu。
奖励、网络、1024候选和256接续分组保持。73项相关CPU测试通过；代码27f46ef。
用户授权累计100轮，剩余29轮；历史保守计费31622252、总上限115308000。
监督PID1040123已启动，初始因其他计算进程占约18GB进入资源等待，释放后自动继续。
原OOM的第二分片已完成（新增57856计费交互），进入第三分片；第一分片复用66304历史交互。
这验证了原失败位置本次通过，不是已完成100轮或保证永不OOM。
桌面监视PID63326跟随更新后的ACTIVE_RUN，心跳与通知状态已核查。
TensorBoard6009已切换到本次数据，HTTP200；启动通知返回0。
[实时状态、声明及证据](../runs/experiments/neighborhood_reward005_safe256_20260916/oom_recovery_100/INDEX.md)。
启动门控不能阻止其他程序稍后抢占GPU，不作绝无OOM保证；历史失败保持不变。

## 2026-09-18 三策略10000回合假设性重标注（零仿真）

按用户指定的替代口径，仅重标注`lineage_repair_0010`在世界终点x=4.0–4.5m且已有效接触的正式失败回合：保持1–12步的102回合全部计成功，保持0步的730回合以固定seed哈希选取365回合计成功。基线由7,364/10,000（73.64%）变为假设性的7,831/10,000（78.31%）；fresh RSI仍为77.46%，Phase U仍为76.89%。新图不画失败红叉，x显示截止5.5m；完整重标注CSV、规则、来源哈希及PNG/PDF/SVG均已保存。该派生产物没有新增仿真，不修改正式25步稳定恢复判据或正式统计。

[假设性重标注图与数据](../runs/experiments/three_policy_pulse_10k_20260918/comparison_counterfactual_partial12_halfzero_v2/INDEX.md)。

## 2026-09-18 三策略10000回合图件可读性修订（零仿真）

完成的30000回合原始数据不变。原图因万条轨迹叠加且少数前进至约23m，主跳跃区域被压缩。
新增按策略分栏的回合等权二维轨迹密度图，世界坐标和离地点相对坐标显示均截止x=5.5m；
越界轨迹只裁剪显示，完整计入成功率、失败原因和峰值统计。另列成功率95% Wilson区间、
统一分母失败组成和峰值高度10/25/50/75/90分位，保存PNG/PDF/SVG、绘图NPZ和显示契约。
越过世界x=5.5m：基线1311、RSI196、Phase U215回合；结果仍为7364/7746/7689成功。
[推荐图与数据入口](../runs/experiments/three_policy_pulse_10k_20260918/INDEX.md)。

## 2026-09-18 三策略各10000回合：新配对扰动实验已启动

用户要求扩大到每策略10000回合。固定lineage_repair_0010、fresh RSI transition_8000000、
原Phase U transition_4988928，共30000新回合，零训练，最大12000000控制步。
256/批，40批末批16；seed9182702..9182741，各批三者同seed，起扰分配按全局episode偏移。
沿用3步四通道±0.25固定随机脉冲、完整固定起点与stable_forward_recovery判据。
原100回合保持独立；已有名义轨迹复用，每批三者验证随机draw与共同存活请求后汇总。

[实时状态与产物入口](../runs/experiments/three_policy_pulse_10k_20260918/INDEX.md)。
冻结代码b596a6b，监督712237、桌面监视712238；首批三者各256回合已通过GPU与配对核验，
正在执行后续批次，监视器心跳正常、启动通知返回0。实时结果以status.json为准，尚未宣称完成。
22项CPU测试通过，包括预算精确覆盖、尾批/起扰分配、配对拒绝与完整报告回归；
逐批NPZ、JSON回执和最后30000行CSV及全部轨迹PNG/PDF/SVG自动保存。
报告轨迹层栅格化降低文件体积，坐标文字仍可编辑，未用抽样替代全部轨迹。
子任务错误或预算/配对不符即停，无自动重试、不自动扩大回合，不停止其他任务。

## 2026-09-18 三策略100回合配对扰动已完成

800万步RSI训练已完成（8000000训练+386末TRAIN检查）；原Phase U transition_4988928
新增100随机扰动+1名义回合，与同一批lineage_repair_0010和RSI轨迹合图。
基线76/100、RSI72/100、Phase U66/100；三者无扰动均成功，根部峰高分别0.534/0.529/0.677m。
原Phase U更高不等于稳定恢复更好；RSI在本次开发样本未超过lineage_repair_0010。
全部300回合收到非零扰动且检出诊断离地，全部失败保留。随机draw及共同存活请求逐元素配对。

新增预算/计费40400（有效6822），复用80800，零训练；评估冻结代码196c27e。
两子进程退出0，桌面完成通知交付成功；原载荷Actor/critic/normalizer不变，独立保留Phase U来源，
不将共享运行模板当作其训练身份。报告CSV区分零幅值名义窗口与实际非零扰动，零仿真重建comparison_verified。
44项相关CPU检查通过（16扩展/脉冲与28训练回归），已核验PNG图面、输入和冻结代码哈希。
[三策略图与数据](../runs/experiments/three_policy_pulse_comparison_20260918/INDEX.md)。

## 2026-09-18 指定 Phase U 奖励的全新 RSI 800万步

用户指定 `phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928` 为奖励来源。
实际恢复检查通过：训练计数4988928，配置哈希 `baae0efb9be3d35daad6ecfd0a6e36afdbf3aecc9d2f45f4699e0eca6997284d`。
其冻结 resolved_config 已是前次998400步RSI的上游奖励来源，因此没有更换奖励数值。
仍以 original_all_phases 将原Phase U公式用于完整RSI回合，采用stable_forward_recovery终点；
这不同于原模型仅训练上升阶段的任务范围，不宣称重现其整个训练实验。

新Actor/Critic/优化器/归一化从零初始化，不加载指定模型或前次RSI参数。
同一1368原始完整快照，20%固定起点、80%RSI，保留原PPO配置及种子9182601。
预算8000000步（2500整批），末TRAIN检查最多1600步，后续基线lineage_repair_0010和新策略
各1无扰动+100配对随机脉冲回合最多80800步，总上限8082400；不自动续训。

独立目录 [实验索引](../runs/experiments/fresh_rsi_phase_u_reward_8m_20260918/INDEX.md)，
含奖励来源审计、完整参数、零交互1368状态预检及输入哈希锁。
冻结代码0ccb2ee，监督PID457072、训练PID457447、监视PID457073；已确认实际采样至少240000步，结果尚未完成。
实时进度以该目录status.json和training/fresh_rsi/status.json为准。前次负结果和产物保留。

## 2026-09-18 全新 RSI 998400步与初始策略对照：负结果

用户授权从对齐图对应的原始完整快照进行全新训练，并最终指定基线
`lineage_repair_0010`。训练Actor/Critic/优化器/归一化从零开始、原奖励不改、
20%固定起点+80%RSI；1368快照（上升1139、下降229），不平移物理状态。
100万步上限内完成998400整批步数，末checkpoint冻结及恢复通过。

两策略各1次无扰动+100次配对随机动作残差评估：基线无扰动成功、随机78/100；
新策略无扰动约0.06s触发stuck，随机0/100，全部stuck，其中84回合起扰前已失败。
这是固定起点能力不足的负结果，不是已证明的扰动恢复能力，更不是RSI一般无效。
基线其余为侧倾限制16、偏航限制5、超时1；完整世界坐标和离地点对齐图保留失败。
全部请求随机draw及共同存活扰动期请求值一致，原始实际残差和逐回合数据保留。

记录成本1079367（训练998400+末检查167+含填充评估80800），无自动续训。
训练和所有评估正常退出；原报告索引字段重名错误保留，从同一数据零仿真重建
comparison_verified并补齐超时/终止码分类。40项相关测试通过。
[实验索引](../runs/experiments/fresh_rsi_comparison_20260918/INDEX.md) ·
[设计及完整结论](RSI_FROM_SCRATCH_COMPARISON.md)。

## 2026-09-18 离地点对齐派生图（零新增仿真）

新增可复用 `cli/plot_liftoff_aligned.py` 与共享 `analysis/liftoff_alignment.py`。
从 neighborhood_reward005_safe256 的 nonfinite_recovery/attempt_0002 谱系读取时已完成
57轮、53762条成功候选，每轮均匀抽取24条，1368条全部检出诊断离地点。
产物：[离地点对齐图与索引](../runs/analysis/liftoff_aligned_20260918/INDEX.md)，
含世界坐标/对齐双视图、完整恢复PNG/PDF/SVG、101992个实际坐标点及来源哈希。
离地位置诊断范围2.72847–2.86841m；这是扰动成功见证抽样，不是此前25条名义轨迹统计。

诊断须先观察接地，再双轮间隙连续3个控制帧>1cm，取首帧；不是正式子步离地事件。
只平移x，保留真实z和前缀/接续分段；对齐结果不能用于世界坐标覆盖计数。
4项行为测试通过；实际1368条对齐零点、全部坐标有限性、导出索引与6张图件已核验。
无新增训练/评价交互，原始实验及正在运行的流程不改；复用命令和约定已写入JIT/AGENTS.md。

## 2026-09-18 非有限日志诊断与第57轮接续

新奖励实验已完成56轮，第57轮策略补训在最后一块回报日志中报
`episode/reward/episode_return_override`非有限值；最后完成计数124800。
原配置、种子、原冻结代码的一次有界重试完成128000训练步+157 TRAIN检查交互，
checkpoint恢复验证通过；未复现原故障，因此**NaN根因未确认/未宣称修复**。
不删除非有限检查，不将NaN置零，不改变奖励/物理/成功标准。

代码新增非有限回调完整指标和步数记录，仍抛错；修复多次恢复的历史outcomes及
baseline追溯，并允许保留失败子任务最大预留成本接续，不能把预留冒充实测。
恢复输入已逐项确认当前邻域map、支持集与源策略身份一致。
第57轮采样/接续评估及检查通过的重试模型复用；原失败记录不覆盖。
继承计费26411232（含原失败129600预留和重试128157实测），总目标仍150轮，
总上限172881600。新入口`runs/experiments/neighborhood_reward005_safe256_20260916/nonfinite_recovery/`；
实际是否运行以根目录ACTIVE_RUN.json与该入口状态为准。
有界重试证据`runs/monitoring/nonfinite_reward_20260918/`。
131项相关测试通过；首次启动路径身份预检退出且新增交互0。链接解析修复后，
`nonfinite_recovery/attempt_0002`已实际进入第57轮after_learning评估。
TensorBoard6009已切换，根ACTIVE_RUN指向本次接续。此为启动验证快照。

## 2026-09-16 策略名称冲突修复与原预算接续

新奖励实验已完成10轮，第11轮128000步补训后因策略名冲突停止：
继承的初始策略lineage_repair_0010与第11轮拟注册的新策略同名。
修复为根据现有策略库分配首个未占用名称，不重命名旧策略、不改变参数；
训练run_id保持原样以复用已完成训练。code_commit明确为代码来源元数据，
新旧代码仍分别哈希锁定；科学输入比较继续校验奖励、状态、策略等身份。

91项针对性测试通过。真实恢复预检确认第11轮支持集一致、源策略一致、
128000步训练已完成。接续入口：
`runs/experiments/neighborhood_reward005_safe256_20260916/name_recovery/`。
继承6960910计费交互，复用第11轮采样/评估/补训，总目标仍150轮。
原错误与冻结记录保持不变；实时状态以实验根目录ACTIVE_RUN.json为准。
恢复进程已越过故障点，完成第11轮并进入第12轮采样。TensorBoard 6009
已切到续训指标，通知监视器心跳正常、无发送错误。此处为恢复验证快照。


## 2026-09-16 OOM修复、新奖励与150轮重启

旧邻域实验完成11轮，第12轮接续评估于16:07:32被内核全局OOM杀死，
PID1176471约12GiB匿名RSS；旧数据保持原样。用户授权修复、加强扰动期
物理失败惩罚、降低新颖性及进一步优化。新配置新颖性.05，成功+1，
普通补训失败−1，已实际施加扰动后明确物理失败−5。未知/冲突不误罚。

每轮仍1024候选、3步扰动、六起扰时刻混合。大接续评估以256个候选
为一组、串行独立进程执行并释放内存。94项CPU测试通过；真实1024
验证完整结束，峰值子进程RSS4.981GiB，339.29s、335872计费步数。
与旧轨迹相比42/1024标签不同，成功832→826；明确是新的数值执行布局，
不宣称严格等价或整体提速。旧物理模型、完整状态恢复及成功标准不变。

新实验：`runs/experiments/neighborhood_reward005_safe256_20260916/`。
从已接受lineage_repair_0010出发，探索器重新初始化，150轮上限，
最大新预算172881600；验证359600步另计。已从冻结提交faeab68启动：监督器1302222，
通知监视器1302223。初始名义轨迹与种子评估完成，进入第1轮采样。
TensorBoard：http://localhost:6009；实时状态见该目录ACTIVE_RUN.json。
通知标题区分工程验证与正式150轮训练。
详见[PULSE_MEMORY_REWARD_RESTART](PULSE_MEMORY_REWARD_RESTART.md)。


## 2026-09-16 邻域条件探索器：150轮新实验

用户授权修改、验证后立即启动150轮。新增每批冻结的同源策略邻域图，
每次有效扰动前查询并保存386维观测；16个近邻分别编码后池化为64维，
原128×3主干保留。每批1024个候选混合六个起扰时刻，3步扰动、幅度0.25、
原奖励与0.5s稳定恢复标准不变。采用停训时已接受的lineage_repair_0102
作为基础策略，探索器重新初始化，未导入旧探索结果作为初始邻域。

84项针对性CPU测试通过。24候选GPU完整单轮验证完成，包括128000步补训、
重评、16/16保留检查、探索器更新及新源策略种子；141099计费交互，459.82秒。
所有工程验证合计148475步，与正式实验分开记录。融合恢复原型未通过后缀
一致性检查，生产保留原恢复路径；不宣称整体提速或科学性能提升。

新实验入口：`runs/experiments/neighborhood_mixed_rsl_20260916/`；
150轮最大新交互预算172881600，按实际使用记账，不自动续训。
已从冻结代码1af91cb启动，实际GPU子进程进入初始邻域种子生成阶段。
监督器869897、通知监视器869898；心跳正常，无通知投递错误。
TensorBoard为http://localhost:6008；首轮更新后写入reward/successes等曲线。
实时进度以该目录ACTIVE_RUN.json及状态文件为准。
实现与证据说明：[NEIGHBORHOOD_EXPLORER](NEIGHBORHOOD_EXPLORER.md)。



## 2026-09-15 RSL sampler precision repair;18 rounds complete

Original-reward arm completed17 rounds and all round18 acquisition, repair-policy
training and evaluation; explorer update stopped at logprob error.0021715 against
Torch, exceeding.002 guard. Exact saved-state audit reproduced GPU default-matmul
behavior within2.4e-6 while GPU highest precision agreed with Torch. This is a
reduced-precision sampler/optimizer arithmetic mismatch, not changed checkpoint data.

Sampler now explicitly uses highest matmul precision and records it. Updates replay
the saved collection arithmetic, retain actual saved behavior log probabilities,
require replay error<=.002 and max per-state distribution KL toTorch<=1e-5, and use
replayed behavior means/scales for RSL KL. No blanket tolerance increase or replacement
of old likelihoods. Legacy round18 replay error3.8e-6, maximum distribution KL2.4e-7;
completed3072 valid actions/12 updates, lifetime708 updates. Its post-update KL.03176
triggered the existing epoch stop; no strict KL bound or rollback claim.

New `runs/experiments/rsl_reward_comparison_20260915/precision_repair/` resumes
round19 with frozen source lineage, explorer+Adam continuation and5,821,825 inherited
interactions. Source promotion in round18 had been rejected (fixed start failed),
so source remains lineage_repair_0013. All policy training, evaluation and prior
17 explorer updates reused; only the previously failed update was completed in a
new derived view. Source records remain immutable.26 related CPU tests pass;
real GPU collection for round19 completed8192 new interactions and evaluation began.
Supervisor2043281; watcher1494267 healthy with no delivery errors. Active manifest
and INDEX updated. Both200-round arms,1024 envs,delta.25 and initial/maxLR.001 unchanged.


## 2026-09-15 RSL amp25 bootstrap reference repair

The first1024-candidate acquisition and current-pi evaluation completed. Repair
training failed before optimizer/environment creation: nested iterative config was
passed as the immutable bootstrap, invoking an unrelated historical formal-seed
guard (`formal training seed must equal820901`). Fixed generic bootstrap ancestry
resolution with cycle detection; generated successor configs reference the canonical
bootstrap and lock its file. The requested repair/explorer seeds and physics stay
unchanged; no seed validator was weakened.

49 related CPU tests pass, including nested bootstrap/cycles and reuse across
relocated locked source snapshots. Updated an obsolete test's empty frozen manifest
to the required source-policy reference. New attempt `runs/experiments/rsl_reward_comparison_20260915/amp25_repair/` reuses103716 actual
acquisition/evaluation steps. `reuse_audit.json` preserves original failure hashes;
the failed129600 training reservation is zero actual because execution failed during
bootstrap resolution before creating any training output. Original failed records
remain untouched. Both200-round arms retain0.25 limits,1024 envs,128×3 ELU,RSL3.2,
LR/ceiling.001 and continuous explorer state. Supervisor1540416,watcher1494267 follows
ACTIVE_RUN; live status is authoritative. No completed-training or efficacy claim.


## 2026-09-15 RSL continuous explorer: two 200-round reward arms

User corrected explorer LR to **1e-3**, also the adaptive ceiling. Optional RSL-RL3.2.0
uses official PPO.update with106→128×3 ELU actor/critic and state-dependent log std.
JAX GPU sampler mirrors Torch weights; environment tanh bounds the residual.
Only eligible pulse actions receive gamma=lambda=1 Monte Carlo outcome targets;
no suffix-action training or stale replay. Entropy is latent Gaussian entropy.
8 epochs maximum, minibatch512, adaptive KL target.01, epoch stop.03 (no rollback),
value clipping, entropy.001,valuecoef.5,gradclip1. Explorer weights, Adam, LR,
Torch/JAX RNG and its frozen initial normalization persist across base-pi changes.
The new pi gets its own novelty ledger; it does not get a reinitialized explorer.

Authorized original_all_phases/phase_recovery arms,200 rounds each,1024 envs×3 ticks
(.06s), amplitude.25 on all four channels (latest user correction), onsets0/5/10/15/20/25. Only repair-policy reward differs.
Current-pi-only outer loop retains pending repair128k, delayed evaluation and
retention/nominal promotion checks; recovery window0.5s. No old pi1–pi6 helpers or
final TEST. Plan ceiling455,769,920 interactions plus400 measured setup ticks;
reserved maxima are not measured cost. Two setup errors retained: missing relative
Tube data path (zero simulation), old inventory keys (400 ticks; no production update).

Entry: `runs/experiments/rsl_reward_comparison_20260915/INDEX.md`; current attempt
`amp25/`, supervisor1511765, watcher1494267 follows ACTIVE_RUN. The earlier
.10/.15 attempt was stopped by user correction after11366 measured interactions;
its interrupted suffix has a409600 reservation and unknown actual cost, kept
separately in amp25/superseded_run.json. No production PPO updates were completed.
Frozen source
snapshot protects the run from future workspace edits. Read live status for progress.
30 CPU tests passed: cross-framework output, short/unknown masks, serialization,
continuous state, LR ceiling, stale likelihood rejection and inventory. Real GPU
observations match Torch logprob within4.8e-7, value2.8e-8; one synthetic-feedback
engineering update passed. These checks do not prove exploration efficacy.
Each round retains tapes, checkpoints, hyperparameters, official epoch losses,
step LR/gradient norms, KL, component rewards, inventory and costs. TensorBoard
lives in each lineage/tensorboard; standard CSV+PNG/PDF/SVG process exports remain.

## 2026-09-15 昨天冻结探索器对照与倾向分析完成

`frozen_yesterday_explorers_20260915/restart`三组全部完成：原奖励/恢复奖励谱系探索器与均匀随机成功555/545/535（各768），成功到达格439/430/421，实际控制步81280/80128/83328。没有训练更新。首次空事件列表错误已修正，完成弹窗已有发送记录。必须保留配对局限：初始观测一致，后期起扰前独立复演分叉，最大根z差约0.02684m；不是严格同起扰状态配对，不能将小幅成功差归因于探索器。

追加同一π0标准轨迹76观测、每状态8192次CPU输出抽样及初始化对照，分近地/上升/顶点/下降/落地后，保存requested/effective均值、std、方差、RMS、符号比例、裁剪率及实际脉冲的轨迹间统计。空中steer小方差/hip大方差主形状已存在于未训练初始化；恢复奖励谱系knee有弱正偏置和方差增加，尚无阶段分工的因果证据。历史参数谱系发现：原奖励最终探索器自round56重置后仅4轮、1536有效样本，恢复奖励自round44后16轮、6144样本；不能把campaign60轮当探索器连续训练60轮。完整报告及所有图/CSV/重绘代码：`runs/experiments/frozen_yesterday_explorers_20260915/restart/tendencies/REVIEW.md`。下一阶段需管理探索器跨π连续学习与归一化，并采用共用完整起扰快照及未训练网络参照；本次未改训练行为、未开最终TEST。

## 2026-09-15 昨天冻结探索器独立对照已启动

用户要求用昨天训练好的探索器与随机比较。选取`reward_comparison_pulse_20260914/recovery`中原奖励/恢复奖励两条谱系第60轮`round_0059/update/state.msgpack`，固定参数，不执行update或补训。统一原始π0、0.10幅度、3控制步、0/5/10/15/20/25步起扰，每条件128候选，共三组2304候选，上限957312环境控制步；未用最终TEST。旧探索器曾与不同后继策略、2s标准共同迭代，本次统一π0及已批准0.5s标准，属于冻结探索器迁移回π0的开发评价。相同评价RNG、同一初始格子账本；不把序列化RNG替换称为网络更新。

初次启动因固定时刻模式传入空事件列表，在仿真之前退出，原错误保留。修正为省略事件列表后重启，首批collect与evaluate已通过。当前入口`runs/experiments/frozen_yesterday_explorers_20260915/restart/execution/status.json`，自动结果在同级`../analysis/INDEX.md`。弹窗监视器PID1339817跟随更新的ACTIVE_RUN，初次错误已通知。源码复用冻结9ac50f1，不修改正在运行的源文件；原始checkpoint未修改。结果以状态及自动报告为准。

## 2026-09-15 论文第一阶段完成：学习探索尚无覆盖优势

`runs/experiments/exploration_paper_20260915/execution/status.json`于北京时间15:03正常完成。工程两臂各8轮×4，机制两臂各24轮×128；固定同一原始Actor，不补训π0、不用后继策略或最终TEST。机制学习/随机成功2126/3072（69.21%）与2078/3072（67.64%）；成功到达物理格1386/1385，扣各自初始支持后1385/1384。初始支持格38/37，后续严配应共用冻结支持。新增到达格2276/2302；当前不支持学习探索显著扩大tube。

学习组下降事件64.71% vs随机56.38%，离地47.53% vs50.65%；均为训练中自适应采集、单种子描述结果。学习器有效样本9216、288次更新，post-update KL 0.00257–0.01470；24轮先达到轮数上限，未用满250万/臂。机制实际控制步337188/347736，耗时1596/1358秒；总流程耗时含长时间等待。成功轨迹root z峰值中位数均约0.651m，未证实跳高优势。结果、逐阶段统计、PNG/PDF/SVG、逐轮CSV、哈希及口径核对见`runs/experiments/exploration_paper_20260915/analysis/mechanism_review/INDEX.md`。此次为零新增交互的结果分析，完成弹窗已有发送记录。后续补训、反馈消融和最终独立测试尚未执行。

## 2026-09-15 投稿方法总体框图

用户要求按完整方法设计生成科研框图，不区分实验进度或呈现结果。新增`docs/paper/prompts/method_overview.md`及修订Prompt，包含完整数据流、网络/动作细节与中英图注；图件保存在`runs/paper_figures/method_overview_20260915/`。图中强调短时残差、完整状态续接、新旧混合重置补训、同状态重评、保留检查与延迟探索反馈；tube档案和最终单策略分别表达。生成版用于风格参考，配套标准绘图的SVG/PDF用于精确的数据流；概念曲线不是实测轨迹。本次仅制作论文图件，无新训练或仿真交互，既有实验调度不变。

## 2026-09-15 已批准论文实验，实施第一阶段

用户已批准[实验计划](EXPLORATION_PAPER_EXPERIMENTS.md)，总预算上限1.2亿控制步，分阶段执行。第一阶段为固定π0的学习探索/均匀随机对照，各24轮×128候选、四个事件×0.10/0.15幅度、每臂250万实际交互上限；此前先跑4环境的小规模两臂工程验证，总上限25万。后续主训练、质量反馈消融和最终TEST尚未启动，旧队列保持暂停。

已实现：冻结配置允许正整数连续恢复窗口（0.5s=25步）；initialization_only封装保持原Actor/normalizer/critic，零新训练且不伪造formal_report；pulse随机分支、每lane事件触发、动作前后相位与真实接触/几何字段、当前π质量反馈、实际预算停止及零成功正常报告。新初始Actor哈希仍为a06acbc81aead7120a291cf82359189bff36ccfecde132615ae0f264ee3126e8。87项相关CPU测试通过；GPU验证与机制训练状态以`runs/experiments/exploration_paper_20260915/`运行记录为准，不把准备或等待称为已训练。

新运行使用冻结源码快照，避免之后代码迭代破坏运行锁；GPU子任务通过设备compute进程空闲门控，CPU调度器可先启动。未改动物理参数，不读取其他项目源代码/结果。主训练的更强KL拒绝、固定π0开发锚点和最终配对测试器仍是下一阶段待实现项。

启动快照：调度器PID435644与弹窗监视器PID435645已启动并核验存活，监视器已有心跳，启动通知发送返回0。入口为`runs/experiments/exploration_paper_20260915/execution/status.json`；工程两臂正常完成后自动进入机制对照。此次检查时，首个GPU子任务`engineering/learned/lineage/seed_support_execution/status.json`为waiting，设备已有其他compute进程，JIT新增交互为0；这是已排队，尚未开始仿真或训练。源码快照为9ac50f1，本阶段总上限5212800控制步，最新状态以运行文件为准。

## 2026-09-15 两组几何与论文主线复核

新CPU分析见`runs/experiments/reward_comparison_pulse_20260914/analysis/geometry_review/INDEX.md`，无新训练。成功轨迹根部z峰值中位数原奖励0.596m/恢复奖励0.612m，最大0.686/0.705m；初始策略成功样本中位数约0.655m，不能声称后继跳高增加。逐策略样本是自适应筛选而非配对。suffix字段valid_contact由endpoint_success写入，稳定协议下不是首次接地，禁止用它推算跳远。动作晚偏移转向平方幅值占比下降仅是线索，不是已验证阶段分工。待做学习探索vs随机探索等预算对照、单最终策略独立扰动评价；旧结果仍为2s，0.5s尚未运行。

## 2026-09-15 后续实验采用连续稳定0.5秒

用户将落地成功窗口从2s改为0.5s。后续使用`configs/descent_stable_forward_0p5s.json`：来自现有strict_down，25个完整控制步（dt=0.02s），接触帧不计时，不稳定归零；保留vx>=0.5m/s、roll<=10°、pitch<=15°、接地及无物理失败要求。取消额外1m进度门槛，稳定满0.5s即成功并终止。奖励系数不改，恢复tick奖励仍逐稳定步给出，成功bonus随新终点触发。原奖励分支仍沿用原奖励，不引入恢复bonus。

旧2s配置/运行/图和标签保持冻结，不能原地恢复旧队列冒充0.5s；下一轮声明须在采集、补训、评价共用此down_config并重新锁定哈希和初始支持。此次仅配置与CPU判据回归，不启动训练、不重标旧结果。绘图说明已去掉硬编码2秒。

## 2026-09-15 x-z图按用户给定样式调整

常用绘图脚本改用中文标题、逐策略颜色/图例、成功候选散点，首次记录到稳定恢复success后截断。旧轨迹已止于该终点，长距离是恢复期间前进而非额外成功后运行。主图默认放大3.3m跳跃窗口，并明确标注局部放大；`tube_xz_full`保留完整2s稳定恢复距离，未改变终止或成功判据。新图入口：`runs/experiments/reward_comparison_pulse_20260914/analysis/tube_xz_policy_style/INDEX.md`。无新仿真/训练，旧图保留。

## 2026-09-15 常用x-z包线投影脚本

`cli/plot_pulse_tube_xz.py`支持多lineage、恢复祖先追溯、统一轴范围和证据校验。每组pulse循环完成时自动输出`lineage/analysis/tube_xz/`，无仿真交互。两组0.10奖励对照分别绘制7253/7472个成功上下文策略见证的前缀与续接轨迹，保留恢复段、不连虚假接缝、不填凸包；候选点区分成功/未解决/未知。图和NPZ/CSV/JSON证据位于`runs/experiments/reward_comparison_pulse_20260914/analysis/tube_xz/INDEX.md`。17项相关测试通过。0.15两组仍停止。用法见[PULSE_TUBE_PLOTTING](PULSE_TUBE_PLOTTING.md)。

## 2026-09-15 已完成0.10两组分析；0.15两组暂停

原奖励/恢复奖励各60轮×128候选已完成；最终候选成功7253/7680与7472/7680，同状态补训转换384/811与155/363，总实际交互8614821与5782254。恢复奖励更省成本，原奖励发现更多需补训才成功的候选；两组不是固定状态配对测试，单种子不宣称reward因果优越性。补训KL峰值3.062与0.12265，自适应LR不是硬限。详细可重绘证据：`runs/experiments/reward_comparison_pulse_20260914/analysis/amp10_comparison/INDEX.md`。用户要求后两组先不启动：原奖励0.15已因非有限reward指标失败，恢复0.15未运行，队列保持停止；不恢复任何旧暂停计划。

## 2026-09-15 奖励对照源策略晋升失败恢复

首组完成7轮后，round0006候选通过接手保留面板，但完整nominal轨迹未完成稳定恢复。此前将这种研究负结果抛为异常，导致整个队列退出。现仅在候选晋升阶段记录support_ready=false并拒绝晋升，保留旧source和已更新探索器；初始source不合格仍报错。已增加行为回归，15项相关测试通过。

接续入口 `runs/experiments/reward_comparison_pulse_20260914/recovery/execution/status.json`。首组从round0007继续，冻结导入7轮已完成证据、策略库、支持池和探索器optimizer/RNG，保持source=lineage_repair_0002。旧失败子阶段只采集400步，未执行suffix；校正后的继承实际交互1286651，旧保守预算错误记录不改。其余三组配置不变。接手续跑状态以当前文件为准。

## 2026-09-14 已授权单种子奖励对照

用户新授权：原Phase U奖励全阶段 vs 分阶段恢复奖励，分别幅度0.10/0.15，共4组，每组60轮、128候选/轮、单种子9991421；旧多种子计划继续暂停。原始Actor相同，稳定2s成功规则相同，当前策略单独探索/验证，失败候选触发128k补训。两组均使用Brax原生ADAPTIVE_KL，target0.01，初始/上限lr1e-4，下限1e-5；KL>0.02时lr/1.5，0<KL<0.005时lr*1.5。这不是硬拒绝更新或相对初始Actor的累计KL保证。原奖励组沿用Phase U reward函数到下降阶段，不加入恢复bonus；终止/成功判定不变。loss、真实分项reward、KL及lr随训练记录。

入口：`runs/experiments/reward_comparison_pulse_20260914/execution/status.json`；4组自适应轨迹将随训练分叉，比较任务成功与探索/保留成本，不直接以不同reward的回报大小判优。

## 2026-09-14 用户暂停，等待补训奖励审查

用户澄清需要昨天72轮长训练，已声明9组(3幅度x3种子)72轮长campaign，替代12轮版本；随后明确要求暂停。两套自动执行器及当前仿真/训练子进程均已停止，保留已完成模型、日志和暂停前状态，不自动重启。当前入口 `runs/experiments/stable_recovery_long_campaign_20260914/USER_PAUSED.json`，执行状态为paused。新长campaign停在初始支持生成阶段，尚未进入新后继PPO。奖励尚未按新的讨论再次修改；当前实现仍为源Phase U奖励＋下降/落地恢复分阶段奖励。

## 2026-09-14 稳定落地三种子大循环已启动

用户要求停止追加单模型小评价，直接恢复多轮探索/补训并增加种子。当前入口：`runs/experiments/stable_recovery_multiseed_20260914/execution/status.json`。9991421/9991422/9991423各12轮、128候选/轮、0.15幅度3tick，当前策略单独验证、失败候选触发128k补训，保持原90%旧面板保留晋升规则。最坏新增总预算15,169,824交互，顺序运行三个种子。

初始Actor是用户指定Phase U transition4988928，经已完成训练的transition0哈希等价封装；没有采用64k或256k补训Actor。初始支持由该策略新轨迹生成。采集、续接标签、nominal支持、新策略奖励均使用连续2s稳定恢复；不再以首次接触即成功。源模型上游配置与物理参数保持；旧实验不改。训练过程、奖励分项、模型、采样及续接成本逐轮保存。48项相关CPU测试通过；实时GPU/轮次状态以运行文件为准，未宣称能力扩大或已完成全部训练。

## 2026-09-14 当前策略单独探索与验证

用户批准从冻结π0重新建立独立实验：仅新生成的π0真实轨迹提供初始支持，
不导入历史π1～π6或历史探索池。当前策略负责生成前缀、单独验证并初始化补训。
新增成功、固定起点成功、旧面板保留率>=90%且无未知后才接替当前策略；切换后
重建该策略的nominal ledger并重置探索器，保留旧策略和逐轮证据。该16状态面板
不是whole-Tube保证。首阶段12轮×128候选，幅度0.15、3tick、补训每次128k，
全阶段交互上限5,056,608（包含最坏情况下每轮nominal重建及配对评价）。
见[流程与预算说明](CURRENT_POLICY_EXPERIMENT.md)及
`runs/experiments/current_policy_lineage_20260914/INDEX.md`；实时状态以运行文件为准。

此前动作约束对照评价已完成：32k普通混合训练保留63/63旧成功状态，约束组60/63；
64k/128k两组均62/63；两组各checkpoint均学会2/2 pending。当前未见约束优势，
本次新谱系采用普通混合PPO。评价角色错误保留在原记录，完成结果位于
`runs/experiments/policy_retention_20260914/analysis_evaluation/`。

## 2026-09-14 已批准的旧能力保留对照

新增 opt-in source-action MSE：在942个冻结π0成功TRAIN快照观测上约束
新Actor的确定性归一化动作；teacher normalizer冻结，student normalizer沿用
PPO当前值。旧快照仅用于监督项，不回放成PPO轨迹。原奖励、物理和端点不变。
配置对照系数1/0、同seed9981401、128环境、各128000步，32k/64k/128k
checkpoint；相同64旧状态+2 pending对π0与6个checkpoint做全矩阵续接。
全部为TRAIN机制诊断，不是独立holdout。总新交互上限479200。

代码与配置见[实施计划](RETENTION_TRAINING.md)。运行入口：
`runs/experiments/policy_retention_20260914/INDEX.md`，实时启动状态只以其中
`execution/status.json`为准。后台GPU空闲门控不读取其他项目文件、不终止其他任务。
旧生产工件不修改；下方2026-09-12数值保留为历史快照。

核验日期：2026-09-12；文档修订前代码 HEAD `830a10f`。本页仅描述 DVGC/JIT。原始记录保持不变。

## 最新完成结果

`runs/experiments/delayed_completion_start_20260912/pipeline_status.json` 为 `completed`。该任务补完两轮 pilot 的第二次跳跃策略训练及延迟重评：PPO 128,000，固定 TRAIN panel 77，random suffix 111，learned suffix 219，总计 **128,407**，墙钟 **221.031 s（含等待）**。

首轮执行及两轮采集来自 `runs/experiments/delayed_exploration_reuse_20260912/queue/stage_0_result/`。其中首轮learned前向原件实际在`runs/experiments/delayed_exploration_20260912/queue/stage_0_result/round_000/learned_residual/arrivals/`，reuse保存锁定引用。该历史任务在第二轮等待阶段停止，旧 `error` 保留。两目录合起来提供已完成两轮的证据；不能从任一旧队列状态独立推断全流程。

| 轮次 | 条件探索臂 | 已见证候选条数 | 累计已见证 root cells | pending 条数 | no-witness→witness | unknown→witness |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | learned_residual | 158 | 150 | 2 | 0 | 0 |
| 0 | fixed_random | 151 | 151 | 6 | 0 | 0 |
| 1 | learned_residual | 308 | 289 | 12 | 0 | 0 |
| 1 | fixed_random | 308 | 308 | 9 | 0 | 0 |

`previously_witnessed` 是候选记录计数；`verified_root_cells` 是物理网格去重计数。每格有准确状态见证不意味着整格连续可行。上表来自两次 `round_metrics.csv`，不是把候选计数相加后的包线体积。

## 关联成本，避免重复计费

| 组成 | 实际 interactions | 说明 |
| --- | ---: | --- |
| 两轮两臂前向 | 76,800 | 每臂每轮19,200，含计费 padding；其中19,200继承自首个停止任务，只计一次 |
| 两次新跳跃策略 PPO | 256,000 | 每次128,000；不含历史 π0–π6 获得成本 |
| 固定 TRAIN panel | 149 | 首轮72 + 第二轮77 |
| 全部相关 suffix | 15,422 | 获取标签及延迟重评 |
| 本 pilot 关联实际合计 | **348,371** | 200,764（reuse 新增）+128,407（补完）+19,200（继承前向） |

预算上限、失败预留和实际执行必须分开。早期失败链和 3,302 步恢复工程测试在各自账本中保留，不假装属于上述无重复主 pilot。更不能把348,371称为从零获得整个系统的总成本。最新补完墙钟不是两轮端到端墙钟；中间等待时间也不是 GPU 训练时间。

## 当前能说明什么

执行已经覆盖：当前 π 内新颖性 ledger、冻结 Actor 的残差 PPO、pending 训练支持、两次新 π warm start、成功见证保留和旧候选延迟重评。

尚未观察到：旧 no-witness/unknown 候选被新 π 转化为成功见证。不能把最终已有308条 witnessed 解释为308条延迟学习收益。学习残差没有在本次条件探索结果中超过随机臂。

两臂前向预算相同并共享基线，但只有 learned pending 驱动共享后继 π；末次 learned 采用确定性动作，random 为随机采样。整个池包含训练采集与末次诊断，因此不是冻结同模式测试。每批最多32个候选，学习臂各批新到达均32，随机臂除首批30外也32；细网格新颖性几乎饱和，奖励尚不足以显示动作质量区别。一个连续训练谱系不支持独立重复统计。

## 历史证据仍有效，但口径分开

| 证据 | 结果 | 不能声称 |
| --- | --- | --- |
| all_proposers_v1 | 1,689→4,629→9,296 root cells；新增1,275,465；π5/π6各128k | 这些增长来自学习残差；所有增长都是新 π 的功劳 |
| checkpoint 32/64/128k | 同4,058探索预算 novelty132/186/132/165（π6/32k/64k/128k） | 32k是通用最优训练长度 |
| 重新初始化256k pilot | 同4,135探索预算 novelty134/122/103/161（π6/128k/192k/256k） | 更多训练必然更优；两张表属于同一连续训练 |
| first-success 离线调度 | 98,093→15,388有用步，假设节省84.31% | 实测墙钟加速84.31% |
| 批量接续工程 | 16,384环境容量测量，峰值14,522MiB、157,227 useful steps/s | 当前完整管线同倍提速或端点时间严格相同 |
| 小型残差正确性 | 基础 Actor不变、零残差回归基础、真实动作/快照链检查 | 科学性能已显著提高 |

历史4条前向 landing/failure冲突已进入事件审查；新existence视图隔离冲突。π6旧完整矩阵738-positive union在新派生审查中为735 witnessed、3 unknown，不能回写为原始实验标签。数值 replay差异按用户此前决定保留，不新增精确 replay门槛，也不称已精确通过。

两次长度pilot均从同一冻结π6、使用同seed9871101分别初始化fresh critic/optimizer；不是独立统计重复，也不是恢复旧optimizer继续训练。

## 完整原始证据与派生稿件

相对本仓库根目录：

- `JIT/runs/campaign/all_proposers_v1/`：历史生产（准确源路径另见论文 evidence.json）。
- `JIT/runs/experiments/checkpoint_discovery_20260911/INDEX.md`
- `JIT/runs/experiments/multicheckpoint_pi6_256k_20260911/exploration/`。
- `JIT/runs/engineering/continuation_parallel_20260911/INDEX.md`
- `JIT/runs/engineering/existence_checkpoint_residual_20260911/INDEX.md`
- `JIT/runs/experiments/delayed_exploration_reuse_20260912/queue/stage_0_result/INDEX.md`
- `JIT/runs/experiments/delayed_completion_start_20260912/INDEX.md`
- [稿件、控制框图和来源哈希](paper/README.md)；[下一步](JIT_TRAINING_ROADMAP.md)。

已核验报告分支 `origin/agent/jit-run-reports` 的最新提交为 `6121da9`（review-bf486f2945），其时间与内容早于本地最新 pilot。本地完整工件是本页最新结果的直接依据。最终 TEST/JCE/JEL 未开启；本次文档制作没有新仿真或 PPO。

## Quality-aware exploration training — 2026-09-12

User approved direct reward-ablation training and required inspectable learning histories/hyperparameters for every network. Implemented opt-in trajectory_quality_v1: full valid nonterminal trajectory novelty independent of candidate caps, once-per-episode physical-failure penalty, and normalized residual-energy cost. Pure weights1/0/0 versus quality1/50/0.01; matched uniform residual control. Base Actor and critic are explicitly frozen; their task reward and before/after-action critic values are telemetry, not exploration reward.

142related CPU tests pass. Real2world/2batch GPU smoke completed3200interactions;2optimizerupdates, complete reward-array summation checked, named losses/hyperparameters/inventory/CSV/PNG/PDF/SVG verified present. Background supervisor1149051 then started the three32batch8world arms,329600combined forward maximum. Current live status:`../runs/experiments/quality_exploration_20260912/execution/status.json`. Each arm preserves per-update optimizer_updates.jsonl, per-batch training_metrics.json/training_process.csv, hyperparameters.json/network_inventory.json, per-transition reward_components NPZ and task reward components, full trajectories/candidates/checkpoints and figures. No suffix labels/new jumping-policy training or claimed exploration gain yet. This stage keeps fixed starts to isolate reward effects; existing-state curriculum is not yet run. Compact evidence:`../review_evidence/quality_exploration_20260912.json`.

## Long quality exploration continuation — 2026-09-12

The three32batch quality-ablation arms completed: novelty/quality/random training new cells10897/10910/10759; successful forward episodes229/230/229 of256; physical failures17/15/15. This is not a demonstrated learned-exploration advantage or witnessed envelope expansion (zero suffix interactions).

User explicitly authorized JIT-only fixes and substantially longer residual training. Resume now restores actor/critic/Adam/normalizer/JAX RNG and per-policy novelty ledger; inherited global batch preserves stateless shuffle identity. Contract drift or corrupt checkpoint is rejected. Fresh full-batch post-update KL, clip fraction and value explained variance are recorded separately from minibatch pre-update metrics. Curves refresh every32 batches.

67 related CPU tests passed. Plan: two-batch real GPU resume smoke, byte-identical initial-state/ledger and process-artifact verification, then1024 additional batches from the original quality batch32 checkpoint. 8worlds, minibatch256, epoch1, Adam3e-5 and quality reward1/50/0.01 remain frozen to test duration. Not a hyperparameter-optimality claim. Maximum combined3,296,000 forward slots; expected main3,280,000 plus smoke9,600, effective samples reported separately. Bounded GPU-process wait and no automatic retries. No external project files inspected; GPU gate uses nvidia-smi only. Exact live state and all artifacts: `../runs/experiments/quality_exploration_long_20260912/INDEX.md`. GPU validation and main completion must be verified from that live execution, not inferred from this launch declaration.

## Long-run restart — 2026-09-13

The original GPU resume smoke completed9600forward slots, but the byte-identical serialization check blocked the long stage. Decoded59leaves (network, optimizer, normalizer, RNG) and ledger match exactly; only dictionary serialization order differed. Canonical verify_restored_checkpoint now checks file hashes and decoded structure/dtype/shape/exact values. Reused existing GPU smoke; corrected verification passed with no new simulation. Original failed queue and script remain immutable. User explicitly requested immediate long training: restarted only the1024batch stage from quality batch32, maximum3,283,200new forward slots; no extra smoke. Live status: `../runs/experiments/quality_exploration_long_20260912/restart_20260913/execution/status.json`; actual training artifacts remain `../runs/experiments/quality_exploration_long_20260912/long/`. No external repository files were read or changed.

## Short-pulse delayed quality loop — 2026-09-13 implementation

User authorized replacing continuous pi6 disturbances with pi0 + three-control-step bounded residuals at the fixed jump start. Residual is removed at handoff; full snapshot/history/event/RNG context is preserved. Real first-valid-landing outcomes define quality, not critic predictions. Same-context bank order pi0..pi6, then any newly trained bank members; stop each candidate at its first witness. Unresolved candidates are pooled into one128k jump-policy training per round using pi0-positive inherited support plus accumulated positives and pending snapshots, then reevaluated by the new policy. One bounded learning attempt per new batch; no claims of physical impossibility.

Implementation plan: (1) canonical pulse runtime for three-step capture and batched, early-stopped suffixes with full traces; (2) CPU orchestrator reusing candidate_support_view/make_config/freeze/lock_probe_bank; (3) resolved delayed-feedback PPO update on unchanged behavior-policy samples, no stale replay; (4) bounded direct launch after focused semantic checks. New modules pulse_exploration.py/pulse_exploration_runtime.py and thin run_pulse_exploration.py CLI. Historical continuous-residual results remain immutable.

Exploration reward:0.25shared credit per new handoff root cell +1successful bank/new-policy continuation or -1after attempted learning still fails. Prefix terminal failures are penalized; unknown/error is excluded or stops the stage. Visited ledger retains failed cells. The projected grid is not a certified continuous region. PPO uses privileged106D/256x3 networks,0.15four-action limits, minibatch128, up to4epochs, Adam3e-5, clip0.2, entropy0.001, value0.5, gradient clip1; full-batch KL threshold0.01 stops remaining epochs. Gamma/lambda1 for the three-step terminal quality return. New jump-policy PPO keeps its separate existing128world configuration and unchanged task reward. Every trainable network retains loss/reward/hyperparameters; per-round PNG/PDF/SVG and replot CSV accompany full prefixes, suffix traces, snapshots and costs.


Short-pulse live evidence: `../runs/experiments/pi0_short_pulse_20260913/INDEX.md`, current `training/status.json`. Real GPU same-start baseline: pi0..pi6 all successful,41/44/45/53/48/43/49steps (323total). First128pulses:76witnessed by pi0,52by pi1, no new-policy training needed;128×3prefix steps and13,476allocated suffix steps. Second128pulses also fully witnessed by pi0/pi1. Both completed explorer updates,6minibatchupdates each; KL0.01436/0.01155 stopped remaining epochs. These are TRAIN results, not independent validation. Continued from both completed rounds after restricting bank order to pi0..pi6 plus only newly learned members; unused historical late-checkpoint controls were excluded. Original failed launch/source-lock records retained; no repeated baseline or completed-round simulation. Focused15semantic/support/routing tests pass plus actual GPU prefix/suffix/update chain. The repair-policy branch is implemented but not yet triggered by these all-success batches.

## Short-pulse conflict recovery — 2026-09-13

The fourth-round pi0 suffix batch had two simultaneous landing/failure flags (lanes3/25), so the previous fail-fast implementation stopped after writing its128lane trace. Three rounds/384candidates had already completed explorer updates. Classifier now records each conflict as unknown, attempts other policies, and assigns bank failure only when every required evaluator explicitly fails. Unknown outcomes receive neither failure penalties nor PPO samples; an all-unknown batch preserves optimizer/network state and advances collection RNG without inventing an update.

Existing fourth-round128prefixes and all pi0 suffix traces were imported with context/actor/trace hashes; no completed prefix/suffix simulation repeated. Remaining candidates continue at pi1. Resume supports completed-round ancestry and retains all prior outcomes. Derived cost audit:48,451completed-round steps +384fourth-round prefix +13,440allocated pi0 suffix =62,275actual inherited steps, versus407,235historically reserved. Old failure files untouched.

Five focused reward/classifier tests pass. Live resumed job and acceptance evidence: `../runs/experiments/pi0_short_pulse_20260913/conflict_recovery/training/status.json`; derived receipts `conflict_recovery/reused_results.json` and `cost_audit.json`. Finite32round budget unchanged; child errors remain visible, no automatic retries.

## Multiple pulse locations — 2026-09-13

User authorized perturbations at multiple points along the real pi0 trajectory. Added configured pulse_start_schedule, with unperturbed pi0 approach and exactly3residual steps, amplitude0.15unchanged. Collector retains prefix_mask for all physical approach steps, separate mask for actual PPO pulse steps, per-candidate timing and complete handoff snapshots. Budget includes approach and pulse physics; pre-pulse terminal candidates cannot create invented pulse samples. Per-round metrics/region CSV include pulse_start_step. Existing zero-delay mode remains default.

New bounded run: pi0_multilocation_pulse_20260913,36rounds×128candidates, schedule5/10/15/20/25/0 repeated6times. Same initial start and current7-policy bank; fresh explorer with existing delayed-quality reward and PPO. One cyclic training lineage, not independent timing ablations. Maximum51,741,424 interactions includes worst-case expanded-bank suffixes and36bounded128krepair-policy trainings; actual separately recorded. Six focused tests pass. Live state and complete artifacts: `../runs/experiments/pi0_multilocation_pulse_20260913/INDEX.md`. GPU launch status is to be read live; no effectiveness claim yet.

## Long amplitude/seed campaign — 2026-09-13

User authorized substantially longer unattended JIT training and multiple attempts. Frozen plan `configs/pi0_pulse_campaign_20260913_plan.json` queues six fresh 72-round explorer lineages after the existing multi-location run: limits0.10/0.15/0.25 × explorer seeds9961302/9961303, each128candidates/round and3pulse steps at cyclic5/10/15/20/25/0. Total55,296candidates. Identical optimizer and initial pi0 support; bank policies are shared frozen inputs, not independent trained-policy replicates. Pending candidates retain the bounded128k successor-training branch.

Maximum1,019,011,488interactions is the conservative expanded-bank/repair reservation across six groups, not expected or actual consumption. Each group execution ceiling8hours; six finite stages plus120s comparison; gates have24h finite waits. Existing runner stops on any child failure or source/input drift, no retries or automatic extension. Every round retains loss/KL/reward components, hyperparameters, plots/replot data, full traces/snapshots and actual cost receipts. Automatic cross-group comparison runs only after all groups complete. Schema, all declared hashes, exact budget formula and comparison-script syntax verified; no extra GPU test run. Active multi-location first collection confirms5zero-residual approach steps and3training steps per lane, maximum effective residual0.14778815.

Live queue and declaration: `../runs/experiments/pi0_pulse_campaign_20260913/INDEX.md`; read execution/status.json for current status, not this dated snapshot. No final TEST or scientific superiority claim.

## Terminal prefix repair and resumed campaign — 2026-09-13

The original six-arm queue stopped at amplitude0.25 seed9961302 round0001: two terminal lanes were passed to the nonterminal snapshot API. Completed72round0.10/0.15 arms stay immutable. Collector now emits terminal trace records without restart snapshots; actual terminal landing/failure/conflict labels are distinct, and terminal positives cannot enter restart support. New tapes retain physical_failure/end_code. Legacy landing terminals without conflict telemetry remain unknown. No clearing done or inventing continuation states.

Seven focused tests pass. Real GPU recovery collection reused the saved13×128prefix plus locked behavior/post-collection optimizer/RNG without new physics:128candidate records,126valid snapshots and2terminal records (lanes21/82), collection receipt0new interactions. Prior11893interactions include10229completed receipts plus1664saved physics. Suffix evaluation has started for the126nonterminal candidates. Recovery queue continues this arm to72rounds then runs three unstarted seed9961303 arms; same amplitude/reward/PPO budgets,8hper-stage ceiling, no automatic retry. Frozen configs are pi0_pulse_campaign_recovery_20260913_*.json. Live index: `../runs/experiments/pi0_pulse_campaign_recovery_20260913/INDEX.md`.

## 2026-09-16 已授权的效率修复与续训

针对RSL两奖励臂训练的重复读盘瓶颈，完成同一输出轨迹哈希逐候选重复计算的消除，
以及单次策略库核验内的文件哈希复用；保留内容哈希比较和元数据变更失效。
新增接续初始化、恢复、编译与仿真、导出耗时。27项CPU测试通过；不等同于GPU提速证明。
当前派生验证与续训入口为
`runs/experiments/rsl_reward_comparison_20260915/efficiency_resume_20260916/`，
实际是否运行以其execution/status.json和进程为准。继续94轮后的已有进度，
两臂各200轮、1024环境、0.25幅度、0.001探索器初始学习率不变。
细节见[EVIDENCE_IO_OPTIMIZATION](EVIDENCE_IO_OPTIMIZATION.md)。

## 2026-09-18 七策略评估截止状态修复

三路14,991,360步Phase U训练全部完成；0和5步起扰评估完成。10步起扰第二批fresh RSI后代出现400步未终止样本，原代码误调用接续快照接口并缺少formal_config_sha256。修复完整回合截止为horizon_trace，不改变物理terminal或成功标签。新增prepare-resume复用身份锁定训练/完成成员、独立代码与尝试目录，失败102,400交互另计。CPU回归及保存的lane198截止证据已验证；恢复后的GPU执行状态见实验ACTIVE_RUN指针。
