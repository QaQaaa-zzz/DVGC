# Codex 任务书：MuJoCo → Isaac Sim 跳跃迁移诊断与落地适配 V2

## 0. 任务范围与基准

用户目标：解释并改善“源策略在 PhysX 能起跳但侧倾摔倒；目标域 PPO 续训仍未稳定落地；从零 PPO 不起跳”。优先利用已有起跳能力，不先扩大网络，不直接追加一轮 1000 万步。

审计基准：
- 仓库：QaQaaa-zzz/DVGC。
- 分支：reports/sim2sim-ppo-20261009。
- 审计时分支头：09f1d1deb4aa3993341e163cc30ad30f8c505c5b。
- 证据根目录：docs/evidence/sim2sim_ppo_20261009/。
- 冻结运行代码：models/physx/runtime/。
- 冻结源奖励/事件：models/mjx/jit_dvgc/{rewards.py,semantics.py,constants.py}。
- 真实工作代码原位置由 source_map.json、运行配置、项目 AGENTS.md 确认。归档中出现 /home/qy/ISAAC——SIM 等绝对路径，不要猜测 Unicode 路径，也不要假定归档就是当前训练入口。

这些是历史证据，禁止覆盖。读取当前工作区 git status、AGENTS.md、JIT/AGENTS.md 和实际 Isaac 项目的说明，建立独立修复分支和新运行目录。保留用户未提交修改。未被授权，不推送远端，不启动无预算长训练。

本任务中的数值为建议首轮参数，不是已验证最优参数。每种修改必须有独立配置版本与消融，不得把改过任务、物理或动作后的结果写回旧实验。

## 1. 已确认现象：先复核，不要重新猜原因

1. 最终确定性单自然初态 PhysX 评估：
   - Source 9904128：apex 1.26 s；最高 root z 0.561 m；1.52 s roll_limit。
   - warmstart final：apex 1.10 s；最高 root z 0.571 m；1.78 s roll_limit。
   - fresh final：无 apex；最高 root z 0.150 m；3.18 s stuck；最终 x 2.500104 m、vx 0.0900 m/s。
2. warmstart 末 100 万步：2576 个 ground 回合、2510 个 apex、2152 个 timeout、平均 357.24 控制步。它是多个更新时点的采样策略统计，不能直接当 final 确定性落地成功率。
3. fresh 末段：4128 个 ground 回合、0 apex；348 个 airborne RSI 回合全部有 apex。RSI 初态直接给了向上速度，不是从地面学会起跳。
4. fresh 首批 kl_mean 101.22396；后续有 55.42096、9.7072、8.6475 等尖峰。warmstart 最大 0.42325。先核实 KL 的定义、方向、求和/平均维度。
5. warmstart 轨迹约 0.94–1.34 s 连续 21 个控制步 reward=50；最后摔倒后总回报仍约 801.8。
6. legacy semantics 中 success 恒为 False；apex_seen 不是稳定落地；没有完整 landing success 事件。不能把这些算作新任务已经完成。

复核来源：results/kd5/{evaluation_summary.json,training_last1m_episodes.json,kl_audit.json,warmstart_final_trace.csv}。

## 2. P0：修好配置溯源与观测记录，不改变历史物理

重点文件：physx_task.py、policy_runtime.py、train_physx_brax.py、brax_physx_learner.py、运行/评估入口。

### 2.1 配置只有一个有效来源

当前 physx_task.py 的模块全局 RAW/CFG 从 ROOT/policy/resolved_config.json 读取，而 spec 中也有 reward/reset 等字段。不能仅修改声明 JSON 的 reward 就宣布改了奖励。

执行要求：
- 引入明确版本的 resolved task config，并通过构造函数传给 Task、Observation、奖励、事件和控制映射。
- legacy_replay 模式不改变实际值与行为，保持可回放。
- jump_landing_v2 模式使用显式配置，不隐式读取另一份冻结配置。
- 对重复字段建立一致性断言；打印并保存最终有效配置和 SHA256，包括奖励、事件、reset、动作、dt、物理参数、观测协议。
- 检查 policy_runtime.py 中 2.5、4.0、3.6 等硬编码位置；v2 使用统一配置，legacy 仍得到原数值。
- 每项奖励做配置敏感性测试：构造同一状态只改变该系数，实际奖励必须随之变化。
- 输出 import 路径与依赖版本，不允许默默使用正在变化的 JIT 主线替代冻结实现。

### 2.2 物理与事件观测

保持初始 baseline：5 ms physics、20 ms control，rear 0/5、steer 2.5/0.4、hip/knee 显式 100/6 且 ±30 Nm，当前轮胎 0.5/min，无 PI/固定动作覆盖。

先在诊断小面板加入当前 Isaac Sim 5.1 支持的真实 contact report/contact sensor：
- 前/后轮与 floor/step 的接触、法向力/冲量、接触点、接触法向。
- 各非轮刚体碰撞与对象，区分平台正面碰撞和平台顶面支撑。
- 多个 wheel convex sector 按刚体/接触对象聚合；不能把重复来源重复计数。
- 区分传感器无效、未刷新和真实无接触；传感器无效必须报告，不能记作飞行。
- 每个 physics 子步采集或正确聚合到 control 步，短冲击不能只读最后一帧漏掉。

记录：root/COM 状态、姿态和角速度、四路动作均值/采样值/分布参数、pre-tanh 动作、映射后目标、实际关节位置速度、执行力矩、饱和比例、轮接触切向速度、碰撞、逐项奖励、clip 前后奖励、旧/新观测归一化统计、reset 类型、env lane、世界 origin、事件位。

不要把当前基于几何包围支持范围的 prohibited-contact 代理，直接当成真实接触传感器。启用真实非轮接触终止前，先检验原地、合法触轮、平台边缘和轻微数值接触案例，避免错误终止。

## 3. P1：先解释训练与评估分歧

### 3.1 相同源 checkpoint 的两引擎评估

对 9904128 使用同一完整落地评估协议，在 MJX 与 PhysX 都评估。当前主图的 Source 9904128 也是 PhysX，历史 4988928 双引擎图不能替代这一步。

分别报告 takeoff、apex、首次轮接触、稳定落地、落地后继续前进、侧倾/俯仰失败；没有接触记录就不声称已证明 landing。

### 3.2 最小因子面板

先固定 checkpoint=9904128 与 warmstart final，固定实际初态，而不只是 RNG seed：
- 动作：deterministic / stochastic。
- 布局：1 个环境 / 384 环境中的指定 lanes。
- reset：新进程首次或 hard reset / 自动 reset 后再次运行。

分阶段执行，先定位主差异，不全排列所有历史 checkpoint。随机策略固定并保存每 lane 的随机噪声序列，至少多个独立动作 seed。确定性运行重复记录重复性，不能把相同初态不同 seed 当独立鲁棒性样本。

训练末段统计有多个 checkpoint，因此需要额外选几个保存的中间 checkpoint，用同一 final 评估接口评分。

布局诊断加入：单车在 origin=0 和若干训练网格远端坐标的位置平移测试；所有 actor 位置特征必须为任务局部坐标。检查 lane 排序、碰撞过滤、障碍相对位置、初态 root/COM 定义、轮速与 contact/solver reset。

先比较完全相同观测张量下：learner deterministic 与导出 Actor 的动作，误差目标沿用已有 export 测试的 3e-5，并扩大到固定状态库和极值输入。

随后保存同一动作序列进行短时 replay，比较首次显著分歧。不要要求不同引擎长时轨迹逐位相等，也不要在本引擎重复性尚不明时把长时差异全部归因于策略。

输出 eval_factorial.csv、first_divergence.csv、action_export_parity.json、repeatability.json 和结论。gate 未通过，停止扩大正式 PPO 预算；可以继续纯测试修复。

## 4. P2：修复 PPO 更新稳定性，先不改奖励验证基础设施

### 4.1 归一化审计

当前 learner 在 rollout 后、SGD 前更新 normalizer。采样使用旧统计，SGD 使用新统计。它是有意复刻的 legacy Brax 行为，不要直接宣称迁移实现写错了；但必须定位这是否造成当前巨大 KL。

对固定 rollout 分解：
A. 参数和 normalizer 都不变：ratio 接近 1，KL 接近 0。
B. 只改 normalizer，不做 SGD：normalizer_only_KL。
C. normalizer 冻结，只做 SGD：optimizer_only_KL。
D. 完整更新：total_KL。

保存各观测维度 mean/std 变化、normalized observation 极值/分位数、动作均值/std 改变、物理动作饱和比例。

首轮 warmstart：冻结源 actor 使用的 normalizer。新奖励不要求修改观测统计。后续确需更新时，必须对统计变化本身执行 KL/动作偏移门控。仅把统计更新延后到下一 rollout，不等于消除了策略突变。

fresh：可先用该 fresh 策略/有界随机中性动作采样若干完整 rollout 收集统计，明确计入交互预算；初始化统计后在首段实验冻结。不得偷偷使用源训练数据然后仍称纯从零对照。

### 4.2 KL gate 与真实回滚

建议首轮 warmstart 参数：
- actor learning_rate=3e-5。
- num_updates_per_batch=2，先不沿用 8。
- clipping_epsilon=0.1。
- entropy_cost 先试 0.001–0.003，保留探索，不突然将 std 清零。
- target_KL=0.01，soft stop 约 0.02，hard reject 约 0.05。
- max_grad_norm=0.5 保留。
- 首轮环境数/rollout 仍为 384/64，避免同时改变样本构成；以后单独比较 128-step rollout。

阈值都是首轮候选，记录 KL 定义。clip epsilon 不是 KL 上界。

当前 lax.scan 固定执行所有 epochs/minibatches。加入可执行的中止/候选拒绝逻辑，不得只是训练结束打印 KL。过大候选更新需恢复参数与 Adam 全状态，包括 step/moments；normalizer 若参与候选也一起恢复。拒绝后的样本消耗仍计入 interaction budget。

记录 grad_norm_pre_clip 与 grad_norm_post_clip；现有 grad_norm 很大不等于梯度裁剪未生效。

增加单元测试：no-update ratio=1；超 KL 的候选不被接受；回滚完整；没有有效样本时不除零；pre-tanh/squashed log_prob 契约一致；导出 actor 包括相同归一化。

### 4.3 GAE / timeout

legacy 采样把 timeout 当 discount=0 且 truncation/time_out=0，代码明确是源 wrapper 行为复刻。

v2 必须先声明任务语义：
- 有限时间任务的失败 deadline：可定义为真实 terminal。
- 仅用于截断采样的人工 time limit：需要从截断前 final_observation bootstrap。

实现时区分 bootstrap mask 与 GAE 跨 reset 的 continuation mask：不能从 reset 后初态 bootstrap，也不能让优势递推跨回合。不要仅把一个 truncation 布尔值改成 1 就声称修复。

在固定小轨迹上与手工 returns/GAE 比较。不要悄悄重解释旧实验。

## 5. P3：建立 jump_landing_v2 的真实成功事件

新增独立模块名称可按工作项目组织，例如 jump_landing_events.py、jump_landing_reward.py；不要修改冻结证据文件。

状态机：APPROACH/PREPARE → FLIGHT → TOUCHDOWN → SETTLE → SUCCESS/FAIL。

要求：
- 从地面 episode 必须先存在合法轮接触，后有双轮脱离并有向上速度/实际离地间隙，才记真实 takeoff。
- 接触与离地使用迟滞和连续步确认，初始模型短暂下落、接触抖动、爬台阶不能算跳跃。
- 首次 touchdown 允许前轮/后轮先接触，不要求两轮同一时刻落地。
- stable landing 需要合法轮支撑、无非轮碰撞、姿态及角速度受控，并持续保持。平地/平台用实际支撑表面，不拿固定 world root z=0.15 判断。
- 台阶从 x=3.6 开始、顶面高度 0.16 m；目标是上台阶并稳定滚行，不是飞越整块 4 米长平台。
- 仅越过台阶前缘，不等于完整落地；完整成功需要后续支撑与姿态条件。
- airborne RSI 的 suffix success 单独统计，永不计入 ground takeoff/success。

建议开发阈值：落地后稳定保持 0.5 s，另评估 1.0 s 严格保持；roll <10°、pitch <15°，roll/pitch rate 各 <1 rad/s，侧向速度 <0.25 m/s，且保持向前运动。接触力门限按整车重力归一化，并依据无接触噪声定标。速度阈值与目标是否 2 m/s 必须明确声明，不能用低速成功冒充 2 m/s 跟踪成功。

正式测试阈值在开发前冻结；可依据源域正常稳态数据定标，不能根据测试结果不断放宽。训练成功后终止可以节约采样，但最终评估需继续运行，检查成功时刻之后是否马上摔倒。

事件机放在奖励/评价侧，首轮可保持 76/106 维 actor/critic 协议。若确认现有观测不能辨别阶段，再单独增加可观测 contact/phase/vertical velocity 等字段。扩展输入时版本化协议、同步 critic 和导出器，并用新输入权重零填充验证初始化策略行为不变，不能修改常数后强行加载旧权重。

## 6. P4：奖励 V2——不继续只增加高度奖励或摔倒惩罚

原则：有界的进展反馈 + 一次性事件奖励 + 落地姿态/角速度反馈。以下为首轮候选，需要用保存轨迹离线重评分后再运行。

1. 去掉 legacy 每步 40–60 的高度驻留主奖励，改用小幅有界高度/推进进展 shaping。可用 r_shape=gamma*Phi(s_next)-Phi(s)，Phi 基于阶段、实际高度增量、向上速度和前向进展；物理 terminal 的 terminal potential 定义为 0。保留负差分，不允许只裁正增量来奖励反复抬高。phase 切换也必须参与统一势函数，避免边界凭空领分。
2. 一次性真实 takeoff +10；apex +5；合法跨越平台前缘/清障 +20；完整稳定落地 +100。每个事件最多一次。airborne RSI 没有 ground takeoff bonus。
3. 候选 physical failure 统一常数 -50；去掉 failed_episode_return-self.returns 这种全回合回报重写。具体失败权重通过成功/失败轨迹排序测试定标，不把 -50 宣称最优。跳跃窗未完成采用明确 deadline/失败，避免原地赖活。
4. 撤销总奖励 [-400,50] 的日常硬裁剪；分项先有界，并监控异常与分位数。一次性 success reward 不得被总 clip 吞掉。非有限值直接报错，不用剪裁掩盖数值问题。
5. 落地阶段 r_stable 可从下面的有界式起步：
   0.5 * exp[-(roll/10deg)^2-(pitch/15deg)^2-(roll_rate/1)^2-(pitch_rate/1)^2-(vy/0.25)^2]。
   仅在真实、合法 touchdown 后给分，且有成功保持时限，不能碰撞支撑或无限站着刷分。
6. 加入 roll_rate 反馈。起跳的必要俯仰运动与落地失控不能用相同权重压制；flight/landing 主要惩罚危险侧倾角速度和超限姿态，prepare 的 pitch rate 惩罚较弱。
7. 髋膝功率罚按 physics 子步积分 sum(dt*abs(tau*qdot))，避免用最后一个子步的功率代表整个控制周期。首先记录能量占比；探索阶段可将 legacy 系数 2 降到 0.2–0.5 作消融，落地稳定后再收紧。所有参数用同一量纲，不突然叠加一个很大的冲击惩罚来阻止机器人尝试跳跃。
8. 地面速度奖励与向前进展保留但不主导，空中不要靠不可能的轮地作用奖励精确跟速。动作平滑以实际执行目标的变化和饱和统计辅助校准，不能鼓励无法作用到执行器的空动作。
9. 撤销每步固定 survival=1.5 的无限刷分空间，prepare 阶段有界进展/小额姿态奖励和 deadline，settle 阶段有限保持奖励。
10. 当前 stuck 在跳跃窗内、未达到高度时可由单步低速度触发。新版本使用 0.3–0.5 s 窗口内的低速且无前向进展判据，并给初始落稳/短暂蓄力合理宽限；实际阈值按源策略准备时长定标。不能允许无限停车躲任务。

离线奖励测试用例：
- 站着/慢行不跳；起跳但落地摔倒；合法完整跳跃；空中 RSI；身体擦/顶平台；反复接触抖动；同高度不同 roll/p；早失败和晚失败；超时截断。
- 完整成功必须优于只到 apex 后失败；站着不跳不能靠延长回合无限获利。
- 检查折扣回报，也检查未折扣回报；缩放后的 critic target 和旧 critic 不应未经审计直接混用。

奖励变化后，先冻结 actor，用 PhysX 目标域数据让 critic 适配新回报若干更新。源 critic 不能默认准确；可对比重置 value 输出层与保留 critic。再开启小步 actor 更新。

## 7. P5：保护起跳的目标域续训，优先于从零

优先初始化：9904128 与经固定开发面板选出的 warmstart 中间 checkpoint。不能仅用 final；不能按训练累计回报挑 best。

先做分阶段后缀训练：
- 在 PhysX 里由冻结源策略执行真实前缀，分别从顶点前、顶点附近、下降早期进入学习控制；使用事件/支撑情况定位，而不是跨引擎硬套固定秒数。
- 首批从较容易、可恢复的下降/接触前状态开始，同时覆盖较早阶段；不要只采集已经大角度摔倒的状态。
- 初始可约 40% 完整地面回合、60% 真实前缀后缀回合；后缀稳定后逐步提高完整地面比重，最终以完整自然任务评估。
- 不要求一开始只更新落地后的动作。过大的入射角动量必须在起跳/飞行早期修正。

状态保存/恢复：root、关节、速度、3 帧历史、history mask、last_action、事件位、previous velocity、轮连续角、目标/控制状态都需一致。PhysX solver/contact 缓存未被完整保存时，不能宣称任意 snapshot restore 与真实前缀逐位等价；优先真实前缀重放，并明确开销。不要 reset 成空白历史后声称同一落地状态。

PPO 只使用学习策略真正采集的 on-policy 后缀；冻结 teacher 前缀不是 student 的 on-policy 数据，禁止伪造旧 log_prob 混入。记录前缀 physics 开销与后缀 learner transitions。

遗忘约束：
- 冻结源 teacher 与其 normalizer。
- 在地面接近和已验证的起跳状态加行为克隆/teacher KL 约束。
- 临近落地不要强迫学习器模仿已知会失败的源动作；约束按阶段减弱。
- 一部分完整起跳回合始终保留，防止后缀训练毁掉起跳。
- 区分 PPO trust-region KL 与 teacher-retention KL，两者用途不同。

从零训练作为独立后续路线：从平地小幅双轮离地开始，再增加起跳高度、再增加 0.05/0.10/0.16 m 台阶。课程按固定开发面板能力晋级，不凭迭代次数自动晋级。不能把空中 RSI apex 当成起跳课程通过。

## 8. P6：独立核验动作可达性与物理差距

### 8.1 后轮速度映射

当前 w_cmd=12+12*a，a∈[-1,1]，实际是 [0,24] rad/s，而轮半径约 0.07 m。水平纯滚动的稳态速度尺度最高约 1.68 m/s；2 m/s 约需 28.57 rad/s。执行器 ctrlrange [-5,40] 不代表策略能够输出到 40。

要求：
- 做水平地面轮速目标阶跃和稳态标定，记录车速、实际轮速、滑移、力矩饱和。
- 控制不变的落地诊断可使用明确声明的可达速度参考，但正式 2 m/s 成功必须在可达控制映射下单独验证。
- 可独立建立 action_v2：后轮目标覆盖 [0,40]，或在旧物理目标后增加有界 residual；严守原力矩上限。
- 不得直接把 base/delta 改成 20/20 后声称源策略保持不变。需要新的动作/previous-action 观测协议、物理目标一致的克隆/适配或零初始化残差，并检验初始行为。
- 首轮奖励消融和物理消融不要混入这一修改。

另查髋膝探索几何：knee keyframe=2.5 正好是上限，legacy 非负 knee action 全映射到 2.5；hip -1.2 接近 -1.3 下限，正负 action 有效行程不对称。它不是移植写错，但 fresh 有效探索受影响。报告每路目标变化、无效动作和饱和比例；新控制参数化需独立版本，不能破坏旧 checkpoint。

### 8.2 接触/执行器消融

已知源端显式 contact pair 的切向系数不等；目标端为各向同性 0.5/min。MuJoCo 的 pair 切向轴须读取 contact frame，不能一概当成所有姿态下的车体纵/横轴。

按顺序：
A. 相同 MJX checkpoint：原接触与仅把轮-地/轮-平台 pair 前两个切向系数改为 0.5/0.5 的副本，其他保持不变。判断源策略对各向异性接触的依赖。
B. PhysX 在固定 checkpoint、固定动作/初态小面板做 wheel friction 0.5/0.8/1.0 的单因素测试。读回轮、地、台阶双方材质与 combine 后有效值，min combine 可能让只改轮胎无效。不要用更大摩擦“刷出成功”后冒充原物理匹配。
C. 两端 5 ms 下重新校验转向/后轮/髋膝受载阶跃和实际力矩；rear/steer 原生隐式 drive 不等于源端显式相同离散响应。不得重复施加 drive+effort 双份 PD。
D. 轮 ellipsoid→16 个 convex sectors 仅作为候选误差源，通过接触/滑移诊断后再做 16/32 或共同碰撞近似消融。不能凭猜测改 mesh、质量、力臂。
E. 只有接触冲量或离散化证据明确时，做 2.5 ms physics、20 ms control 的单因素检查；IMU 加速度定义与能量积分必须随采样定义正确处理。不要照搬历史 1 ms 的 20 ms 平均加速度适配，归档已显示它在 5 ms 下可能变差。

若 A 明确证明各向异性依赖，优先在 MJX 建立与目标端一致的各向同性“桥接环境”进行短适配，再去 PhysX 微调；报告它是适配，不是 zero-shot 原始策略迁移。不要优先靠人为横向稳定外力、固定后轮、拉长结构或视觉回正掩盖问题。

## 9. 运行预算、实验矩阵与验收

先做配置/事件/GAE/PPO 单元测试及小面板，不直接启动所有训练。固定物理与 checkpoint 下，最少比较：
E0 legacy baseline；
E1 仅稳定 learner（normalizer/KL gate）；
E2 E1 + landing_v2 reward/events；
E3 E2 + 真实前缀后缀课程 + 起跳保持约束。

课程和保持约束在 E3 有收益后再拆开消融。动作重参数化、摩擦与桥接环境属于独立实验轴，不与 E1/E2 同时修改。

采用小预算分级：
- 384×64 时 20 PPO blocks=491,520 learner transitions，40 blocks=983,040。
- 先 20 blocks 检查趋势，再决定到 40 blocks；中途每 10 blocks 固定开发评估。
- 若 rollout 改为 128，必须重新按实际 block 计算预算，不沿用上述数字。
- 一个方案有稳定落地改善后，再补多个训练 seed。未实际运行的实验标为未运行。

开发评分顺序：完整自然初态稳定落地率 → 更严格保持时长/落地后前进 → 合法清障率 → apex率 → 其他诊断。训练 return 只作辅助，不作为主要 best 标准。

测试包含标准 natural case、事先固定的小扰动初态（roll、vx、关节等）和独立测试集。不要把同一确定性初态跑 N 遍当 N 种泛化条件。记录分子/分母，必要时给区间，不用一条视频声称成功率。

明确失败停止条件：
- 配置改动未真正生效；
- actor/export 不一致；
- contact 数据无效仍用于成功判定；
- accepted update 超硬 KL 阈值；
- train/eval 模式差异未解释；
- ground takeoff 继续为零，却用 RSI apex 宣布课程成功。
这些情况停止扩大训练预算，先修复/报告，不删除失败证据。

## 10. 交付

新输出目录需有：
- resolved_config.json、代码/模型/checkpoint SHA256、依赖版本、实际 CPU/GPU backend；
- tests 与运行命令、测试结果；
- eval_factorial.csv、repeatability.json、first_divergence.csv；
- kl_decomposition.csv（normalizer-only/SGD/total）；
- contact/landing 逐步记录和 reward-components CSV；
- deterministic/stochastic × ground/RSI 分组评价；
- source/best/final 同协议图，坐标/速度/高度/姿态/角速度/接触/动作/奖励；
- 消融表、所有耗用的学习和前缀物理步数、best 选择记录；
- REPORT.md：分开写“已证实”“支持但未隔离的解释”“未运行的建议”。

当前代码显式把 JAX 设为 CPU，且用 CPU/NumPy PhysX 采样和每车 Python/kinematics 循环。正确性通过后再单独 profiling；不要把更换 GPU backend 与奖励/物理修复混为一个实验，也不要只改设备字符串就宣布完成 GPU 迁移。

最终回报必须回答：
1. 同一个 9904128 在 MJX 是否完成了严格落地？
2. train/eval 分歧来自哪里，有何对照证据？
3. fresh 是探索局部最优、更新失稳、还是物理可达性问题，各有多少证据？
4. 是否真正提高了完整地面任务稳定落地，而不仅是 apex 或 return？
5. 哪项修改最有效，代价是什么，哪些假设仍未证实？
