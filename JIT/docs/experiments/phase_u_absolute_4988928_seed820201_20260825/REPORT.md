# Phase U 绝对关节目标 4,988,928 步正式训练报告

## 1. 最终结论

本次 fresh GPU 训练已经完成，严格 provenance verifier 通过，但实验结论是
**NO_PROMOTION**：最终策略不是可用的 Propulsion-Ascent expert。

- 自然起步：五个里程碑全部 `0/8` Apex、`0/8` 到达起跳窗口、`8/8`
  物理失败。最终策略在 2 个控制步内触发 `illegal_wheel_contact`。
- 强制空中 RSI：五个里程碑全部 `8/8` Apex，且没有物理失败。
- 两者不能合并。RSI 结果仅说明从 `z≈2 m、vz>0` 的状态很容易满足当前
  高度/下降 Apex 判据，不能证明策略学会了从地面推进、准备和起跳。
- 最终自然轨迹直接违反用户提出的“即使不能跳也不应直接摔倒”。继续把同一
  配置追加到更多步数没有科学依据。

本轮代码、配置和验证源提交为
`96cfd81c5bc2f462c6718e0ccd4313e0e62b40bd`，在训练前已经推送到
`origin/agent/two-phase-soft-tube`。初始训练没有传入
`--restore-checkpoint`，run manifest 的 parent checkpoint 为 null、起始步为 0。

## 2. 本轮具体修改及原因

| 模块 | 修改 | 原因 |
|---|---|---|
| 动作映射 | 活动 v3 中 hip 和 knee 都使用 XML keyframe 为零点的分段绝对目标 | 按用户选择的第一种方案，消除 knee 依赖当前角度的增量语义 |
| 配置身份 | 新增严格的 `jit_phase_u_*_v3` smoke/formal schema 和完整常量漂移拒绝 | 防止旧 checkpoint、错误动作语义或被修改的训练配置冒充本次实验 |
| PPO 预算 | 新增 4,988,928 步、384 并行环境的 block 对齐配置 | 接近用户要求的 500 万步，同时满足 Brax 整块更新约束 |
| formal 调度 | checkpoint/evaluation 调度改为读取已精确验证的配置 | 支持新预算，而不是把历史 998,400 步常量写死在生产逻辑中 |
| RSI 诊断 | 训练仍为 5% 空中 RSI；每个 checkpoint 额外执行强制 RSI panel | 单独判断稀有空中子问题是否学到，且不污染自然起步 promotion |
| checkpoint 安全 | 先验证 identity sidecar，再反序列化 payload | 旧网络/旧动作合同在读取 pickle 前就被拒绝 |
| 轨迹证据 | 自然/RSI 分别保存每个 seed 的完整 NPZ，最终另存 MP4、PNG、NPZ | 同步检查奖励、位置、姿态、动作、控制、终止与 reset 来源 |
| provenance | 校验 artifact 类型、互异路径、哈希、视频实际帧数、PNG 解码、NPZ 样本数和代表轨迹 lineage | 防止只改 JSON 或调换自然/RSI 文件就伪造一套“通过”的证据 |

没有修改 XML、mesh、碰撞几何、仿真步长、奖励公式、观测维度、jump signal、
Apex 判据或动作顺序。历史 v2 的 incremental-knee 语义只为验证旧证据保留，
没有进入本次 v3 训练。

## 3. 动作映射

策略动作顺序仍为 `[steer, rear-wheel drive, hip, knee]`，每项先裁剪到
`[-1,1]`。

| action | -1 | 0 | +1 |
|---|---:|---:|---:|
| steer control | -0.8 | 0 | +0.8 |
| rear speed | 0 | 12 | 24 |
| hip target | -1.3 rad | -1.2 rad | +0.5 rad |
| knee target | -1.5 rad | +2.5 rad | +2.5 rad |

hip/knee 使用同一公式，但 knee 的 keyframe `2.5 rad` 正好等于 actuator 上界。
因此 knee 的正半轴是退化区间：`action>=0` 都映射到 `2.5 rad`；只有负动作能
把膝关节从 keyframe 向 `-1.5 rad` 移动。这是权威 XML 数值与用户选择共同导致
的真实合同，不是程序给 knee 另设了一套规则。

## 4. PPO 配置及与此前 CPU 配置的对应关系

本次 Actor/critic 仍分别为 `76 -> 256 -> 256 -> 256 -> 8` 和
`106 -> 256 -> 256 -> 256 -> 1`。Actor 的 8 个输出参数化四维 tanh-normal
分布；critic 包含完整 Actor 输入，因此两者都能看到 jump signal。

| 项目 | 此前 CPU/SB3 示例 | 本次 GPU/Brax |
|---|---:|---:|
| learning rate | 3e-4 | 1e-4 |
| gamma / GAE | 0.99 / 0.95 | 0.99 / 0.95 |
| clip | 0.2 | 0.2 |
| entropy | 0.01 | 0.01 |
| gradient norm | 0.5 | 0.5 |
| optimizer passes | 8 epochs | 8 updates per batch |
| effective minibatch | 1,024 transitions | `16 x 64 = 1,024` transitions |
| rollout collection | 2,048 steps x 少量环境 | `384 x 64 = 24,576` transitions/block |
| total budget | 未列出 | 4,988,928 = 203 blocks |

Brax 的 `batch_size=16` 表示每个 minibatch 中的 16 条 unroll，不等于只有 16
个单步样本；乘上 unroll 64 后恰好也是 1,024 transitions。因此本次并没有
盲目把 CPU 的 1,024 minibatch 改成极小批次。主要差异是用大量 GPU 并行环境
缩短单环境 rollout，并把学习率从 3e-4 降到 1e-4。

## 5. 奖励和终止合同

本轮没有再次改变已经对齐参考文件的奖励。精确分项为：

- roll：系数 3.0 的分段姿态奖励；pitch 系数 1.0；yaw 系数 0.3。
- speed：`0.2 * exp(-0.5*((vx-3.5)/0.5)^2)`。
- survival：每个环境转移 +1.5。
- height：仅当前 `jump_signal=1` 时有效；高度形状乘 20，`z=0.35/0.50/
  0.80/>0.80 m` 分别约为 `20/30/12/8`。
- action smoothness：`-1.5e-4 * sum((a_t-a_(t-1))^2)`。
- action magnitude：`-0.15 * sum(|a_t|^1.5)`。
- pitch rate：`-0.01875 * pitch_rate^2`。
- joint energy：`-2 * 0.02 * (|tau_h*qdot_h| + |tau_k*qdot_k|)`。
- 首次 Apex +50；非法接触 -30；首次物理失败 -30；timeout -10。
- 分项先求和得到 unclipped reward，再裁剪到 `[-50,50]`；PPO 使用
  `reward_scaling=0.1`。

起跳窗口为 `2.5 <= root_x <= 3.1 m`。jump signal 只在尚未消费窗口且当前位于
窗口内时为 1；离开窗口后永久为 0。Apex 要求历史上访问窗口、出现过
`vz>=0.05 m/s`、达到过 `z>=0.5 m`，随后当前 `vz<=-0.05 m/s` 且没有物理
失败。成功就在 Apex 终止，不包含平台后续减速。

## 6. 训练声明和账本

| 项目 | 结果 |
|---|---|
| run ID | `phase_u_absolute_4988928_seed820201_20260825` |
| config SHA-256 | `58e0302c82de0e267f28679dbe680fb5ef4a1538ffbfcd6ac63904cf6c2bc210` |
| seed | 820201 |
| training transitions | 4,988,928 |
| natural fixed evaluation | 192 |
| forced-RSI diagnostics | 88 |
| Brax evaluation | 0 |
| total environment transitions | 4,989,208 |
| checkpoints | 0, 245,760, 983,040, 2,506,752, 3,981,312, 4,988,928 |
| final checkpoint restored | yes |
| final payload SHA-256 | `1125d9edbec3cd31ec08bbe5cf88777e84974044ba980e3531fdb938f34596fd` |
| formal status | completed |
| strict provenance | exit 0 |
| reported training walltime | 110.74 s |

最终 interaction 总数严格等于 `4,988,928 + 192 + 88`。视频与绘图只重放保存
状态，没有增加环境交互。

## 7. 所有里程碑结果

### 7.1 自然起步 promotion panel

| checkpoint | Apex | 物理失败 | 总 ticks | 最大 root z | jump/ascent/height | 终止原因 | action saturation |
|---:|---:|---:|---:|---:|---:|---|---:|
| 245,760 | 0/8 | 8/8 | 112 | 0.402 m | 0/0/0 | 8 pitch limit | 0.0% |
| 983,040 | 0/8 | 8/8 | 24 | 0.150 m | 0/0/0 | 8 illegal wheel contact | 8.33% |
| 2,506,752 | 0/8 | 8/8 | 24 | 0.150 m | 0/0/0 | 8 illegal wheel contact | 33.33% |
| 3,981,312 | 0/8 | 8/8 | 16 | 0.150 m | 0/0/0 | 8 illegal wheel contact | 25.0% |
| 4,988,928 | 0/8 | 8/8 | 16 | 0.150 m | 0/0/0 | 8 illegal wheel contact | 50.0% |

第一个里程碑还能存活平均 14 ticks，但因 pitch limit 失败。之后退化为平均
3 ticks，最后两个里程碑平均只存活 2 ticks。训练步数增加没有改善自然任务，
反而让动作饱和和立即非法接触更严重。

八个自然 seed 的轨迹在最终 panel 完全相同，因为当前 natural reset 不使用随机
扰动。它们证明确定性复现失败，不构成八种不同初始条件下的稳健性测试。

### 7.2 强制空中 RSI diagnostic panel

| checkpoint | Apex | 物理失败 | 总 ticks | 最大 root z | jump/ascent/height | 终止原因 |
|---:|---:|---:|---:|---:|---|
| 245,760 | 8/8 | 0/8 | 24 | 2.157 m | 8/8/8 | 8 Apex success |
| 983,040 | 8/8 | 0/8 | 16 | 2.144 m | 8/8/8 | 8 Apex success |
| 2,506,752 | 8/8 | 0/8 | 16 | 2.144 m | 8/8/8 | 8 Apex success |
| 3,981,312 | 8/8 | 0/8 | 16 | 2.140 m | 8/8/8 | 8 Apex success |
| 4,988,928 | 8/8 | 0/8 | 16 | 2.138 m | 8/8/8 | 8 Apex success |

该 panel 的初始状态本身已经位于窗口内、高于 0.5 m，并带有 `vz=0.8..1.2`
的向上速度。因此只要随后速度转负，Apex 条件就成立。最终 RSI 代表轨迹在
0.04 s 内成功，但终止步关节功率约 3,173 W、joint-energy 分项约 -126.94，
未裁剪总奖励 -80.95，仍被裁剪为 -50。这个“成功”主要验证事件/终止链路，
不能被解释为优质或稳定的空中控制。

## 8. PPO 优化指标

203 个训练 block 均有有限指标，checkpoint 可以恢复，说明没有 NaN、崩溃或
序列化错误。但“数值可运行”不等于“PPO 没问题”。

| transition | KL | policy mean std | value loss | policy loss |
|---:|---:|---:|---:|---:|
| 24,576 | 276.797 | 0.7525 | 31.498 | +0.2245 |
| 49,152 | 0.0152 | 0.7979 | 51.950 | -0.0214 |
| 245,760 | 0.0320 | 0.5330 | 0.0568 | -0.0494 |
| 983,040 | 0.0534 | 0.3551 | 0.1239 | -0.0898 |
| 2,506,752 | 0.0496 | 0.4535 | 0.3609 | -0.0893 |
| 3,981,312 | 0.0659 | 0.6023 | 0.4764 | -0.1071 |
| 4,988,928 | 0.1151 | 0.2283 | 0.000215 | -0.0681 |

首 block 的 KL 混入了 cold observation normalizer 更新前后坐标系差异，不能
单独理解为策略真的移动了 276。为确认这一点，训练前做过一次固定率 smoke；
策略输出仍有限。两个隔离的 adaptive-KL smoke 反而把 loc/std 和 KL 放大到
不合理范围，因此 adaptive 实验被拒绝并从活动代码/配置删除，没有作为 formal
checkpoint 输入。

即使排除首 block，KL 最高仍到 61.279，最终为 0.115，说明过程中存在较大的
更新尖峰。更关键的是 loss 最终很小，但自然任务性能稳定为零。这表明当前主要
问题不是 GPU、网络维度或 minibatch 实现错误，而是训练分布、奖励裁剪和地面
动作之间形成了错误最优解。

## 9. 最终自然轨迹逐步诊断

最终代表轨迹为 seed 930001，共 2 次环境转移、3 个保存状态/视频帧：

| t | x / z | vx / vz | action `[s,d,h,k]` | ctrl `[s,d,h,k]` | unclipped / clipped | 事件 |
|---:|---|---|---|---|---|---|
| 0.00 | 1.500 / 0.150 | 2.000 / 0.000 | `[0,0,0,0]` | `[0,12,-1.2,2.5]` | 0 / 0 | reset |
| 0.02 | 1.540 / 0.148 | 1.985 / -0.176 | `[-.006,-.002,-.088,.389]` | `[-.005,11.970,-1.209,2.5]` | 5.527 / 5.527 | ongoing |
| 0.04 | 1.579 / 0.124 | 0.914 / -0.086 | `[1,-1,1,-1]` | `[.800,.002,.500,-1.5]` | -103.607 / -50 | illegal wheel contact |

终止 tick 的主要分项：

- joint energy `-47.559`；hip/knee 力达到 `+50/-50 N m`；
- illegal contact `-30`；physical failure `-30`；
- action magnitude `-0.600`；pitch rate `-0.218`；
- 正向分项 roll/pitch/yaw/survival 合计约 `+4.77`。

策略在一个控制周期内把 hip 从约 -1.21 指向 +0.50，把 knee 从 +2.50 指向
-1.50，同时把 steer/drive 推到相反饱和值。这产生高关节速度和非法车轮接触。
未裁剪奖励为 -103.607，但 PPO 只看到裁剪后的 -50（再缩放为 -5），因此
`-53.607` 的额外严重程度对梯度不可见。这是下一轮需要重点审查的奖励信息损失。

## 10. 最终产物

自然起步和强制 RSI 的最终产物完全分离，均通过实际解码、样本数、lineage 和
SHA-256 校验。

### 自然起步

- MP4：`evaluations/transition_4988928/representative.mp4`，3 帧，SHA-256
  `a32f6db81769c12d8d20223c7426bb2e18253bffa76fbe93a7eb28253ed2bd46`。
- PNG：`evaluations/transition_4988928/representative_diagnostic.png`，SHA-256
  `a57b0c8047511ff8a56c531dd8b264080eaa02f4a0f11daf8b6d8583c43cf32e`。
- NPZ：`evaluations/transition_4988928/representative_diagnostic.npz`，SHA-256
  `2c342b9daac60da0143b1e1abc26cf0248bab2f2004b1f6593bef5e38377ee10`。

### 强制空中 RSI

- MP4：`diagnostics/airborne_rsi/transition_4988928/representative.mp4`，3 帧，
  SHA-256 `240940e3a5e15ba741f4b627efee0aeba5488384e19818425e2b6b3e455fc69a`。
- PNG：`diagnostics/airborne_rsi/transition_4988928/representative_diagnostic.png`，
  SHA-256 `1250a8ca765b48a74aea74d25d05048f78f18e5f5038dce43369df5650ee8c07`。
- NPZ：`diagnostics/airborne_rsi/transition_4988928/representative_diagnostic.npz`，
  SHA-256 `a2167161e7116aefcd5dc2478ff802d6ef21ec107f6a843a8a85310ad3bdd462`。

视频只有 3 帧不是渲染丢帧：每回合确实在 2 个控制步后终止，合同要求
`frames = transitions + 1`，因此应当正好为 3。

## 11. 科学判断

1. PPO 工程管线是可运行和可恢复的，没有发现网络输入输出维度、GPU batch
   形状或 checkpoint 加载错误。
2. 当前策略学习失败是确定的：自然起步所有 checkpoint 都是 0 任务进展和
   100% 物理失败。
3. 5% RSI 让“空中转为下降”事件大量进入训练，但没有把价值传播到窗口前的
   地面推进和起跳准备；natural/RSI 之间仍存在巨大状态分布断层。
4. RSI 的 100% Apex 不能证明学到有意义的跳跃控制，因为 reset 已经提供高度
   和向上速度，且终止步动作/能量非常激烈。
5. 绝对 knee 映射在 keyframe 上界处退化，加上策略可在单步跨越完整关节范围，
   是立即失败的重要候选原因；但在做因果干预前不能只凭相关性断言它是唯一原因。

## 12. 接下来怎么做

不要加载 final checkpoint 继续训练，也不要把 RSI 8/8 当作专家。下一轮先做
一个不超过正式训练预算的诊断/小规模 A/B：

1. **动作因果干预**：固定同一个自然 reset 和 final policy，只分别冻结 steer、
   drive、hip、knee，记录前 10--30 ticks 或终止；确认哪个通道导致非法接触。
2. **地面保持基线**：用零动作、手工 keyframe 保持动作、参考轨迹起始动作跑同样
   的自然 trace。若这些都触发非法接触，先查几何/初态；若只有策略失败，则是
   policy/action 学习问题。
3. **奖励未裁剪审计**：对同一 frozen minibatch 分解 actor/critic 梯度，比较
   clipped 与 unclipped terminal loss，判断 `-50` 裁剪是否抹平了严重摔倒差异。
4. **桥接 RSI，而不是简单增加 RSI 比例**：若需要方法修改，设计从地面窗口前、
   窗口内低高度、早期离地到当前空中状态的分层 reset curriculum，并让每层有
   独立 panel；不能把 5% 直接调成 50% 后继续称为同一实验。
5. **动作速率约束 A/B**：比较当前全范围绝对目标与受限目标变化率/残差动作，
   目标是先通过“自然起步不立即失败”的 retention gate，再考虑百万步训练。
6. **promotion gate**：至少要求自然 panel 不立即物理失败、能稳定到达窗口、出现
   ascent/height，再谈 Apex；只有独立自然起步评估可以决定 expert promotion。

完成以上诊断前，继续追加到 1,000 万步很可能只会强化当前立即失败策略，而
不会弥补地面到空中的 credit-assignment 断层。
