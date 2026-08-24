# JIT Phase U 正式训练与实验分析报告

## 1. 执行结论

本轮已经完成一条可审计的 `Propulsion-Ascent` 正式 PPO 训练：

- 成功 run：`phase_u_formal_998400_seed820101_20260824_retry1`
- 训练 seed：`820101`
- 训练 transitions：精确 `998,400 = 39 x 25,600`
- 固定评估 transitions：`904`
- Brax 内建评估 transitions：`0`
- diagnostic transitions：`0`
- 六个 checkpoint：`0 / 102400 / 256000 / 512000 / 742400 / 998400`
- 五个固定评估 panel：每个声明八个 held-out seed
- final checkpoint：已经重新加载并完成有限四维动作推理
- final video：`21` 个环境 transition，`22` 个保存状态，`22` 个编码帧
- 严格 provenance verifier：通过

关键身份：

| Artifact | SHA-256 |
|---|---|
| authoritative XML | `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192` |
| reference CSV | `612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f` |
| resolved formal config | `cd17f3c8746c74332c652ca63ee876db9b629d9b7dc280f23515824c42c2ad58` |
| final checkpoint payload | `26abb3ff24a15c8fe750cacf990d87510a787a2e72d6b48eb37a674ec2d355e8` |

工程闭环是成功的，但科研结果是否定的：五个评估 milestone 均为
`0/8` Apex success、`8/8` physical failure、统一终止于 `roll_limit`。
因此该 final policy **不能冻结为 Phase U expert**，也不能开始 Phase D
snapshot/continuation 工作，更不能支持 learned Tube、统一 PPO 或 JCE/JEL
结论。

## 2. 范围与不变合同

本轮只实现和运行 Phase U 正式训练，没有实现 Phase D、continuation
label、`V_up`/`V_down`、learned soft Tube、unified Tube-RSI PPO 或独立安全
认证。训练前后以下合同均未改变：

| 合同 | 保持值 |
|---|---|
| 权威模型 | `assets/orange_bike_4kg_horizontal.xml` |
| payload | 2 kg；XML 路径中的 `4kg` 只是历史文件名 |
| 仿真/控制步长 | `0.005 s / 0.020 s`，每控制 tick 四个 substep |
| 动作顺序 | steer, rear-wheel drive, hip, knee |
| hip/knee force limit | `[-50, 50] N m` |
| Actor 输入 | 三帧真实 FIFO，`3 x 27 = 81` |
| Actor 网络 | `81 -> 256 -> 256 -> 256 -> 8`，输出四维 tanh-normal 参数 |
| deterministic Actor 输出 | 四维动作 `[steer, drive, hip, knee]` |
| critic 输入 | `114` 维 privileged state |
| critic 网络 | `114 -> 256 -> 256 -> 256 -> 1` |
| 并行环境 | 1,024 |
| horizon | 200 control ticks |
| PPO block | unroll 25、batch 128、8 minibatch、1 update，共 25,600 transitions |
| PPO 超参数 | learning rate `1e-4`、entropy cost `1e-3`、discount `0.995`、GAE `0.97`、clip `0.1`、global grad norm `0.75` |

formal 与 smoke 配置有自动回归测试，除了 schema、训练预算、callback 数、
training seed 和 checkpoint/evaluation schedule 外，物理、reset、事件、Apex、
reward 和 PPO 布局逐字段相同。

## 3. 具体代码修改与原因

### 3.1 严格 formal config

新增 `JIT/configs/phase_u_formal.json`，并在 `config.py` 增加
`FormalTrainingConfig`。入口在创建 run 目录之前拒绝：

- 不是精确 998,400 transitions；
- 不是 39 个对齐 block；
- 不是精确六个 checkpoint；
- 固定评估不是五个非零 checkpoint；
- held-out seed 不是 `920001..920008`；
- training seed 不是 `820101`；
- 声称 bit-exact/optimizer-exact resume；
- smoke schema 混入 formal 字段。

原因是正式运行的计算成本、评估时点和恢复语义必须在任何环境交互之前
冻结，不能由命令行或运行中逻辑悄然改变。

### 3.2 run 状态机与真实 resume 语义

`provenance.py` 增加 `predeclared -> running -> terminal` 状态迁移，并记录
PID、UTC 时间、absolute start、target、parent checkpoint、segment seed 和
resume semantics。恢复只能叫
`parameter_warm_start_optimizer_reset`：当前 Brax 公共接口只能恢复 observation
normalizer、Actor 和 critic 参数，不能恢复 optimizer、minibatch RNG 或完整
TrainingState。

原因是参数热启动和 bit-exact continuation 是不同科研语义，二者不能混称。
本次成功 run 是 fresh 单进程运行，没有使用 warm resume。

### 3.3 完整固定评估轨迹

`evaluation.py` 新增每 seed 独立的 NPZ/JSON artifact，保存：

- qpos、qvel、ctrl、Actor action；
- reward 和每个 reward component；
- 所有 runtime metric；
- terminated、truncated、end code、success、physical failure、timeout；
- transition 数、state 数和 NPZ SHA-256。

原因是 video 只是可视化，不能承担数值证据；评估必须在 done 后立即停止，
并满足 `captured states = transitions + 1`。

### 3.4 formal runner 与 absolute callback controller

新增 `formal_training.py`，把 Brax segment-relative step 转为 absolute training
transition，只在声明的 milestone 保存 identity-bound checkpoint 和运行固定
评估。异常会闭合为 `engineering_error`，不会自动改变 reward 或 PPO 参数。
final checkpoint 必须重新加载，并产生有限四维 deterministic action，run 才能
关闭为 completed。

原因是一次正常 formal run 必须保持 optimizer 连续，而 artifact/ledger 必须以
绝对 transition 对齐。独立 controller 也避免破坏已经验证的 smoke runner。

### 3.5 CLI、preflight 与 verifier

稳定入口 `train_phase_expert.py` 现在要求恰好一个 `--smoke` 或 `--formal`；
`--restore-checkpoint` 只允许 formal。preflight 只静态加载 formal config，不会
启动正式训练。严格 verifier 会重新检查：

- authoritative XML/reference/config identity；
- 所有 checkpoint payload hash；
- 五个 panel 的精确 seed 集、trace hash 和 transition 总账；
- final checkpoint restore；
- final video 的 state/frame 计数和 artifact 路径边界。

原因是“文件存在”不等于证据闭环；每个计数和 hash 必须能由 run 目录内部
证据重算。

### 3.6 首次正式启动暴露的 callback 问题

第一次 run `phase_u_formal_998400_seed820101_20260824` 在首个 25,600-transition
block 后关闭为 `engineering_error`。根因不是物理、reward、GPU 或 PPO 数值：
Brax 在 `log_training_metrics=True` 时通过 `jax.debug.callback` 让
`EpisodeMetricsLogger` 在 epoch 内调用同一个 `progress_fn`，其时序早于
block 结束后的 `policy_params_fn`，违反 controller 的有序回调合同。

修复只把 formal runner 的 `log_training_metrics` 设为 `False`：

- 每 block 的 PPO loss、KL、SPS progress 仍由 Brax 正常回调；
- checkpoint 固定评估仍保存终止原因和完整轨迹；
- 物理、reward、网络、PPO 更新和预算均未改变。

失败发生在 policy callback 更新 ledger 之前，原 status 错记为 0。根据 Brax
源码中单 epoch 恰有一个 training step、logger threshold 和 callback step 均为
25,600，已在忽略的 `accounting_correction.json` 中显式更正为 25,600。该 run
只有 transition-0 checkpoint，不能作为 warm-start parent；成功 run 因此使用
新 run id 从头 fresh 训练。

## 4. 验证与 Git 证据

正式训练前完成并推送了以下聚焦 commit：

| Commit | 内容 |
|---|---|
| `67bc91d` | formal design、budget、milestone、claim boundary |
| `30573f8` | formal config、runner、trace、verifier、CLI 和 TDD tests |
| `fd534b9` | 记录 source-push-before-launch gate |
| `fd3372c` | 修复 Brax in-epoch progress callback 冲突 |

训练使用的 GitHub branch 是 `origin/agent/two-phase-soft-tube`，成功 run 启动前
远端精确指向 `fd3372c1e1a7ffc6099ae366e8302e7552474773`。

最新 JIT preflight 结果：

- non-GPU：`106 passed, 5 deselected`；
- GPU：`5 passed`；
- retained smoke provenance：通过；
- legacy `dvgc` import AST audit：通过。

根仓库兼容性结果为 `1070 passed, 1 deselected`。唯一 deselect 是用户当前未
提交的 relative-x diagnostic test 使用 `mode=`，但同一用户修改中的 production
function 尚未接受该参数；JIT 没有修改这两条外部路径。

## 5. Interaction accounting

### 5.1 成功 run

| 类别 | Transitions |
|---|---:|
| PPO training | 998,400 |
| Brax evaluation | 0 |
| 固定 held-out evaluation | 904 |
| diagnostic | 0 |
| 成功 run 总计 | 999,304 |

### 5.2 本次 formal 任务总成本

| 类别 | Transitions |
|---|---:|
| 首次失败 run 的一个已执行 block | 25,600 |
| 成功 run | 999,304 |
| 本次 formal 任务总计 | 1,024,904 |

此前独立 engineering smoke 的 25,631 transitions 是先前交付成本，不混入本次
formal run ledger。所有 rendering 都重放保存状态，消耗 0 环境 transitions。

## 6. PPO 优化趋势

共保存 39 条、每 block 一条 metrics。Brax 报告的纯 training walltime 为
15.12 秒；它不包含完整进程初始化、checkpoint、固定评估和 video 时间。

| Transition | KL | Policy loss | Value loss | Total loss | Mean policy std |
|---:|---:|---:|---:|---:|---:|
| 25,600 | 357.8810 | 0.3269 | 32.1123 | 32.4367 | 0.7174 |
| 102,400 | 0.00131 | -0.00162 | 23.6940 | 23.6899 | 0.6961 |
| 256,000 | 0.04838 | -0.07260 | 0.8526 | 0.7775 | 0.7385 |
| 512,000 | 0.06943 | -0.03866 | 0.000259 | -0.03896 | 0.4416 |
| 742,400 | 0.16515 | -0.06607 | 0.000013 | -0.06471 | 0.3614 |
| 998,400 | 0.13177 | -0.01810 | 0.000028 | -0.01576 | 0.3114 |

全程 KL 中位数为 0.07209，最小 0.00105，最大 357.881。首个 update 的巨大 KL
随后立即降到 0.0111、0.00207 和 0.00131，没有 NaN/OOM，也没有自动触发参数
改变。后半程 KL 再升至约 0.13–0.17，同时 policy std 逐渐收缩。critic loss 降到
接近零只说明当前回报模式变得容易预测；结合全部评估失败，它更像是 critic
拟合了一个稳定失败吸引域，而不是学到了 jump。

## 7. 五个固定评估 panel

每个 milestone 都运行声明的八个 seed，done 后立即停止。

| Transition | 每 rollout ticks | Mean return | Window | Liftoff | Ascent | Apex | Physical failure | Saturation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 102,400 | 30 | -217.894 | 8/8 | 0/8 | 0/8 | 0/8 | 8/8 roll_limit | 0.0000 |
| 256,000 | 25 | -93.050 | 8/8 | 0/8 | 0/8 | 0/8 | 8/8 roll_limit | 0.0000 |
| 512,000 | 17 | -218.017 | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 roll_limit | 0.0000 |
| 742,400 | 20 | -239.142 | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 roll_limit | 0.0125 |
| 998,400 | 21 | -262.387 | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 roll_limit | 0.2857 |

最好的 checkpoint 是 256,000：它仍然没有 liftoff/Apex，但 return 最不差、
能到达 window、角速度和 pitch 相对较小。此后训练反而退化：512,000 起 policy
在进入 window 前已经触发 roll limit，final return 比 256,000 差约 169.34。

所有 milestone 的 clearance、liftoff、stable airborne、ascent 和 Apex reward
component 总和都为零。final 八条 rollout 的 rate penalty 总和约 -1834.10，
physical-failure penalty 总和 -240，而 drive reward 仅约 +34.71；这表明最终策略
不是“差一点跳起”，而是明显进入高角速度、横滚主导的失败模式。

## 8. final policy 行为诊断

final seed 920001 与其他声明 seed 都在第 21 tick 以 `roll_limit` 结束。四维
action 的 rollout 均值约为：

```text
steer = -0.889
drive = +0.831
hip   = -0.628
knee  = +0.865
```

终止动作约为 `[-0.991, +0.994, -0.931, +1.000]`。从第 7 tick 左右开始 steer、
drive 和 knee 已接近饱和；roll 从约 -0.153 连续恶化到 -0.709 rad，最终在
obstacle relative-x 约 1.290 时失败，尚未进入配置 window 上界 1.25。final
maximum angular speed 约 8.90 rad/s，action saturation fraction 为 0.2857。

这给出一个很强的机制性线索：策略利用 drive/knee 并形成持续单侧 steer，导致
横滚先于跳跃事件发生。继续单纯增加 PPO transitions 很可能只会强化这个失败
吸引域。

## 9. 重要统计限制

虽然 panel 使用八个互斥 seed 标签，但当前 `env.reset(rng)` 没有使用 `rng`，
reset 始终来自同一 keyframe 和同一 2.0 m/s 初速度；deterministic policy 也不
采样。因此八个 rollout 实际上是同一初始条件的重复执行，绝大多数 milestone
得到完全相同的长度、动作和终止结果。它们证明确定性重复性，但**不能证明对
初始条件或扰动的 seed 泛化**。

因此本报告不对 8 条重复轨迹计算置信区间，也不把 `0/8` 或 `8/8` 解释为八个
统计独立试验。formal verifier 能证明 seed 身份和 artifact 完整性，但不能把
未被环境使用的 seed 变成独立条件。若要建立真正的 held-out 泛化协议，需要
显式批准 reset 扰动范围；这会改变 reset 分布，不能在本轮自动加入。

其他限制：

- 只有一个 training seed `820101`；
- checkpoint 不含 optimizer/RNG，不能 bit-exact resume；
- 为解决 Brax callback 冲突，训练内 episode logger 被禁用；仍保留每 block
  loss/KL/SPS 和 milestone 完整轨迹，但没有训练分布的 episode-terminal 直方图；
- reference CSV 仍只是离线弱先验，没有进入 `env.step` 或 PPO reward。

## 10. 决策与下一步

### 10.1 立即决策

1. 不冻结 final 或 256,000 checkpoint 为 Phase U expert。
2. 不开始 Phase D、snapshot continuation、`V_up`/`V_down` 或 Tube 工作。
3. 不追加更长 PPO，不因“预算跑完”而宣称 learnability。
4. 保留六个 checkpoint，256,000 仅作为“失败中最接近可诊断状态”的候选，
   不是 promotion candidate。

### 10.2 下一轮优先做零训练成本的 frozen-policy 诊断

建立新的、预声明 interaction ceiling 的 diagnostic run，对六个 frozen
checkpoint 进行以下 action intervention：

1. 原始 deterministic policy；
2. 仅强制 `steer=0`；
3. 限制 steer 绝对值但保留其他动作；
4. 分别把 hip 或 knee 置为 neutral；
5. 限制 drive，检查高速度是否是横滚放大器。

每个 intervention 保存前 10–30 tick 的 action、roll/pitch/angular speed、
relative-x、support/clearance 和逐项 reward。首要问题是：去掉持续单侧 steer 后，
策略能否至少到达 window、产生 liftoff 或延迟 roll failure。这个实验诊断 frozen
policy，不改变训练 reward 或物理。

### 10.3 在下一次 PPO 前补足观测与梯度证据

1. 增加一个与 block ledger 分离的 episode-metric adapter；只把 `episode/*`
   callback 写入单独 JSONL，不能推进 checkpoint transition。
2. 对首个 update 的 Actor/critic gradient norm、clip 前后 norm、四个动作头的
   gradient contribution 做固定 minibatch 分解。
3. 检查 observation normalizer 后 steer 相关输入及三帧 history 是否出现偏置。
4. 比较 transition 0、102,400、256,000 和 final 的 Actor distribution loc/std，
   确定单侧 steer 偏置在何时形成。
5. 检查 per-tick unclipped component sum 与最终 reward clamp，判断巨大 rate
   penalty 是否因 total clipping 失去相对梯度分辨率。

### 10.4 需要用户明确批准的研究改动

以下不是本轮自动修复，应在诊断结果后单独设计和批准：

- 让 reset seed 真正控制有限的初速度、姿态或关节扰动，从而形成独立 held-out
  条件；
- 对 steer 增加 straightness 约束、mask 或 curriculum；
- 调整 rate/attitude penalty 的归一化、clamp 或 event gating；
- 改变 PPO learning rate、clip、entropy 或更新预算。

批准任何一项前都应先说明它改变的是 reset 分布、动作可达集、reward meaning
还是 optimizer behavior，并建立新的 config hash 和 run id。

### 10.5 再训练的最小进入条件

只有在 frozen intervention 和 gradient/reward audit 给出明确机制后，才建议做
一个最多 102,400-transition 的 bounded pilot。pilot promotion gate 至少要求：

- 真正有扰动差异的 held-out 初始条件；
- 多条合法 window reach；
- 至少出现非零 liftoff 和 ascending；
- physical failure 不再是 100%；
- 不出现持续单侧 steer/多动作饱和；
- 所有 checkpoint、trace 和 interaction ledger 再次闭合。

若 bounded pilot 仍为 0 liftoff/100% roll failure，应暂停 PPO，回到 reward、
reset 和 actuation semantics 设计，而不是扩大预算。

## 11. 证据位置

成功 run 的 machine-readable 证据位于：

```text
JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824_retry1/
```

关键文件：

- `run_manifest.json` / `resolved_config.json` / `status.json`
- `metrics.jsonl`
- `formal_report.json`
- `checkpoints/transition_*/identity.json` 和 `payload.pkl`
- `evaluations/transition_*/summary.json`
- `evaluations/transition_*/seed_*.json` 和 `seed_*.npz`
- `evaluations/transition_998400/representative.mp4`
- `evaluations/transition_998400/video_report.json`

失败启动及账本更正保存在：

```text
JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824/
```

这些 run artifacts 全部被 Git 忽略；本报告、代码、配置、测试和过程文档全部在
`JIT/` 下并由同一分支管理。
