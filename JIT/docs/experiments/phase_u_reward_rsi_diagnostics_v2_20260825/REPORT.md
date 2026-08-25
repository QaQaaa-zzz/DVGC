# Phase U v2 奖励、RSI 与诊断重建报告

## 1. 结论

本轮完成的是训练前的环境契约重建，不是新一轮 PPO 实验。活动版本已经从
v1 的窗口/离地/净空塑形改成用户批准的 v2 平面跳逻辑，并通过 Host、GPU、
视频、哈希和旧证据兼容性验证。未启动 PPO，因此目前没有可声称为“v2 最优
模型”或“v2 最终视频”的产物。

最重要的结论是：旧 v1 模型的输入为 81/114，活动 v2 输入为 76/106，奖励和
终止含义也发生变化。旧模型只能作为历史失败证据查看，不能恢复到 v2 继续
训练。

## 2. 为什么修改

旧 Phase U 契约存在四个方法层面的歧义：

1. 高空奖励与起跳许可没有严格绑定；
2. 前后轮支撑和结构净空属于近似推断，不能作为可靠 Apex 门；
3. 终止中混有平台后段语义，可能在 Phase U 成功前截断轨迹；
4. 视频只有物理画面，难以判断总奖励是否由错误分项、裁剪或姿态异常造成。

v2 因此只保留 Propulsion-Ascent：合法访问起跳窗口、达到高度、确实向上运动，
然后开始下降即为 Apex 成功并终止。

## 3. 具体代码修改与理由

| 模块 | 修改 | 理由 |
|---|---|---|
| `config.py` 与两个 JSON | 引入 v2 schema、RSI 范围、root-x 起跳窗口、速度/高度阈值和参考奖励系数；删除 ApexConfig、平台余量和目标/减速字段 | 让活动配置只表达当前任务，旧字段不能静默生效 |
| `semantics.py` | 新增一次性 `jump_signal`、visited/consumed、ascent/height/Apex 事件；删除支撑/净空/平台越界门 | 精确实现 `0 -> 1 -> 0` 且离开后永久关闭 |
| `rewards.py` | 对齐参考文件的 roll/pitch/yaw 分段函数、速度高斯、生存、高度、动作、pitch rate、关节能耗和终止代价；保存未裁剪总和 | 使每个数值可手算、可测试、可诊断 |
| `env.py` | 训练 reset 使用 95% 自然 + 5% 空中 RSI；新增强制自然 reset；记录完整状态、奖励、事件、力和能量 metrics | RSI 帮助稀有跳跃信号，高层评估不被 RSI 成功污染 |
| `observation.py` | 删除前/后轮支撑，结构净空改为 root height；jump signal 只在历史外追加一次 | 不把不准的支撑状态输入网络，也不重复 jump signal |
| `checkpoint.py` | identity 增加 Actor task fields | 明确拒绝 81/114 的旧 checkpoint |
| `evaluation.py` | 汇总改成 jump-zone、height、ascent、Apex 和 maximum root height | 报告与新任务语义一致 |
| `diagnostics.py` | 新增全轨迹 PNG、完整 NPZ、同步视频遥测面板和 SHA-256 | 能同时检查每项奖励、总奖励、位置、姿态、速度、动作和终止原因 |
| `video.py` | 每个物理帧右侧拼接同 tick 遥测；不调用环境 step | 画面和数值严格同步且不增加交互成本 |
| `provenance.py` | v1/v2 分支验证；v2 强制检查 MP4/PNG/NPZ 路径与哈希 | 保留旧证据可审计，同时提高新证据完整性 |

## 4. 活动奖励函数的精确数值

角度奖励先把弧度转为角度，并使用绝对角度。

- roll 系数 3.0：0° 为 3；5° 为 0；15° 为 -3。
- pitch 系数 1.0：0/3/8/10/20° 分别为 1/0.9/0.5/0/-1。
- yaw 系数 0.3：0/8/15/25/50° 分别为 0.3/0.15/0.06/0.015/0。
- speed：`0.2 * exp(-0.5*((vx-3.5)/0.5)^2)`。
- survival：每个非 reset 转移为 1.5。
- height：仅当前 `jump_signal=1` 时有效；z=0.34/0.35/0.50/0.80/0.81 m
  分别为 0/20/30/12/8。
- action smoothness：`-1.5e-4 * sum((a_t-a_(t-1))^2)`。
- action magnitude：`-0.15 * sum(abs(a_t)^1.5)`。
- pitch rate：`-0.01875 * omega_pitch^2`；Phase U 的 roll/yaw rate 分项为 0。
- joint energy：`-2 * 0.02 * (|tau_hip*qdot_hip| + |tau_knee*qdot_knee|)`。
- Apex success +50；illegal contact -30；首次 physical failure -30；timeout -10。
- 所有分项先求和，另存 `reward/unclipped`，然后总奖励裁剪到 [-50, 50]；
  PPO 再按配置乘 0.1，诊断中三种总值全部保留。

不存在 target position、direction、target reached、platform、deceleration、
drive、liftoff、stable-airborne、clearance 或 apex-progress 奖励。

## 5. Jump signal、RSI 与终止

起跳窗口为闭区间 `2.5 <= root_x <= 3.1 m`。reset 在窗口前得到 signal=0 且
尚可触发；在窗口内得到 signal=1；在窗口后得到 signal=0 且已消费。首次离开
窗口后 signal 永久为 0，即使倒车重新进入也不会再打开。

训练 reset 的 5% RSI 独立采样：x [2.7,2.9]、z [1.8,2.2]、vx [1.8,2.2]、
vz [0.8,1.2]。正 vz 防止高空 reset 在第一步因自由落体直接获得 Apex。
正式 held-out panel 强制调用自然 reset，不使用这一混合分布。

Apex 首次满足以下条件即成功终止：历史上访问过起跳窗口、历史上出现
`vz>=+0.05 m/s`、历史上达到 `z>=0.5 m`、当前 `vz<=-0.05 m/s`，并且当前没有
物理失败。不存在后续平台减速分支。

## 6. 网络输入输出与维度

Actor 每帧 25 维：重力 3、角速度 3、加速度 3、steer/hip/knee 位置 3、对应
关节速度 3、前后轮速度 2、上一动作 4、估计前向速度 1、障碍相对 x 1、root
height 1、history-valid 1。三帧共 75 维，当前 jump signal 在历史外追加一次，
得到 76 维。

Actor 是 `76 -> 256 -> 256 -> 256 -> 8`。8 个输出是四维 tanh-normal 动作
分布的参数；确定性评估输出四维动作，顺序为 steer、rear-wheel drive、hip、
knee。

critic 输入为完整 Actor 76 维加 30 维 privileged state，共 106 维，网络为
`106 -> 256 -> 256 -> 256 -> 1`。因此 critic 一定能看到 jump signal，而且
signal 在 Actor 历史中没有重复。

## 7. 诊断产物

每个 v2 representative trace 生成：

- `*.mp4`：左侧物理画面，右侧为同 tick 的所有奖励分项、未裁剪/裁剪/PPO
  总奖励、位置、姿态、速度、动作、控制、关节力/功率、事件和终止状态；
- `*_diagnostic.png`：全轨迹 5x2 dashboard；
- `*_diagnostic.npz`：所有原始 qpos/qvel/action/ctrl、奖励、metrics 和终止数组；
- JSON：样本数、帧数、fps 以及 MP4/PNG/NPZ SHA-256。

保存状态数必须等于环境转移数加一，编码帧数必须等于保存状态数。绘图和编码
只重放已保存状态，不会调用 `env.step`。

## 8. 验证结果与实验分析

| 验证 | 结果 |
|---|---|
| reward/config/formal-method drift focused | 47 passed |
| diagnostics/evaluation/video/formal/provenance focused | 33 passed |
| complete non-GPU | 134 passed, 7 deselected |
| complete GPU | 7 passed, 134 deselected |
| JIT local preflight | exit 0 |
| complete repository compatibility | 1,100 passed, 1 exact user-dirty test deselected |
| retained v1 smoke provenance | 25,600 training + 31 diagnostic，hash/accounting 通过 |
| retained v1 formal provenance | 998,400 training + 904 fixed evaluation，全部历史 hash 通过 |

GPU 测试覆盖 1,024 个可复现混合 reset，空中样本数必须落在 20 到 85，且每个
样本的 x/z/vx/vz、signal 和 76/106 维输入均满足合同。短 rollout 的 qpos、qvel、
Actor/critic 输入和奖励全部有限。

这些结果证明的是实现与运行完整性，不证明奖励可学习。尤其是高度奖励只在
当前 signal=1 时存在；小车高速穿过 0.6 m 窗口时可获得奖励的时间较短，5% RSI
是否足以传播价值仍需通过新训练数据判断。

## 9. 下一步建议

1. 先提交并固定本轮 v2 源码，不恢复任何 v1 checkpoint。
2. 原始保守建议是先运行 25,600-transition engineering smoke；用户在完整复验后
   明确授权直接运行一次 fresh 998,400-transition v2 formal run，因此以该授权为准。
3. 分开统计自然 reset 与 RSI reset 的 jump-zone、height、ascent、Apex、物理
   失败率和 episode return；查看新 PNG/NPZ/视频，确认高度奖励只在 signal=1。
4. 若早期自然 panel 完全无法进入窗口，继续保存已声明里程碑证据，但不得把 RSI
   结果当作 natural promotion；完成后先检查动作分布、signal tick 数和失败原因。
5. promotion 必须仅依据 frozen natural-start held-out panel，RSI 成功不能计入。

本轮未修改 XML、碰撞几何、动作顺序、仿真步长或 PPO 拓扑宽度，也未改动
`JIT/` 外的用户文件。

## 10. 998,400 步 v2 formal 实验结果

用户在上述实现完成、测试和独立复审通过后，授权直接运行约 100 万步。实际
使用严格配置的 998,400 步，fresh 启动且没有恢复任何旧 checkpoint。

| 项目 | 结果 |
|---|---|
| source commit | `c55a5d0f7236ddc1217ac84743b149634f7629bf`，训练前已推送并与远端一致 |
| run ID | `phase_u_v2_formal_998400_seed820101_20260825` |
| training | 998,400 |
| fixed natural evaluation | 1,160 |
| Brax evaluation / diagnostic interactions | 0 / 0 |
| total interactions | 999,560 |
| final checkpoint restored | yes |
| final checkpoint SHA-256 | `1aef0f09e77e5100aade31d0add281eb10e7f005540cb67681839f2ef4a07a78` |
| strict provenance | exit 0 |
| decision | **NO_PROMOTION；不是 trained expert** |

六个 checkpoint、五个 panel、40 条逐状态 trace、最终 checkpoint restore、
MP4/PNG/NPZ 路径和哈希全部通过 verifier。步数完成只证明账本闭合，不代表
策略成功。

### 10.1 自然起点 panel

| checkpoint | 平均长度 tick | jump-zone | ascent | height | Apex | 物理失败 | 每回合 return | 最大 roll |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 102,400 | 30 | 8/8 | 0/8 | 0/8 | 0/8 | 8/8 | 9.286 | 36.62° |
| 256,000 | 28 | 8/8 | 0/8 | 0/8 | 0/8 | 8/8 | -5.363 | 37.59° |
| 512,000 | 32 | 8/8 | 0/8 | 0/8 | 0/8 | 8/8 | -59.94..-59.24 | 36.22° |
| 742,400 | 33 | 8/8 | 0/8 | 0/8 | 0/8 | 8/8 | 11.186..11.191 | 37.45° |
| 998,400 | 22 | 0/8 | 0/8 | 0/8 | 0/8 | 8/8 | -26.273..-26.267 | 38.72° |

所有 panel 的最大 root z 都是 reset 值 0.15 m，说明策略从未把车体抬高到
初始高度以上。早期四个 checkpoint 能进入 x=[2.5,3.1]，但进入窗口后已经没有
`vz>=0.05` 的上升证据，因此 height reward 和 Apex bonus 始终为零。最终策略
在 x=2.298 m 即触发 roll limit，发生了明显回退。

742,400 checkpoint 只能称为“按 return 和存活长度最不差”，不能称为最好跳跃
模型，因为它同样是 0 ascent、0 height、0 Apex、100% 物理失败。

### 10.2 PPO 优化趋势

- 39 个 block 全部记录；首个 KL 为 322.274，说明第一次更新非常激烈。
- 排除首个 block 后，KL 最小/中位/最大为 0.0020/0.0429/0.1995。
- value loss 全程最小/中位/最大为 0.000141/0.000741/13.362。
- policy mean std 从 0.7205 持续缩到最终 0.1218，最终最小动作 std 仅 0.0303。
- 最终 deterministic panel 的 action saturation fraction 为 0.1364；此前四个
  panel 均为 0。

优化 loss 变小并没有对应 natural task improvement。相反，最终策略的探索尺度
明显收缩，且自然起点表现从“能到窗口但侧翻”退化为“到窗口前侧翻”。

### 10.3 最终轨迹诊断

最终 representative trace 有 22 次环境转移、23 个状态和 23 个复合视频帧，
时长 0.44 s。数值 NPZ 含 77 个同步序列且全部有限。

- root x：1.500 -> 2.298 m；root z：0.150 -> 0.100 m；
- roll：最终 -38.72°，超过 35° 安全限制；yaw 最终约 -39.57°；
- jump signal 和 jump-zone-seen 全程为 0；
- 最终动作约 `[-0.291, 0.836, -1.000, 0.685]`；
- 对应控制约 `[-0.233, 22.029, -1.300, 1.749]`；
- 终止 tick 的未裁剪/裁剪奖励均为 -39.233，说明该 tick 没有被 [-50,50]
  裁剪隐藏；PPO scaled reward 为 -3.923；
- end code=3，即 `roll_limit`。

最终每回合的主要累计贡献约为：survival +33.0、pitch +20.72、yaw +3.33，
roll -34.15、joint energy -11.03、action magnitude -7.55、physical failure -30。
普通姿态/生存正奖励没有形成稳定起跳，height 分项始终为零。

最终证据：

- MP4：`evaluations/transition_998400/representative.mp4`，SHA-256
  `c87cfdb6d5b0e56d1af4f9cf64c37d5035d1a6da3bbceb989614308fe80b5a5e`；
- PNG：`evaluations/transition_998400/representative_diagnostic.png`，SHA-256
  `10587f1f3768bfbf56a7f556d6324bd1df65e5069a3344370a6ab65bc47e5f84`；
- NPZ：`evaluations/transition_998400/representative_diagnostic.npz`，SHA-256
  `356446907cd345a3723ba9bff3ee58f004bd1805d6f3098dd8f833ba4dafe188`。

### 10.4 科学限制

1. formal panel 强制自然 reset，因此没有 RSI 污染，但自然 reset 本身不使用
   held-out seed 产生状态扰动。八条轨迹不是八种独立初始条件；少量差异来自底层
   数值执行，不能当作稳健性覆盖。
2. 为避免旧 Brax callback 顺序问题，formal 训练关闭了 in-epoch episode logger。
   环境虽然记录 `reset/source_airborne_rsi`，本次 `metrics.jsonl` 只有 PPO loss/KL/
   分布统计，无法回溯 5% RSI 回合的成功率或优势贡献。
3. 因此本实验只能可靠下结论：所有 frozen natural panels 失败；不能从现有证据
   判断策略是否在空中 RSI 分布上学会了某种控制。

## 11. 下一步决策

不要继续追加 PPO 步数，也不要选择 final 或 742,400 作为 Phase U expert。下一轮
交互前优先完成两个只针对证据的设计：

1. 建立单独预声明、固定预算的 natural-vs-forced-RSI checkpoint panel，分别报告
   jump signal ticks、ascent/height/Apex、return 和终止原因；不得合并成功率。
2. 对 742,400 与 final checkpoint 做逐 tick Actor 输出、value/advantage 和动作干预
   对比，重点检查 hip 长期接近 -1、drive 接近高值、roll/yaw 发散的因果关系。
3. 若 RSI panel 成功而 natural panel 失败，问题是从地面到窗口/起跳的 credit
   assignment；若 RSI 同样失败，则先检查动作映射与奖励梯度，不应扩大预算。
4. 在用户批准前不改变当前参考奖励。若之后需要方法修改，优先讨论“窗口前没有
   可学习的起跳准备信号”和“policy std 过快塌缩”，并用独立小预算 A/B 验证。
5. 如需真正的 held-out 稳健性，必须另行批准 natural 初始扰动协议；当前八个 seed
   标签不能被描述为八种独立自然条件。
