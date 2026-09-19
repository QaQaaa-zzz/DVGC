# 5 cm 多状态 Tube pilot：运行与回传

## 当前决定

用户已取消 hip/knee-only 限制。允许对 steer、rear_wheel_drive、hip、knee 的合法输出施加有界动作扰动；不得通过位置/速度注入或外力制造到达状态。继续接受已记录的近地初始化和数值快照差异，不追加原快照重放研究。

上一批四策略比较已在生产环境完整运行：48 个评价分片，新增 197,604 次环境交互；TRAIN 的 pi_2 与并集分别覆盖 1,218 和 1,225 个 root 单元。该批只证明 pi_0 共同到达候选上的接续支持，并非所有历史 Tube 的完整并集。

## 服务器命令

```bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/run_dense_tube.py --gpu 0
```

需要保留上次已经完成的 `JIT/runs/policy_comparison/expanded_pi0_pi3_20260907`、原始 TRAIN catalog/snapshots、冻结 checkpoint 和 centerline。可以用 `--baseline` 指定上次输出目录。新目录默认为 `JIT/runs/dense_tube/pi0_5cm_pilot_v1`；不覆盖旧实验。

完成或执行失败时打印 `Return this file: .../results_to_send.zip`，把此 ZIP 回传即可。它包含图、表、执行比较、阶段日志和成本，不包含大 checkpoint/snapshot。原样再次运行可以续跑，已完成任务有身份和输出哈希检查。运行期间不要更新源码。更换 GPU 可以续跑；更换预算或基线必须使用新的 `--output-dir`。

## 一次运行自动完成的步骤

1. CPU 预检并锁定输入、四个冻结策略、源码、起点与原 centerline。
2. 从原 TRAIN 固定选出 16 个候选（upstream/downstream 各 8 个），对每个 evaluator 分别运行串行和 batch=8 设备端执行。比较候选身份、标签、终点类别、接续步数和最终阶段等。相同结果且测得更快才为该 evaluator 选择设备端路径；设备端失败、结果不同或较慢则自动选择串行。执行比较数据只作为工程记录，绝不混入新采集标签。这不是快照恢复等价性重放。
3. pi_0 从固定起点生成 16 条前向扰动轨迹，沿途按 5 cm 最近 x 单元、每阶段每单元首个真实帧取样。轨迹跨过的单元不填充、不插值。每条最多保存 32 个候选；达到采样上限即停止本次尝试，尚无落地见证的候选仍必须单独评价。
4. 四个 evaluator 对相同新候选评价；每个独立进程最多 128 个候选，设备端内部最多 8 个同时执行。保留首次有效落地终点、原 horizon 和每候选/每步随机键定义。
5. 输出四策略及并集的 5 cm 截面图，以及相对上次 TRAIN 成功支持的新增覆盖图。

该默认入口是**有界软件/采样 pilot**，只运行 TRAIN，不做新 PPO，不是已经完成的多 proposer 同预算论文主实验。以后扩大扫描前，应先看这个结果包。多个同轨迹帧有相关性，不能当作独立试验；与既有 CALIBRATION/ACCEPTANCE 的物理重叠单独报告，未授权自动进入训练支持。

## 精确采集语义

- 保留原 pi_0 实际 centerline 帧、起点及 corridor，不重算或插值 centerline。
- 5 cm 是新候选的空间取样间距及图表 x 截面间距；每个实际帧只落到一个最近单元，接收半宽为 2.5 cm。未观察到的单元保持空白。
- 默认动作窗口终点为 x=2.9 与 3.1 m，提前 0.15 m 激活动作偏置，强度为归一化动作增量 0.075，四通道分别正/负扰动。因此 2 个窗口 × 4 通道 × 2 符号 = 16 条轨迹。
- 动作根据当前状态重新推理，加偏置后裁剪到 [-1,1]。到达窗口终点后停止额外偏置，继续用同一个策略运行，采集后续真实帧。
- 只保存已开始扰动后、实际发生于有效阶段且尚未落地/终止的帧。候选保留 FIFO、事件、计数等完整上下文；来源包含 trajectory_id、真实步数、动作前缀与有效扰动。
- 物理去重仍采用旧 0.10 m / 0.10 m/s / 0.5 deg / 2 deg/s 等分辨率。**5 cm 采样不会偷偷把统计网格也改细。** 高度等其他物理维度也不改动。

## 速度和成本

设备端路径将策略推理、环境步进、有限值检查及首次落地判断置于一个有界 JAX 循环内，减少逐步 CPU/GPU 往返；候选身份及快照检查仍在入口执行。批量已结束通道保留其首次终点状态，但其可能继续执行的仿真计算按 `batch_size × 实际循环步数` 保守计费，`inactive_lane_interactions` 与有效接续步数分别报告。

串行路径仍保留显存维护；设备端路径依靠最多 8 个并行候选及有界新进程限制显存，实际 Warp/JAX 批处理兼容性由生产小批比较检验。没有预先承诺加速倍数；benchmark 的 `elapsed_seconds` 是本次冷启动/缓存条件下的标签器耗时，`worker_wall_seconds` 包含工作进程内预检/加载等，不能伪装成纯稳态吞吐率。

每 30 秒输出当前任务和日志位置；标签器输出候选进度，设备端还输出逐批耗时、执行 slots 和有效步数。为后续进程设置独立 JAX compilation cache 目录；是否命中以实际运行结果为准。

无失败重试时最大预算为 **878,400 次环境步进槽位**：小批执行比较最多 51,200，采集保守上限 8,000，最多 512 个候选 × 4 evaluator × 400 horizon = 819,200。默认预算 2,000,000 包含失败/未知尝试按完整上限计费的余量。该账本覆盖本次运行，不声称包含全部历史 bootstrap/PPO 成本。

## 输出文件

`figures/` 输出 PNG（300 dpi）、PDF、SVG 与原始 CSV：

- `all_policy_projections`、各 `pi_N_projections`、`union_projections`：新候选上的共同面板接续支持。
- `cross_section_occupancy`、`cross_section_spans`：5 cm x 截面的观测覆盖与跨度。
- `old_new_support`：旧成功支持灰色、相对旧 root 网格的新单元绿色，统一坐标。
- `old_new_coverage.csv/json`：逐策略和并集的旧覆盖、新面板覆盖、真正新增单元、两批累计并集。
- `policy_metrics.csv`、`candidate_outcomes.csv` 等保留原数值；`backend_decision.json` 给出每个策略选择串行或设备端的原因与实测速率。

这里的旧覆盖只指上次共同面板 TRAIN 的成功支持，不代表已整合全部历史 Tube 版本。更密采样产生新增见证不等同于策略能力提升，也不证明连续区域内部可行。

## 验证边界

CPU 回归包括真实采集控制流的假环境多帧测试、真实标签器的串行/设备端行级一致性测试、终止与非有限值处理、批量无效通道成本、续跑/失败回退和图表集合运算。生产 checkpoint/Warp GPU 和实际加速倍率仍等待服务器运行。见 [CPU 记录](verification/dense_tube_cpu_20260907.json)。
