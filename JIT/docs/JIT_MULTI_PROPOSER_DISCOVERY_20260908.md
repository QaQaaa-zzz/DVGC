# 四策略独立探索：运行与回传

## 先前结果与当前阶段

2026-09-08 回传的 5 cm pilot 已完成：16 条 pi_0 前向轨迹、242 个真实候选，240 个有至少一个首次有效落地见证。四策略成功数为 235 / 236 / 235 / 223。相对上一轮共同面板 TRAIN 并集新增 227 个 root 单元，累计 1,452；这不是几何宽度增长百分比，也不是训练提升。已知实际交互 19,944 次，四次失败设备测试按上限另计 25,600，账本合计 45,544。子进程耗时合计约 946 秒。

四次设备测试都在 Warp 的 contact__dim 维度处理处失败，正式标签由串行路径完成。此次没有证明批量加速成功。原始结果保留，见 [结果摘要](verification/dense_pilot_production_20260908.json)。

当前阶段是已完成采集—接续—覆盖闭环，下一步比较已有策略各自的前向发现能力。本轮无 PPO、不追加数值快照重放验证、不改变近地初始化、物理模型、原 pi_0 centerline、动作顺序或首次有效落地终点。

## 服务器命令

~~~bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/compare_discovery.py --gpu 0
~~~

保留：
- JIT/runs/policy_comparison/expanded_pi0_pi3_20260907，原始候选快照与冻结 checkpoint；
- JIT/runs/dense_tube/pi0_5cm_pilot_v1，包括其 labels、projected、plan 和结果；
- 原 pi_0 centerline。

可用 --baseline、--previous-pilot 指向搬迁后的两份结果根目录；历史 JSON 内部引用的原始文件仍须可访问，参数不自动重写历史绝对路径。

默认新输出：JIT/runs/discovery/four_proposers_5cm_v1。最后回传顶层打印的 results_to_send.zip，其中包含四个 proposer 的阶段日志、标签、图和汇总。失败也打包；原样重跑可复用完整任务。运行期间不要更新源码。改变实验参数或源码应使用新的 --output-dir；旧结果不覆盖。

## 执行与规模

复用现有 dense supervisor，依次执行 pi_0、pi_1、pi_2、pi_3，每个都是 proposer，四个策略也都是 suffix evaluator。

每个 proposer 使用相同的：

- 固定完整起点，TRAIN，采集种子 9841101；
- x=2.9、3.1 m 两个窗口终点，提前 0.15 m 激活偏置；
- 四个动作通道分别 ±0.075 的归一化偏置，裁剪到 [-1,1]；
- 每条轨迹按 5 cm 保存真实帧，每阶段每 x 单元首帧，不插值；
- 16 条尝试，每条最多 400 步、32 个候选；
- 四策略接续，400 步 horizon，标签种子 9841201，每进程最多 128 候选。

总共最多 64 条前向轨迹、2,048 个候选、8,192 次策略—候选接续评价。没有失败重试时交互槽位上限 3,513,600；默认每 proposer 预算 2,000,000（合计上限 8,000,000，包含未知失败尝试余量），可用 --budget-per-proposer 设置，但不能小于预声明首次尝试上限。

四个 proposer 使用相同尝试次数和预算上限，但真实消耗不一定相同，所以不能直接把各自最终点称为同成本结果。报告额外按固定顺序生成共同预算截断表。

## 设备端执行修复

直接对独立快照堆叠后调用 vmap(env.step) 会错误处理 Warp 的非 vmap 接触/约束缓冲区。新实现改为在有界设备循环内通过 lax.map 按序执行单世界步进，不给 Warp 的 step 施加 vmap 变换。策略推理仍可向量化。

这不是八个世界并行的承诺，主要尝试减少逐步 CPU/GPU 往返。仍用相同 16 个工程候选比较每个 evaluator 的串行和设备端终点、步数及耗时；一致且更快才启用，否则自动串行。设备执行的 schedule 版本写入 execution，拒绝把旧设备缓存当作新版本。实际 Warp GPU 兼容性和速度仍需要本次服务器结果确认。

## 图与表的含义

顶层 figures 目录：

- own_policy_tubes：每个 proposer 真实到达的状态中，该同名 Actor 能接续成功的状态；四策略统一坐标并画并集。前缀含合法动作扰动，后缀使用已声明的 fresh continuation；不声称完整无中断原轨迹精确重放。
- bank_continuation_tubes：各 proposer 到达、至少一个 evaluator 能接续成功的支持，以及跨 proposer 并集。
- proposer_evaluator_matrix.csv：四种到达面板 × 四个接续策略；不同 proposer 行的候选不同，不能当作同状态 Actor 排名。
- proposer_metrics.csv：独立/组合支持、每个 proposer 的独有物理单元与新增单元。
- coverage_cost、coverage_cost.csv、matched_budget.csv：相同 evaluator 集合下，各 proposer 与固定轮转集合的成本—发现比较。
- 所有顶层图输出 PNG、PDF、SVG。每个子目录另保留四 evaluator 共同面板的高度、速度、姿态及 5 cm 截面图。

新增单元的顶层基线为上一轮共同面板 TRAIN 并集，加上已完成 pi_0 5 cm pilot，当前报告应为 1,452 个 root 单元。子目录沿用旧 dense 图，其 old/new 图仅与旧共同面板比较；论文累计新增应使用顶层报告。

物理网格保持 0.10 m / 0.10 m/s / 0.5 deg / 2 deg/s 等旧配置。5 cm 只改变取样和绘图截面。root 包含速度、姿态与角速度，不能把单元数当成二维面积或连续可行体积。

## 成本曲线与推断边界

成本曲线是预声明顺序的离线重放统计：

1. 各 proposer 保持 catalog 原顺序，不按成功/新颖性排序。
2. 单 proposer 的曲线先计入全部采集、工程 benchmark、失败尝试和无效通道开销，再逐候选计入四个 evaluator 的实际接续步数；四标签全部完成后才记覆盖。
3. 组合曲线按候选索引轮流取 pi_0、pi_1、pi_2、pi_3，先支付四者全部前置成本，单元取并集。
4. 在所有单 proposer 末端成本的最小值处做共同预算截断，不插值、不按比例分配一个未完成见证。低预算下组合可能尚无见证，这是成本口径的真实后果。
5. 这是每个 proposer 均使用四个 evaluator 的发现比较；同名 Actor 自接续图是单独诊断，不能用其形状声称采用了更便宜的单 evaluator 调度。

本轮只有一套探索种子，不是正式多次独立重复结论。多个同轨迹状态相关，不能视作独立试验。历史 bootstrap/PPO 成本另列，未计入本轮曲线。TRAIN 可用于下一步选择缺口；本次不自动入训练支持、不触碰最终 TEST。若某 proposer 未保存候选，保留其采集成本并报告零支持，不伪造负标签。

下一步根据图表决定：已有策略集合是否扩展到新方向，哪些新增方向值得继续采集，哪些到达状态缺少接续见证；然后锁定互补策略的训练支持、初始化、起点/RSI 混合和预算，再训练新探针。

## 本次验证

66 项针对性 CPU 测试通过；11 个 Python 文件语法与 57 个本地文档链接检查通过。新增入口缺少旧 pilot 时在启动 GPU 前退出，并生成失败 ZIP。测试不替代真实 Warp GPU 验收或实际加速测量。见 [CPU 验证记录](verification/discovery_cpu_20260908.json)。
