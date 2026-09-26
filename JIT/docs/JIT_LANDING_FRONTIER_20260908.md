# 落地段与经验边界探索：运行说明

## 已完成的阶段

四 proposer 生产实验完整完成：64 条轨迹、1,116 个候选、1,115 个集合成功见证；累计 root 单元从 1,452 增至 2,356，新增 904。108 个子进程正常退出，账本 111,232 次交互。设备路径能运行，但存在比较差异且一致部分更慢，正式标签全部使用串行。见 [生产摘要](verification/four_proposer_production_20260908.json)。

本轮不训练 PPO，不追加快照重放研究。目标是记录更完整的下降段，并在更强但合法的动作扰动下寻找尚无接续见证的状态。不同强度/次数是根据上一轮 TRAIN 信息锁定的开发分配，不是严格同条件策略排名实验。

## 服务器执行

~~~bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/explore_frontier.py --gpu 0
~~~

新结果默认写入 JIT/runs/discovery/landing_frontier_v1。保留原始 checkpoint、快照、centerline 和三轮结果：

- JIT/runs/policy_comparison/expanded_pi0_pi3_20260907
- JIT/runs/dense_tube/pi0_5cm_pilot_v1
- JIT/runs/discovery/four_proposers_5cm_v1

对应路径选项为 --baseline、--previous-pilot、--previous-discovery。旧 JSON 内部引用的原始绝对路径仍需可访问，选项不改写历史工件。

最后回传顶层打印的 results_to_send.zip；失败也有包。原样再次运行可续跑，不重复完整任务。运行期间不要改源码；变更源码、预算或输入需用新的 --output-dir。每 30 秒仍打印当前阶段和日志位置。

## 预先锁定的探索规则

按照上一轮 TRAIN 的“新增 root 单元 / 账本交互”排序，前两名获得两档强度，其余保留一档。以当前结果，分配为：

| proposer | 归一化动作偏置强度 | 尝试数 |
| --- | --- | ---: |
| pi_0 | 0.15 | 8 |
| pi_1 | 0.10、0.20 | 16 |
| pi_2 | 0.10、0.20 | 16 |
| pi_3 | 0.15 | 8 |

每档强度均覆盖 steer、rear wheel drive、hip、knee 的正负单轴偏置。所有 proposer 的窗口终点均为 x=2.9 m，提前 0.15 m 激活；通过窗口后停止额外偏置，继续同一冻结策略。动作仍裁剪至 [-1,1]。合计 48 条尝试，不依赖本轮结果临时增减尝试，不调整物理状态、不施加额外外力。

采集种子改为 9842101，接续标签种子为 9842201。旧结果不改种子、不重解释。TRAIN 可指导开发，最终 TEST 不使用，候选不自动入训练支持。

## 扩展范围与停止条件

保留原 pi_0 centerline 的真实帧和哈希，将它继续作为参考，不外推参考轨迹。新的采集协议单独声明最大 x=8 m 的安全护栏；状态必须由 env.step 实际生成。

沿轨迹保持 5 cm 真实帧取样，每阶段每 x 单元首次观察，跨过的单元不填充。遇到以下任一条件结束：

- 首次有效落地；
- 任务在落地前终止（保留 physical_failure / timeout 标记）；
- x 超过 8 m；
- 已保存 128 个候选；
- 本次前向尝试达到 400 步。

后三项明确标为截断，不冒充已经画全或落地失败。每条尝试都有成本和终止原因，包括没有保存任何候选的尝试。非有限值属于工程错误，不写成物理失败标签。

原参考范围仍单独统计；新增覆盖分为“旧范围内新增单元”和“只在扩展范围出现的新增单元”。扩大可见范围本身不能写成原有范围的 Tube 变宽。物理去重保持原 0.1 m、0.1 m/s、0.5 deg、2 deg/s 等网格。

## 执行成本

本轮固定用已有生产结果支持的串行路径，不重做设备 benchmark；保留分片进程、显存维护、缓存身份检查和重试计费。仍对每个候选完成四 evaluator 的接续标签，确保“全部失败”是完整评价结论；尚未实现按需少跑 evaluator 的调度。

48 条轨迹少于上一轮 64 条，但轨迹更长、每条可保存更多状态，不能保证总时间短于上一轮。最多 6,144 个候选，首次无失败重试的交互槽位上限为 9,854,400；实际通常因终点提前结束而明显较少，但未预先承诺。

默认每 proposer 预算 4,000,000（四者合计上限 16,000,000，含重试余量）。最高单 proposer 首次上限 3,284,800；更低的 --budget-per-proposer 在启动 GPU 前拒绝。所有失败/未知尝试按预留上限计费。

## 看哪些结果

顶层 figures 目录：

- own_policy_tubes、bank_continuation_tubes：统一坐标的各策略及组合支持；PNG/PDF/SVG。
- boundary_states：灰色为四策略都成功、橙色为部分成功、红色为四策略都没有成功见证。红点不是已证明不可达或物理不可行。
- boundary_candidates.csv：逐状态的接续状态分类、扰动方向/强度、阶段、轨迹和原快照路径。
- boundary_groups.csv：按 proposer、动作方向、强度、阶段的结果计数；同轨迹多个帧相关，不能作为独立样本置信区间。
- trajectory_endings.csv、frontier_summary.json：真实落地、落地前终止、距离/数量/时长截断等记录。
- proposer_metrics.csv、coverage_cost、descriptive_budget.csv：本轮开发分配的收益与成本，不作为严格同条件排名。

累计 root 新增以此前全部已纳入的 2,356 个单元为基线。各子目录沿用旧 dense 图，其 old/new 只与最早共同 TRAIN 面板比较；累计新增使用顶层报告。

收到结果后，先看是否仍有截断，再看强扰动下的无见证状态与策略分歧在哪些方向集中。确认缺口后再锁定互补探针的训练支持、初始化、起点/RSI 混合和 PPO 预算。若仍几乎全成功，说明本轮仍未找到边界，不用人为制造失败或直接宣称边界已确定。

## 验证边界

69 项针对性 CPU 回归通过，包含真实采集控制流的假环境测试、串行跳过 benchmark、预算前置拒绝、新标签种子、域内/域外新增分开计数。没有重跑生产 GPU 或启动 PPO。见 [验证记录](verification/landing_frontier_cpu_20260908.json)。
