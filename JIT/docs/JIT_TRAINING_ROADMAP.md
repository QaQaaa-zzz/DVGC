# JIT 后续实现与训练路线

Current next run (2026-09-08): JIT/cli/refine_jump_boundary.py --gpu 0 runs nine pi_1 positive-knee trajectories (offsets 0.15/0.175/0.20; window endpoints x=2.85/2.90/2.95 m), with all four frozen continuation evaluators. Source TRAIN frontier evidence is hash-locked. This is local gap refinement, not a global action restriction, independent-seed study, or PPO. Default budget 2,000,000 interactions including failures/retries; first-attempt ceiling 1,847,700. Return compact results_to_send.zip. See JIT/docs/JIT_BOUNDARY_REFINEMENT_20260908.md from the repository root. Older next-run notes below are historical.


Latest production result (2026-09-08): landing_frontier_v1 completed 48 trajectories, 1,141 arrivals, 1,129 bank witnesses, +1,107 root cells (cumulative 3,463), with 92,058 interactions and no PPO. Forty trajectories landed; eight terminated before landing; none hit sampling limits. Twelve no-witness states come from one pi_1 positive-knee 0.2 trajectory, not twelve independent failures. Next: targeted TRAIN boundary refinement and a locked complementary training recipe, not another complete broad scan or automatic PPO. Review bundles are now compact by default; use JIT/cli/package_results.py --output-dir <existing-run> to repackage without simulation. See JIT/docs/JIT_FRONTIER_RESULT_AND_PACKAGING_20260908.md (relative to repository root). Earlier next-run notes below are historical.


Current action after four-proposer production results (2026-09-08): 1,116 arrivals, 1,115 bank witnesses and +904 root cells; cumulative baseline 2,356. All formal labels used serial execution. Run [landing/frontier exploration](JIT_LANDING_FRONTIER_20260908.md) via JIT/cli/explore_frontier.py --gpu 0. Use a locked TRAIN-informed 48-trajectory allocation, stronger legal action offsets, and an explicitly versioned sampling guard to x=8 m while retaining the original pi_0 reference. Report landing/failure/truncation and separate old-corridor novelty from newly observed extended-domain support. Skip repeated device benchmarks; preserve serial identity checks and cost accounting. No PPO, no extra snapshot replay checks, no automatic training admission. Older “next” actions below remain historical.


Latest action (2026-09-08): the 5 cm pi_0 pilot completed with 242 arrivals, 240 bank witnesses and +227 root cells (cumulative 1,452 versus the prior TRAIN panel). All four Warp device benchmarks failed at contact__dim; serial fallback completed labels. The user authorized continuing with the single-world device-map fix and four-proposer discovery comparison. Follow [the current run guide](JIT_MULTI_PROPOSER_DISCOVERY_20260908.md) and run JIT/cli/compare_discovery.py --gpu 0. Preserve prior labels, 5 cm real-frame sampling, the old physical grid and accepted replay limits. New discovery counts use the old shared TRAIN union plus the completed dense pilot. No PPO, no extra snapshot replay investigation; one exploration seed remains development evidence. Earlier “next run” instructions below are historical.


Latest action (2026-09-07): the four-policy comparison completed in production (48 shards, 197,604 new interactions). The user authorized 5 cm real-frame multi-state acquisition and measured label acceleration, with all four legal action channels allowed. Run [the bounded dense Tube pilot](JIT_DENSE_TUBE_PILOT_20260907.md) next; it automatically compares serial/device execution and falls back to serial if needed. Preserve the original centerline and physical-grid resolution, all old evidence, and the accepted replay limitation. No new PPO or additional snapshot replay investigation. This is a 16-trajectory TRAIN pilot, not a completed matched-budget multi-proposer experiment. Earlier run instructions below are historical.


2026-09-07 用户决定：接受约 3.1 cm 初始轮胎间隙和本批数值轨迹差异，不再追加重放验证。六状态 GPU 运行已完成，落地结果与步数一致，serial/shard 一致；逐状态重放未全部通过，保留原记录。当前直接运行[四策略包线比较](JIT_POLICY_ENVELOPE_COMPARISON_20260907.md)，补标签、分角色画图与统计独有物理贡献，再进入现有策略同预算探索对照。下文早期重放验收要求由此决定替代。

依据：[论文大纲](JIT_PAPER_OUTLINE.md)、[代码审查](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md)。原审查基线为 `bfc22f2`；代码后续实现见[实施说明](JIT_PROBE_BANK_IMPLEMENTATION_20260905.md)。目标是经验包线发现，不再要求一个新 Actor 接管整个 Tube。

## 阶段 0：修复证据链，再依赖批量结果

**必须处理：**审查 F01/F02/F03/F04/F07/F08 中与待运行路径相关的问题。先修身份、缓存、逐行终点与合并发布，再依赖分片结果；新 warm-start 训练前必须修公共入口。预测器分数/AP 修复只阻断预测器审计，不阻断不使用预测器的探索。

验收：

- 混用终点的历史报告不能变成新的有效资格记录。
- 换 catalog、Actor、payload、seed、horizon、endpoint 或分片编号后，旧缓存必须拒绝复用。
- 每行标签身份和终点一致；错误合并不留下 completed 输出。
- 已在训练 Tube 内但缺少到达证据的状态可以补证据；同物理状态的不同控制上下文不被错误合并。
- 相同分数的 AP 与行顺序无关；修改分数但未更新锁定记录时审计拒绝。

当前进展：身份、终点、缓存、合并、预测器和公共 warm-start 修复及 CPU 回归已完成；新 bank 采集模式已拆分训练支持去重与到达证据。六状态生产 GPU 运行已完成，落地结果及 serial/shard 标签一致；精确轨迹存在差异。用户已接受该限制，不再追加重放验收，当前进入阶段 1。

历史诊断入口及范围保留在[运行与回传说明](JIT_GPU_EVIDENCE_VALIDATION_20260906.md)。结果已经回传并记录，不再要求运行 `validate_jump_evidence.py`。这次用户决定允许继续补标和比较，不表示精确重放通过，也不自动启动训练。

## 阶段 1：闭合旧实验，保留其真实含义

沿用旧 expanded predictor-audit 目录的 pi_0 proposer 和 pi_0/pi_1/pi_2 evaluator，保持候选、seed、horizon、终点和角色不变。

1. 在生产环境核对已完成 evaluator 缓存的完整身份。
2. 保留已完成的小批串行/分片结果；后续每次缓存复用和合并仍逐行检查全局索引、状态、seed、终点及结果身份。
3. 分进程补齐缺失标签，保留 OOM 尝试，不把工程失败计为物理失败。
4. 严格合并 family 和逻辑角色，复核隔离。
5. 修复锁分审计后评估旧预测器；此项作为旁路诊断。

CALIBRATION/ACCEPTANCE 已记录 pi_0/pi_1 完成，校验后复用；TRAIN 的 pi_0 是未完成尝试。比较入口默认每个独立进程最多 200 个候选，串行补齐并保留失败尝试。

退出产物：闭合标签、隔离报告、完整尝试/重试成本、可选预测器审计。不要修改旧 run 名称来冒充新多探针实验。

## 阶段 2：恢复真实 bootstrap 与现有探针链

物化并核对 up/down、初始价值模型和 Tube0、统一策略开发、Round1 pi_0、pi_1/pi_2/pi_3 的准确工件。使用 manifest 的实际身份，不根据名称或相同步数猜 checkpoint。

特别记录：

- Tube0=222 行，其中 42 行历史标签为负；权重有非零下限。
- 2026-08-28 统一训练完成不等于当前 Round1 pi_0 的成功证据。
- 固定起点的完整状态、centerline 真实帧和成功轨迹需要可读取/可重放。
- 旧 bootstrap `test` 角色与新最终评价分布分别管理。

退出产物：可移植工件索引（位置、版本、已有内容身份、大小、角色），附已完成的小批 checkpoint/动作推理及重放诊断记录和用户接受的限制。大 checkpoint 不必进入 Git，但必须可获得；不要修改旧绝对路径，而应新增路径解析/物化映射。

## 阶段 3：先用已有策略验证新研究方向

使用已实现的 `probe_bank.py` 版本化集合、分进程 suffix 评价和观察索引作为起点，再补完整累计物理注册流程；保留已有的单 proposer 采集原语，为每个 proposer 生成独立 catalog，再以全局来源 ID 汇总。先都从固定起点重放；一般祖先分叉可稍后独立验证。

pi_3 不因旧 core 实现下降而自动排除。先通过身份/上下文兼容检查，再在新协议下测量它的真实互补贡献；旧混终点 gate 不能代替这个测量。

以同预算进行小规模探索对照：

| 组别 | 用途 |
| --- | --- |
| A：冻结 pi_0，固定均匀扫描 | 单策略基础线 |
| B：固定现有策略集合，均匀分配探索 | 测多探针增益 |
| C：同样现有策略集合，预声明的候选分配规则 | 隔离探索分配效果；规则未锁定前只执行 A/B |

不要把改变扫描范围、策略数量和训练预算的效果混成一个增益。记录新增到达、旧到达的新续接、独有见证、并集物理覆盖和成本。累计并集不能作为一个“超级 Actor”的成功率。

退出产物：可重放见证注册表、逐探针贡献表、物理分辨率分析、初步覆盖—成本曲线。此阶段不需要先训练 π4。

## 阶段 4：锁定互补探针的训练配方

这是论文方法 4.5 的缺口。进入大规模训练前必须形成一份可执行配置，明确：

| 必填项 | 要解决的问题 |
| --- | --- |
| 训练支持 | 从哪个 TRAIN 视图选状态；如何兼顾成功稀少、几何边缘与已见证支持 |
| 初始化 | 哪个冻结 Actor/normalizer，或 fresh；critic/optimizer 怎样处理 |
| 起点混合 | x=2.5 地面起点与 RSI 的比例和实际 reset 实现，不能照搬 natural10 名称 |
| 奖励 | 是否沿用历史奖励；若改变，应单独声明并做可比消融 |
| 互补机制 | 除换随机种子外，具体如何改变经验访问或目标；不要把设想当已验证方法 |
| 预算 | PPO、训练外采集、评价、重试的上限 |
| 资格与调度 | 技术资格、见证有效性、边际贡献、活跃执行集合分别判定 |
| 停止规则 | 固定预算或事前声明的边际收益规则，不以全 Tube 掌握率为门槛 |
| 统计与数据 | 独立训练/探索重复、祖先分组、固定最终评价分布 |

可先检验一个简单候选机制：从已见证 TRAIN 支持中按物理单元和阶段分配采样，控制重复密集区域占比，并与均匀/原权重的训练对照。该机制只是待预声明的实验建议，尚未被结果验证；不在本次文档中指定未经讨论的比例或阈值。

退出产物：训练配置、固定比较矩阵、预算账本、small-run 参数恢复/重置/奖励/计数 smoke。已授权范围内的常规修复和只读检查继续执行，不额外制造逐项确认流程。

## 阶段 5：正式验证论文贡献

1. bootstrap 对照：直接完整任务学习 vs up/down 加权支持引导，计入获得专家/种子的成本。
2. 发现效率：pi_0、固定集合、迭代增长集合，在相同物理指标和总成本下比较。
3. 互补性：逐策略独有见证、重复开销、加入顺序敏感性，保留单 Actor 实现作为辅助。
4. 有效性：披露已有重放诊断及时间语义限制，报告分辨率敏感性与未知区域；不追加用户已明确停止的重放验证。
5. 至少三组独立 pilot 重复；根据变异性和预算规划主实验，报告按祖先/轨迹组的区间。不同 checkpoint 不算独立种子。

方法和停止条件冻结后才开启保留的最终评价。若无实机，本论文只支持声明仿真模型/工况的经验结果；实机能力需要另行证据。

## 当前最小可执行顺序

已完成代码守卫及小批诊断（数值差异已接受）→ 闭合旧扫描并画四策略共同候选包线 → 现有策略新协议 A/B pilot → 锁定互补训练规则 → 新探针训练与同预算对照。

不要直接沿旧 selected-policy DAG 启动 π4，也不要因为暂时没有一个 Actor 覆盖全部 Tube 而停止保留有效新见证。
