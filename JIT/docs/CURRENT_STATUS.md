# JIT 当前状态与证据

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
