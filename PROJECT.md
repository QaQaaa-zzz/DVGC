# DVGC / JIT 项目与当前工作

更新：2026-09-12。范围：`/home/qy/DVGC/JIT`；代码分支 `agent/two-phase-soft-tube`。本页替代此前按时间追加、含过时“待实现/等待训练”的状态描述。历史原始工件与 Git 历史保留。

## 我们要解决的问题

在固定模型、载荷、起点和落地语义下，以有限总交互成本发现经验跳跃包线。种子策略展示可行跳跃；残差探索策略使冻结跳跃策略偏离已有轨迹，产生真实到达的新候选。当前策略库没有成功续接的候选仍可能被未来策略学会，因此独立保存为 pending，再训练新跳跃策略并延迟重评。只有同一完整状态及上下文的真实前缀与无冲突成功落地续接才能进入已见证包线。

分阶段 bootstrap → 加权 Tube0 训练支持 → 成功统一策略与真实 centerline → 冻结 π + 残差探索 → 到达新颖性/候选池 → 新跳跃 π 的 PPO → 旧候选延迟重评 → 累计成功见证。

这是一条有限预算下交替探索和学习的外循环，不是贯穿多个训练阶段的端到端反向传播，也没有证明收敛到完整物理空间。PPO 是训练工具；研究价值必须由发现效率、可学习候选转化和完整证据支持。

## 当前真实进展

| 阶段 | 核验结果 | 解释范围 |
| --- | --- | --- |
| 历史 all_proposers_v1 | π5/π6 各 128,000；root cells 1,689→4,629→9,296；新增交互 1,275,465 | 历史协议的多策略开发结果；不是残差网络收益 |
| 训练长度 pilot | 32k–256k checkpoint 及小预算探索对照完成 | 暂用 128k–256k 有界预算；没有普适最优步数结论，不再无目的扫描 |
| 接续加速/事件审查 | first-success 独立协议、unknown/冲突隔离、GPU 批量工程验证已实现 | 不重写旧标签；16384 容量不是当前所有训练使用的并行数 |
| 残差探索 | frozen π + Actor 规模残差 PPO，106D 历史输入，4D 有界输出 | 监督拟合旧随机扰动已被替代；当前基础 critic 仅 telemetry |
| 延迟评价两轮 pilot | 两次 128k 已完成；关联实际交互 348,371 | 两臂旧无见证→见证均 0，未证明探索收益或长期可学习性 |
| 论文与维护 | 当前正文、公式、图件和证据索引统一维护 | 详细方法及开发证据草稿；仍需主实验才能支持投稿结论 |

最新完成入口：`JIT/runs/experiments/delayed_completion_start_20260912/summary.json`。其补完任务新增 **128,407** 交互（PPO128,000 + TRAIN panel77 + suffix330），墙钟 **221.031 s（含等待）**。它仅补完第二轮；首轮及两轮采集在 `delayed_exploration_reuse_20260912/queue/stage_0_result/`。首轮learned的19,200步前向原件在`JIT/runs/experiments/delayed_exploration_20260912/queue/stage_0_result/round_000/learned_residual/arrivals/`，reuse保存锁定引用。原任务的 `error` 是历史事实，不能改为 completed。

| 条件探索臂 | 首轮累计已见证 cells | 第二轮累计已见证 cells | 第二轮 pending 候选 | 两轮延迟 0→1 |
| --- | ---: | ---: | ---: | ---: |
| learned_residual | 150 | 289 | 12 | 0 |
| fixed_random | 151 | 308 | 9 | 0 |

两臂每轮前向安排均 19,200 计费步，共用相同 π 基线。只有 learned 臂 pending 参与共享后继 π 的训练，因此这是**条件探索对照**，不是独立全流程等成本对照。已见证候选条数与物理单元数不得混用。单条训练谱系、候选截断和新颖性奖励饱和限制解释。

## 当前实现契约

- 固定起点 x=2.5 m，约 3.1 cm 轮胎初始间隙；固定 XML、2 kg 载荷、仿真 0.005 s / 控制 0.020 s；终点为首次有效落地。
- 归一化动作顺序为 `[steer, rear-wheel drive, hip, knee]`。本轮髋膝使用 `keyframe_centered_absolute`，转向是位置目标，不能混用其他项目接口。
- 基础 π 输入 76D；残差 Actor 与独立值网络输入 106D privileged observation，包含原有 3 帧历史。残差 Actor 为 256×3 Swish、8 个高斯分布参数、4 维 tanh 动作，四通道幅度 0.15；本轮训练过程中基础 Actor/normalizer 冻结。
- 新颖性针对当前 π 的基线和本轮 ledger，不针对全历史策略库。新到达奖励不要求当前 bank 成功；包线准入仍必须有成功续接。
- 新 π：旧 Actor+normalizer warm start，fresh critic/optimizer。20% 完整固定起点、80% 快照；可用 phase 内 pending 占快照抽样质量 25%，其余 witnessed；保留 phase/group 权重。
- 待评价/错误/冲突为 unknown；完整当前 bank 没有成功才是 no-witness。旧有效见证保留，无见证不等于物理不可行。
- 延迟反馈改变候选状态、重置支持和下一轮 source π，不把旧轨迹改奖励后回放为新的 PPO on-policy 数据。
- 冻结基础 critic 的质量评分惩罚、扩散模型、自动按收益分配 proposer、无限外循环/连续域收敛均未实现或未成立。

## 下一步：先解释现有结果，再做有限对照

1. **零新仿真分析**：对已保存状态离线重算多分辨率的新颖性、重复率、候选上限命中率、残差有效幅度/饱和和 pending 相位分布；报告原始与派生口径，定位奖励缺少区分度的原因。
2. **小型机制实验**：预先声明共同 π、动作采样模式、分辨率、训练/评价分离、预算及停止规则，对比当前新颖性与一个有区分度的新颖性方案。先验证奖励能区别有用行动，暂不加 critic 惩罚来掩盖当前问题。
3. **公平外循环**：固定 bank、随机扰动、学习残差等主臂按完整成本匹配；如声称延迟学习有效，必须比较 pending-enabled 与 witnessed-only 支持，并分别训练各自后继策略。
4. **论文闭合**：独立训练/探索种子、相位几何和分辨率敏感性、全部成本、冻结最终评价；bootstrap 若没有匹配消融，仅作为初始化背景。

本次文档任务没有启动任何新训练、采集或续接评价。后续实验必须先写配置、数据角色、预算和停止条件；既有授权范围内无需重复询问。不要把过期队列或旧文档中的命令当作新启动要求。

## 维护与交付

- [当前证据与成本](JIT/docs/CURRENT_STATUS.md)；[接手入口](JIT/docs/CODEX_HANDOFF_20260911.md)；[路线与证据缺口](JIT/docs/JIT_TRAINING_ROADMAP.md)。
- [论文大纲](JIT/docs/JIT_PAPER_OUTLINE.md)、[中文草稿](JIT/docs/paper/JIT_PAPER_DRAFT.md)、[可打印 PDF](JIT/docs/paper/JIT_PAPER_DRAFT.pdf)、[图件/数据/重绘索引](JIT/docs/paper/README.md)。
- 原始轨迹/快照/checkpoint/完整图保存在 `JIT/runs/`，本次派生 CSV/JSON、来源哈希、SVG/PDF/PNG 和绘图脚本在 `JIT/docs/paper/`。
- 2026-09-12 已 fetch 核验报告分支：`6121da9`，最新 compact report 为 `review-bf486f2945`，早于本地两轮 pilot。不能用该远端报告证明最新训练状态。
- 本次更新范围覆盖全部现行项目入口、状态、方法、路线、验证、模块和论文文档。带日期的旧结果报告保留为历史证据；旧根目录论文规格加入历史标识，见 [维护清单](JIT/docs/paper/README.md)。

## Approved quality-aware residual exploration — 2026-09-12

User approved implementing and directly training the agreed novelty/physical-failure comparison; all networks must retain loss, reward composition, hyperparameters and process artifacts. First isolate reward effects from a common frozenpi6 and fixed complete starts: pure full-trajectory novelty, novelty+physical-failure cost+small normalized residual-energy cost, and matched fixed uniform residual control. No current-bank failure is a permanent infeasibility label; no raw critic gate, physics change or whole-Tube Actor claim. Existing-state curriculum follows evidence from this controlled stage; this experiment does not change reset sources.

New opt-in trajectory_quality_v1 pays each new real nonterminal physical cell once perbatch (first occurrence perepisode, shared across duplicate episodes), independent of candidate export cap. Padding earns nothing; physical failure costs once perepisode. Declared quality weights novelty1,physical_failure50,normalized_delta_mean_square0.01 are an initial ablation setting, not tuned optimum. Pure arm uses1/0/0. Actual task reward and all environment reward components remain telemetry, not silently mixed into exploration rewards.

Process artifacts: hyperparameters.json, network_inventory.json, per-update optimizer_updates.jsonl with named actor/critic/entropy/total losses, approximate KL/clip fraction/gradient norm, per-batch training_metrics.json and CSV including reward decomposition, per-transition reward_components NPZ plus full trajectory/task-reward components, checkpoints and source/input hashes. Frozen base Actor/critic are explicitly marked nontrainable; they have no fabricated optimizer losses. Replot PNG/PDF/SVG is generated at completion/error.

Authorized bounded execution:2env×400×(2batches+2diagnostics)=3200-forward engineering check; if it completes, three matched arms8env×400×(32batches+2diagnostics)=108800 each (326400 total), seed9950101,lr3e-5,oneepoch,batch256,delta0.15all4channels. Combined forward ceiling329600; suffixes deferred and no new jumping-policy training in this reward ablation. More residual learning (32batches vs earlier4), one attempt perstage,1800s perchild, external GPU job gate. Each arm retains all evidence and common exact per-pi baseline; training novelty counts and weighted total reward are separate fields.

Validation/launch:142CPU tests passed; real3200-interaction smoke completed and component-array totals/process artifacts verified. Supervisor1149051 is running the matched three-arm32batch experiment. Live entry:`JIT/runs/experiments/quality_exploration_20260912/INDEX.md`. No outcome/superiority claim before the comparison completes. Existing-snapshot curriculum remains the following stage rather than a silent reset-source change in this ablation.
