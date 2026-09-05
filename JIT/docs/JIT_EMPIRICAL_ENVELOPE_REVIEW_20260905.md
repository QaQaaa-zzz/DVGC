# JIT 论文大纲对照代码与证据审查

审查日期：2026-09-05。远端目标：`agent/two-phase-soft-tube`。只读 fetch 核实的基线：`bfc22f2e32cb78cb269b0e522c3bdd7c6e7a8d42`。

本次变更重写研究与运行指导文档，**没有修复运行源码、重算历史标签或启动 GPU 训练**。缺陷复现与源代码审查用于明确下一阶段工作，不能把文档中列出的验收条件当作已通过测试。

## 1. 总判断

已有工程基础支持 up/down bootstrap、价值加权训练 Tube、固定地面起点因果采集、固定策略家族落地标签和物理单元分析。它值得保留。

新论文关注“固定模型/工况下，互补探针以有限成本发现经验跳跃包线”。当前源代码仍以固定三策略家族和单后继 Actor 选策为主，尚未完成探针集合、见证注册和互补训练的执行闭环。

同时存在已复现的证据完整性缺陷。应先修这些缺陷，再闭合旧扫描并用现有冻结策略验证多探针发现收益；无需先训练更大的 π 编号。

## 2. 核对范围和验证限制

阅读：root/JIT AGENTS、PROJECT、状态/协议/交接/组织文档、确认后的论文大纲；检查采集、family、snapshot、物理分析、预测器、训练和 workflow/selection 路径；交叉核对已提交 run 摘要与 bootstrap 工件。

本地环境缺少生产 Python `/home/qy/mujoco_playground/.venv/bin/python`、JAX、MuJoCo、Brax 和 pytest。进行了语法、JSON 和源码函数隔离复现；没有冒充生产测试、仿真重放或 GPU 标签验证。大型 checkpoint、catalog、snapshot 和 centerline trace 部分未提交，无法仅凭摘要重建完整实验。

## 3. 发现的问题

### F00 — bootstrap 叙述需要根据实际数据修正（事实核对，已在本次文档修正）

证据：[初始 Tube manifest](../runs/soft_tube/soft_tube_train_v1_20260828/manifest.json)、[diagnostics](../runs/soft_tube/soft_tube_train_v1_20260828/diagnostics.json)、[soft_tube.py](../src/jit_dvgc/soft_tube.py)。

Tube0 实际有 222 行：upstream 117 行中 99 正/18 负，downstream 105 行中 81 正/24 负。`WEIGHT_FLOOR=0.05`，采样权重来自 `floor + scale * value_score`。因此不能写成“筛出全部可靠成功状态作为 Tube0”；应写“按阶段数据与价值评分构建加权初始训练支持”。历史 phase label 不自动等于当前首次落地标签。

[最早统一训练报告](../runs/pi_unified/pi_unified_formal_10009600_seed821101_20260828_retry1/formal_report.json)记录 10,009,600 transitions 完成；但[当前 family 摘要](../runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/frontier_train/labels/summary.json)引用的是 `pi_0_round1_10009600_20260831`。同一步数或同为 unified 不能证明同一个策略。配置与 loader 也保存了早期 pre-jump failure 的开发来源。

论文应补专家/种子开发谱系、阶段访问率和成本对照，不能把“训练完成”直接当成“当前 pi_0 已从固定起点成功”。

### F01 — 混终点旧 gate 仍可生成前瞻资格（高优先级，已复现，未修）

位置：[capability_progression.py](../src/jit_dvgc/analysis/capability_progression.py) `analyze_capability_progression`；[select_iteration_policy.py](../cli/select_iteration_policy.py) `_verify_capability_decision`。

将仓库真实 pi_3 gate 输入分析函数，它的 `core_source` 明确记录 baseline=`stable_recovery`、candidate=`first_valid_landing`，结果仍为：

```text
candidate_policy_authority_eligible = true
formal_prospective_selection_claim = true
```

新 baseline helper 的检查是有效的，但旧报告的分析/选策入口没有统一检查终点和事前协议。哈希匹配无法使不可比的成功定义变得可比。

修复验收：所有新资格入口验证 endpoint/panel/horizon/remaining-time/预声明协议；对旧混终点报告明确拒绝作为有效比较；加入新的机器可读证据资格记录，保留原始结果。新方法不再使用旧 core-retention 门槛，但仍必须拒绝混终点证据。

### F02 — evaluator 缓存和逐行合并身份不完整（高优先级，已复现，未修）

位置：[policy_family_landing.py](../src/jit_dvgc/policy_family_landing.py) 的 `run_policy_family_evaluator_shard`、`_load_completed_evaluator`、`merge_policy_family_evaluator_shards`、`merge_any_policy_landing_labels`；[family CLI](../cli/label_policy_family_first_landing.py)。

复现一：临时缓存只放 `{"status":"completed_shard"}`，本次 catalog/策略路径全部不存在且 shard_index=99、shard_count=1，函数仍返回完成。它在检查请求身份前直接复用缓存。

复现二：同名 evaluator 的第二行换 Actor、payload 和 `stable_recovery` 终点，family OR 仍接受。函数仅从首行建立 family 身份，没有逐行验证所有关键字段。

静态核对：完整 evaluator 缓存未绑定请求 catalog/seed/horizon；family merge 上层只传 evaluator 名称。归档发生在验证前，部分终点/身份检查发生在底层结果写出后。底层 shard merge 已有全局索引和分片间协议检查，问题是上层请求契约和逐行绑定仍不足。

修复验收：完整缓存键、标签文件身份、逐行二元标签与策略/终点验证、请求一致性、完整无重叠索引覆盖；检查全部成功后再原子发布。错误 merge 不留下 canonical completed 输出，已完成合法输出不被当成 incomplete 归档。

### F03 — 分数锁定在审计入口没有闭合（预测器使用前必须修，已复现）

位置：[family_landing_predictor.py](../src/jit_dvgc/family_landing_predictor.py) `lock_forward_predictor_scores` / `evaluate_locked_forward_scores`。

锁分函数保存 `scores_sha256`，但审计函数不调用已有自哈希检查。临时修改分数并保留旧 hash，独立 verifier 拒绝，audit 仍返回 `completed_fresh_forward_audit`。

还缺 target evaluator bank、catalog/protocol 和精确快照上下文的完整绑定。相同 candidate_id/qpos-qvel 哈希下更换评价策略集合，会改变预测目标。报告中的前瞻布尔值本身不证明时间顺序。

修复验收：读新标签前验证分数锁定及完整来源；明确模型目标 bank；保留事前记录；角色/候选/快照与标签精确连接。本次独立核对三份已提交锁分文件的自哈希均匹配（TRAIN 1,038 / CALIBRATION 342 / ACCEPTANCE 333 条）。此缺陷不等于已上传结果被修改；它表示代码不能可靠拒绝变化，也不能仅凭自哈希证明时间顺序。不使用预测器的实验不以该模块效果为前置条件。

### F04 — 训练支持去重阻断能力证据补齐（高优先级，源码确认，未修）

位置：[acquisition/causal_jump.py](../src/jit_dvgc/acquisition/causal_jump.py) `collect_jump_start_connected_candidates`，尤其 `support_hashes`、`existing_control_tube_state` 和 `seen_states`。

采集器将训练 Tube 的物理状态哈希收集后，在保存候选前丢弃命中状态。这适用于旧“只找新训练行”的目的，却不满足新包线定义：训练 Tube 中某状态可能还没有当前固定起点的前向到达证据，应允许补充来源链或互补策略见证。

同时，`state_sha256` 在相关路径是 qpos/qvel 的物理哈希。物理相同不等于 FIFO、事件、last_action、阶段/剩余时间相同，不能同时充当完整续接身份。

修复验收：分别管理训练行去重、完整见证身份、物理覆盖去重；允许已有 S 状态追加证据；保留不同 proposer/context；指标计数不重复膨胀。既有历史去重结果保留原义，不事后伪造被丢弃状态的见证。

### F05 — 固定三成员 family 与多 proposer 新目标不一致（必需功能缺口）

位置：[policy_family_landing.py](../src/jit_dvgc/policy_family_landing.py) 强制名字集合恰为 pi_0/pi_1/pi_2；[causal_frontier_protocol.py](../src/jit_dvgc/causal_frontier_protocol.py) 每 role 解析一个 proposer；[causal_jump.py](../src/jit_dvgc/acquisition/causal_jump.py) 每次接收一个 policy。

采集原语可以接收冻结 policy，故无需重写动力学采集。缺的是版本化集合、多个 proposer 的调度、跨 catalog 的稳定来源 ID、全局角色隔离和覆盖汇总。一般祖先分叉复用也没有实现；当前从固定起点逐次重放仍合法。

修复验收：保留旧协议固定集合兼容性；新版本允许声明成员及 proposer/evaluator 角色；逐探针 catalog 和成果可追溯；增加新成员不改变旧 OR 的身份；不同集合的未成功结果不混用。

### F06 — 旧 workflow 仍把单 Actor 资格当迭代门槛（必需迁移）

位置：[prepare_iterative_envelope_workflow.py](../cli/prepare_iterative_envelope_workflow.py)、[iterative_frontier_protocol.py](../src/jit_dvgc/iterative_frontier_protocol.py)、[capability_progression.py](../src/jit_dvgc/analysis/capability_progression.py)。

DAG 要求 `candidate_policy_authority_eligible=true`，随后写 selected successor。它不会因为文档改变就自动成为新探针集合工作流。

修复验收：技术资格、见证有效性、边际贡献/执行调度、单 Actor 实现分开；用 bank manifest 和 witness registry 驱动新实验，旧 selector 只服务历史兼容。pi_3 core 实现下降不能否决它所有独特成功见证；同样训练完成也不自动证明探针有收益。

### F07 — 公共 warm-start 与训练 reset 语义尚未统一（训练前必须处理）

位置：[unified_formal.py](../src/jit_dvgc/unified_formal.py)、[training/formal.py](../src/jit_dvgc/training/formal.py)、[train_unified_from_pi0.py](../cli/train_unified_from_pi0.py)。

通用 loader 支持 warm start，但 flat `run_unified_formal` 仍调用 fresh loader，trainer 默认 `restore_params=None`。历史 CLI 用模块函数替换与 trainer 包装注入参数；因此历史 pi_3 warm start 可以成立，但公共 API 不能宣称已完整统一。

另一个新目标相关问题：`natural_reset_probability` 和 `_load_reset_mixture` 锁定的是 `existing_phase_u_natural_reset`，不是 x=2.5 固定跳跃 reset。复制旧 natural10 配方不会自动实现新任务起点混合。

修复验收：公共接口明确路由 Actor/normalizer 恢复与 fresh critic/optimizer；对真实调用做参数路由检查与小 GPU smoke；配置明确训练 reset 的实际语义和支持来源。训练可有合法 RSI，但不能以其产生前向见证。

### F08 — 精确恢复与时间语义需要端到端验证（尚未验证，不宣称已有 rollout 错误）

位置：[unified_envelope_snapshot.py](../src/jit_dvgc/unified_envelope_snapshot.py)、[unified_continuation_labels.py](../src/jit_dvgc/unified_continuation_labels.py)、[causal_jump.py](../src/jit_dvgc/acquisition/causal_jump.py) `_normalize_snapshot_context`。

快照确实保留 physics、FIFO、actions 和 up/down events，恢复路径也有兼容检查。但候选会规范化部分 transition 标志，续接路径涉及 administrative counters，最终“同一任务的完整前后缀”需要验证上下文与时间预算没有改变语义。

验收：同一 captured state 原地续跑 vs 保存/恢复续跑的动作、事件、阶段、终点及剩余时间一致（按声明容差）；多 evaluator 的正规化不得遗漏各自必要观测上下文。先完整起点重放，再引入一般祖先复用。不用物理哈希代替这项验证。

### F09 — 包线生命周期与完整成本账本缺失（论文主实验前必需）

位置：[analysis/causal_jump_capability.py](../src/jit_dvgc/analysis/causal_jump_capability.py)、[analysis/capability_tube.py](../src/jit_dvgc/analysis/capability_tube.py)、[policy_family_landing.py](../src/jit_dvgc/policy_family_landing.py)。

已有正例投影、阶段分布和 source comparison 能复用；没有贯穿 growing bank 的完整见证注册、成员版本和边际归因流程。family summary 汇总 completed evaluator interactions，但失败尝试的总成本不能靠它闭合。

验收：新到达与旧到达的新续接分别归因；累计覆盖去重、每个尝试/重试成本可追溯；固定解析规则拒绝跨工况/分辨率/终点的不可比来源。物理覆盖—总成本曲线必须和同预算对照一起报告。

### F10 — AP 同分次序依赖与拟合资格看 ACCEPTANCE（预测器科学性问题）

位置：[family_landing_predictor.py](../src/jit_dvgc/family_landing_predictor.py) `_ranking_metrics`、`assess_phase_support`。

同样 scores=[0.5,0.5]，labels=[1,0] 的 AP=1.0，交换顺序后 AP=0.5。应按分数阈值分组计算，使结果不依赖行顺序。还应明确 AP 与梯形 PR-AUC 的区别。

拟合资格包含 ACCEPTANCE 是否两类，审计也要求两类才能输出。新设计应让 TRAIN/声明的校准条件决定拟合，独立评价报告真实类别与可定义指标；不要通过要求负样本来阻断有效全正例包线，也不要补造负标签。

### F11 — 可移植证据与命名尚需清理（复现/写作缺口）

大量轻量报告引用生产机 `/home/qy/...` 和未提交的 checkpoint/catalog/trace。Git 摘要足以核对记录，不足以重新计算所有结论。增加物化/位置索引，保留工件身份，不改旧绝对路径来冒充原始记录。

`continuation_viability_proven` 等旧字段只是经验结果；`selected`、`acquisition_guidance_authorized` 等历史布尔值也不等于新实验资格。新 schema 使用 witness 术语并配合版本迁移，不修改历史 JSON。

bootstrap continuation 文件已有 `test` 分片结果。新最终 TEST/JCE/JEL 必须与它们区分并隔离，不能笼统称所有 test 从未使用。

## 4. 本次可复核的局部结果

方法：使用 AST 提取源码函数，保留原函数体，以显式 fixture 替代会引入 GPU 依赖的外部模块。预测器排名的其他指标在 fixture 中留空；AP 逻辑来自原函数。所有修改分数/缓存/行的操作仅发生在临时目录，未改真实运行工件。

| 检查 | 观察结果 | 含义 |
| --- | --- | --- |
| 最新提交 47 JSON | 全部可解析 | 文件语法通过，不等于科学有效 |
| 最新提交 8 Python | 源码编译通过 | 不验证导入依赖或 GPU 执行 |
| 三份已提交 forward-score 自哈希 | 全部匹配 | 当前保存内容一致；不替代审计入口校验或时间证据 |
| 新 endpoint helper，缺 core 身份 | 拒绝 | 正确 |
| 新 endpoint helper，core/boundary 不同 | 拒绝 | 正确 |
| 新 endpoint helper，均 first_valid_landing | 接受 | 正确 |
| 真实旧混终点 pi_3 gate | authority=true / prospective=true | F01 复现 |
| 锁分后改变分数，旧 hash 不变 | verifier 拒绝，audit 却 completed | F03 复现 |
| completed_shard 缓存＋不存在输入 | 直接返回 completed_shard | F02 复现 |
| family 第二行更换 Actor/payload/终点 | 两行仍被合并 | F02 复现 |
| 相同分数交换标签顺序 | AP 从 1.0 到 0.5 | F10 复现 |

没有重跑此前报告的 51 个 pytest 测试。新代码迁移应把这些反例转为真实入口的回归测试，不能依赖只检查正常输入的测试。

## 5. 已有结果与证据路径

| 证据 | 路径 | 可支持的结论 |
| --- | --- | --- |
| up/down 冻结 | [frozen_experts.json](../runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json) | 专家身份及训练步数记录 |
| Handoff | [manifest.json](../runs/handoff_bank/handoff_bank_9977856_jit8/manifest.json) | 56 captured snapshots，记录 1,876 interactions |
| Tube0 | [manifest](../runs/soft_tube/soft_tube_train_v1_20260828/manifest.json) / [diagnostics](../runs/soft_tube/soft_tube_train_v1_20260828/diagnostics.json) | 加权训练支持已生成，非全正例 |
| 最早 unified | [formal_report.json](../runs/pi_unified/pi_unified_formal_10009600_seed821101_20260828_retry1/formal_report.json) | 训练完成，不等同当前 pi_0 |
| Wide family | [summary.json](../runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/frontier_train/labels/summary.json) | 1,230/1,258 TRAIN family first-landings |
| Strict isolation | [role_isolation_strict.json](../runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/role_isolation_strict.json) | 当前旧轮的隔离报告 PASS |
| 因果物理支持 | [summary.json](../runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/causal_jump_capability/summary.json) | 报告 713 个新增因果 TRAIN root cells |
| pi_3 realization | [increment_summary.json](../runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/pi3_train_realization/increment_summary.json) | 1,130/1,258 source、1,061/1,159 increment；不是互补见证计数 |
| 旧混终点 gate | [summary.json](../runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/pi3_landing_replay_acceptance_gate/summary.json) | 显式终点混用，不能作为公平比较 |
| Expanded audit | [历史索引](JIT_JUMP_START_TRAJECTORY_INDEX_20260904.md) | 三组 arrivals/scores 已记录、family 标签未闭合 |

上表的完成/PASS主要是 committed artifact 的记录；本次未重新执行底层轨迹。旧 family 比 pi_2 单独多 8 个 source TRAIN 成功（1,230 vs 1,222）；其物理价值和边际成本应实测。不能由 pi_3 总成功较少推断独有贡献为零。

## 6. 论文证据还缺什么

- “有限起跳机会使学习困难”的阶段访问率、失败阶段分解和对照证据。
- up/down 加权支持相对于直接完整任务学习的净收益，包含专家/seed 成本。
- 新策略集合的版本化执行、互补探针生成规则和预声明停止规则。
- 同预算 pi_0、固定集合、迭代集合的覆盖—成本曲线。
- 逐策略边际贡献、新 arrival 与新 suffix 的归因。
- 前后缀重放、分辨率敏感性、独立重复和祖先分组统计。
- 最终未用分布及完整工件索引；若主张实机能力，需要另行实机证据。

## 7. 本次文档变更的作用

root/JIT AGENTS、PROJECT、CURRENT_STATUS、协议、训练路线、交接和组织说明均改为经验包线目标。仓库加入确认后的论文大纲，并注明 Tube0 与 pi_0 身份的事实修正。旧科学审查响应和实验索引只加历史状态提示，原始实验 JSON 不变。

**文档已对齐；运行源码缺陷尚未修复；新方法的实验收益尚未证明。**按 [训练路线](JIT_TRAINING_ROADMAP.md) 分阶段推进，无需重做全部 bootstrap，也不应直接沿旧 DAG 训练 π4。
