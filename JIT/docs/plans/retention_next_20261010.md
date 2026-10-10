# Retention-next implementation and execution plan

The user approved this finite V/D1 sequence on2026-10-10; the attached findings file is guidance, not evidence of execution. Base b83b586. Inline execution uses existing isolated experiment/jit-retention-first-20261010 worktree; no subagents, no R73 continuation, no full D0 rerun. Unchanged reward, physics, network and success contract.

- [x] Trace existing collect/evaluate, source identities, D0 and numerical-repeat limitations.
- [x] `retention_next.py`: prepare source locks and CPU host-forward initial legality; audit actual checkpoints; encode actual physical layouts and independent role namespaces.
- [x] `retention_next_runtime.py`: one-stage execution, watcher, conservative failure reservations, frozen collection and source/two teacher comparisons, no optimizer functions.
- [x] `retention_next_report.py`: paired per-cell gains/losses and clustered onset intervals; same-root teacher/source/repeat manifest, no student claims.
- [x] Regression tests for charged full-batch repeats, mixed baseline/winner exclusion, policy-specific neighborhood flags, balanced teacher-independent selection.37 related CPU tests passed.
- [ ] V: pi0/R5, A32+B256+C256+D256 perpolicy,250-world physical capacity,400horizon;8batches=800000,1M/4h hard cap; DEV_CONFIRM stays isolated.
- [ ] D1: fixed pi0 source/tail, fixed E73/G73EMA; TRAIN128 with optional128 only if stable selected roots insufficient; solver-dev32 independent episodes. Immediate postpulse snapshot and all legal pi0 suffixes; select target16 roots by pi03-label criterion, <=32qualification candidates. Read-only R5 suffix. Eachmethod16+closed-loop source, H16+pi0tail; one success-first/actiondelta winner,2fixed original17-world repeats, no reselection. Budget worstcase1800384 (32TRAIN+8solver roots),2M/6h; all updates0.
- [ ] Independent result review: R5 remains best_dev_candidate, published_policy null. At least8 distinct TRAIN G lessons over>=2onsets required for D2 preparation. D1 command never crosses stage.
- [ ] Conditional D2: actual lesson receipt, new successful TRAIN anchors, inherited LR and quarter-PPO LR, full-stage cumulative clock, actual same-update weighted gradients/Adamdelta, tiny GPU validation and TensorBoard; no long training. If gate fails report that condition and leave students unstarted.

Canonical restoration numerically reconstructs physics via mj_forward; no claim of simulator internal bitwise restoration. New D1 opt-in preserves snapshot administrative/event/FIFO/return context and restores recorded physical time, while original fresh-continuation semantics remain unchanged. No state/history/counter reset at H16handover. Complete TRAIN collection ledger retains pre/duringpulse failure and no-snapshot denominator. E history retains evaluated Actor identity; source pi0 current flags derive only from matching pi0 labels, others UNKNOWN.

2026-10-10 completed V/D1 boundary: V800000 physics, R5 A32/32 B231/256 C245/256 D210/256 versus pi0 27/32 189/256 187/256 167/256. R5 remains frozen best_dev_candidate. D1 charged180909 including failed6800 reservation; TRAIN3G_only/5Noise_only/5both/2neither/1ambiguous, solver-dev1/1/2/3/0. Eight qualified G lessons span all4onsets. Original D1_002 OOM is preserved; explicit D1_003 uses bounded per-root GPU workers and original failed-root proposals.

D2 preparation is conditional and separate. Actual layout1172900/2000000 includes A/C256000, BC2000updates, new TRAIN keep12800, realGPU micro100, and declared DEV/combinations. Formal arms are not authorized by the preparation/micro command. Source PPO LR3e-5 becomes whole-learner7.5e-6; historical BC LR1e-5/demo1/keep1 is retained. Full BC checkpoints opt-in save Adam/RNG/absolute supervised clock at100/500/1000/2000. Same-stage PPO resume explicitly preserves completed transitions. Actual telemetry uses the same RNG/minibatch/demo/keep samples as the executed loss, weighted gradients and actual Adam parameter deltas; fixed-probe telemetry remains separately named. All normalizer statistics are frozen. Micro reset is nominal complete engineering TRAIN only and cannot validate the formal four-pool mixture. Formal launch gates remain explicit until successful keep pools, reset adapter, new DEV panel and BC-to-PPO handoff are verified.

Final preparation refinement: supplement `D2_002/report_004/formal_preparation.json` locks actual26positive TRAIN snapshots reconstructed from saved nonterminal tick3 context,8teacher-recoverable roots, complete nominal/newrandom pools, new64DEV cells and BC selection score. Added all4BC solver-dev selection nodes, conditional warning repeat-confirm nodes and200transition reset GPU check: refined worst-case1846700/2M physics; these stages remain unexecuted. Actual engineering12900physics,100training transitions,2updates.100step realGPU passed entire-normalizer/fulllearner/RNG/inference/clock checks; current actual reward scalar loaded on TensorBoard6025.57CPUtests passed. R5 candidate only, formal A/B/C absorption/old-loss/independent success NOT_RUN. Full evidence `JIT/docs/evidence/retention_next_20261010/README.md`. Original D1_002 and R74 errors retained; all immutable locks reverified against original committed collection code,101teacher trace/proposal files checked.

## 12. 用户补充：单因素边界与条件性下一试验（2026-10-10）

本补充只进入任务书与设计；不授权新的训练。已启动的 B_keep_coverage_003 保持全部冻结配置完成，不中途改权重、网络、数据来源，不因负结果自动重复增加同类 keep。它只检验旧技能 TRAIN 数据覆盖，不是完整遗忘方案，也不预设容量足够或不足。

当前 demo+keep 已是双来源监督：demo 学习经过验证的恢复教案，keep 学习冻结旧策略在旧技能数据上的动作。不重复添加功能相同的旧模型蒸馏项。报告分别给出旧能力损失、新教案吸收、闭环失败和数值重复变化；MSE 下降不能宣布解决。

根据本次完整结果，只提出一个最有信息量的下一试验，附选择依据、独立预算及可区分的预期结果；A/B/C 不自动启动，也不要求全部执行：

- A 权衡：固定数据和每组 batch 数，比较 keep/demo=1/1 与1.5/0.5，其余保持一致。若更保旧但少吸收则支持局部权衡；不能仅由不改善推断容量不足。
- B 闭环偏移：只在 TRAIN 学生自身访问状态查询新恢复指导，物理验证后才能成为标签，不用 DEV 失败回流。若教师能救且自访问示范改善闭环，则支持分布偏移解释；查询无解/指导不能吸收需分别报告。
- C 容量：固定同一学习通路、数据和预算，比较256×3与512×3学生。仅宽学生更好属于该条件下容量证据，不能单次排除优化、迁移或数值因素。

容量对照接口可以提前设计，但本轮不改执行代码：Actor 结构随 checkpoint 声明，旧声明默认保持256×3兼容；宽学生和小旧老师分别构造，或使用身份/行为验证后的旧老师动作标签。扩宽迁移需保持原策略行为，先做零更新动作与同布局闭环检查。E/G、观测、动作权限和物理不变；不同形状网络不得直接恢复小网络 Adam 状态，不从头训练新大底层。本轮仅记录接口设计，未声称已实现或验证扩宽。

长期能力账本区分原始旧技能、历史学生曾学会的新技能、当前恢复教案；遗忘后重学旧任务不能统一计为新增。不得对所有困难状态简单平均旧老师与恢复老师动作，应记录状态来源、各老师是否有合格解及实际采用标签。新训练均需独立冻结配置与明确预算，不据此自动恢复PPO、E/G外循环或200轮。

当前运行源代码仍锁定0ebfd3c，HEAD cb1e139；该设计补充只改Markdown，执行锁与已启动配置不变。下一试验的实际选择与预算在本次完成后填写，不能预先以容量或权衡取代结果。

容量接口设计细化（仅设计，当前运行不使用）：未来checkpoint增加版本化Actor声明，绑定输入76维、层宽、激活、分布及4维动作映射；未带新字段的历史checkpoint保留既有256×3构造并核对真实参数形状，声明/形状冲突拒绝。critic继续独立256×3声明，宽Actor不隐式扩critic/E/G。保留老师使用其自身Actor+normalizer构造；student有独立网络对象，keep标签不能通过宽学生网络应用小老师参数。

行为保持迁移可采用经验证的隐藏单元复制及出边分摊方案；只在实际网络层/激活/分布满足条件后实现。必须核对所有分布输出以及确定性动作，不能只检查前两维或MSE平均值。固定旧技能/教案观测上的零更新最大动作差、normalizer/critic身份与同布局完整闭环重复分别报告，允许物理数值波动单列。宽学生新建Adam状态，迁移权重与精确resume分开。该接口、迁移和GPU闭环检查目前全部未实现/未执行，不据设计宣称容量对照已经可用。

本次B_keep_coverage_003完成：选BC1000、learner_last2000；稳定吸收仍4/8且历史已吸收root30丢失，DEV迁移3/7；π0旧损失首次7→5、重复3→4。单因素覆盖未建立可靠改善，停止阶段、保留π0/R5。下一试验仅选择B的TRAIN自身访问状态诊断，拟544000 charged/3h/0更新；8根×j[4,8,12]，严格剩余H16-j、固定π0尾部、G16+source和2次固定重放。提案尚未实现输入/代码/时间锁，需独立授权，不自动执行A/C或新BC。梯度冲突支持关注A但不证明其因果；容量C仅接口设计，不能从本次效果推断充分或不足。完整证据见../evidence/retention_next_20261010/keep_coverage/INDEX.md。
