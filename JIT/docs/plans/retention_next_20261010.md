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

## 2026-10-11 用户改选A：BC keep/demo权重对照

用户“再换一下demo和keep的比例试试呢？”选择此前已声明的A设计：固定同一合并keep、G8教案525帧、每组batch256、相同seed和采样键，从原始π0/fresh Adam/RNG开始，keep/demo由1/1改为1.5/0.5。冻结normalizer/critic、Actor-only Adam1e-5/clip1、全部任务/动作/物理/成功判据不变。以已完成B_keep_coverage_003的1/1结果作为固定对照，不重测基线或采keep，不修改原阶段和原时钟。

独立阶段最多2000BC，0/100/500/1000/2000节点；原7DEV主分数/最早平局/固定B/D警戒、8TRAIN候选2次复核及64题四格全部保持。π0/R5完整物理和同身份0更新15根轨迹复用，记录来源。新增actual-layout上限4×15×17×400+4×64×400+4×2×64×400+2×8×17×400=824000charged；此前累计776098，最坏1600098/2000000。新阶段声明3h独立时钟，旧12h元数据不改，旧阶段已结束。运行前必须锁定该新时钟和本轮明确试验指令，不用旧授权布尔绕过。只有本次单次试点，不自动扫描比例或重复失败节点。

执行顺序（writing-plans/TDD，当前隔离worktree原位实现，无subagents）：
- [x] 写失败测试：权重真正进入total loss和同mini-batch梯度；默认1/1兼容；同batch/RNG；比例/源数据/时钟/预算漂移拒绝。
- [x] warmup_actor显式demo_coefficient/keep_coefficient，默认1/1；原优化器和采样不改；完整状态与实际加权梯度记录系数。
- [x] 现有run_retention_b新增prepare-weight-ratio，独立schema/目录/输入与代码锁/预算/grant/3h时钟；静态audit无GPU；worker实际读取原合并keep，无采集。
- [x] CPU定向回归、实际输入audit/dry-run、零GPU检查，commit执行版本后prepare；登记有限阶段后真实GPU连续≤2000，TB新目录/HTTP标量和watcher同步核验。
- [x] 完整状态审计、同根吸收/全部7DEV/64题保持与固定复核、XY图/梯度与参数增量；比较固定1/1、记录π0旧损失和历史学生新技能损失；选候选后停止、台账与Git交付。

可区分结果：更保旧而少吸收支持此数据通路上的权衡；二者都改善属于有限开发证据；keep误差下降而闭环不改善，不能继续扫权重制造通过；失败不推断容量不足。不新增旧模型蒸馏项，不启用自身状态新标签或宽网络，不开启PPO/E/G。

本次A已完成并停止：B_weight_ratio_001，执行3cc875e，BC100 B格保持性复核确认退化，learner_last100/选BC0。TRAIN首次2/8、未见DEV2/7；首次新22旧损失6，复核新17旧损失7，同更新1/1为3/8、1/7、新21旧8、复核新16旧8。旧损失虽部分下降，随机无扰动B两次少1/16仍按原警戒拒绝，未建立可靠改善。新增101416charged、累计877514/2M、735.7s；0/100fullBC通过真实保存状态核验，未来节点不伪造。保留π0/R5，下一步仍只提案B自身访问诊断544000/3h/0更新，不自动执行。证据见../evidence/retention_next_20261010/weight_ratio/INDEX.md。

## 2026-10-11 本轮授权：TRAIN学生自身访问状态诊断与投稿证据路线

本节记录用户本轮明确授权的设计、代码修复和单次有界诊断启动，覆盖上一段“仍只提案、未授权”的历史状态。本次先登记并执行544000 charged、3小时、0优化更新的诊断；后续阶段须依据结果另行冻结配置和有限预算，不能把日期安排或D2余额当成无界自动训练许可。用户本轮要求直接安排、修改并开始后续工作；此处有限分阶段安排是执行决策，不是声称用户只授权了诊断或必须再次确认日常推进。实际run目录、代码提交、输入锁、时钟及运行状态由本轮执行manifest和共享`PROJECT_STATE.md`登记；本计划不是第二份当前事实台账。冻结全局R5仍仅为best_dev_candidate，published_policy仍为null。

选择依据（已保存开发证据）：增加keep覆盖后，BC1000只有4/8根稳定吸收、3/7未见SOLVER_DEV成功，π0旧损失首次7→5但固定复核3→4，未建立可靠保持改善。新权重在同BC100把TRAIN首次吸收3/8降为2/8，在随机无扰动B格两次少1/16并触发原警戒。两条路线都未解决“验证老师有解、独立学生闭环失败”的瓶颈；负梯度余弦、动作MSE和这些负结果均不证明容量不足。故本轮不继续扫比例、补同类keep或扩宽，先查学生实际访问状态上是否存在可验证指导。

本轮冻结合同：

- 学生为`B_keep_coverage_003/B/update_1000.pkl`，SHA-256 `b981714174b59e5fbea4178582e43318b6d3bae8fb73db18c412c1b518d63f09`；Actor `f1d1d460082fbc20eafc39d0ad4f318a38877b5b1b4122195ed3f12d8a52057e`，normalizer为原π0 `db84aad7750991d6b720e9f065ed7e49f862893c7753f9839c593d7951b6ea86`。这是通过原节点警戒的开发候选，未被部署采用；不使用被拒1.5/0.5 BC100、BC2000或R73学生。
- 全部8个既有TRAIN根，后缀68/37/10/18/26/30/55/63；每根只在原桥接窗口内完成学生j=4/8/12控制转移后查询，共24个预声明槽位。保留完整真实状态、物理时间、Actor观测历史、动作FIFO、事件及累计终止上下文；缺失或先终止不得改点补题。快照恢复的数值重建限制单列，不宣称MuJoCo内部逐位恢复。
- 恢复老师固定G73 EMA及原π0尾部，以已有D1 source binding核验完整身份。G每个查询给16候选，另加1条闭环π0 source，共17世界；只执行剩余H16-j控制步，原总交接时刻H16不后移，之后继续同一π0直至原成功/失败/时限终点。保持观测、动作权限、奖励、物理、控制频率和成功规则，不注入第二次外扰。
- 赢家沿用冻结的success-first/action-delta规则，保存全部候选和无解结果；选中后仅做两次相同17世界完整布局的固定赢家复核，不重采、不重选、不重复直到成功。source与G分开记首次/复核标签。搜索有解但固定重放未通过者为ambiguous，工程错误、缺失、非有限为UNKNOWN，不冒充无解。
- 仅TRAIN查询，无DEV回流、无TEST、无学生/BC/PPO/E/G更新；不从本诊断生成自动训练队列。旧π0技能、历史学生新技能和当前教案三类分别列账，root30不能计入π0旧技能丢失。所有8根均参与，不能只挑18/30/55或只保留成功查询。

独立费用和停止条件：学生自身访问采集8×17×400=54400 charged；24查询×3批（搜索及两次固定复核）×17×400=489600 charged；合计上限544000。此前D2累计877514，完成最坏累计1421514/2000000。400为原完整任务控制转移上限，不因短剩余桥接窗口少计完整物理批次；active、padding、失败预留和复核分别收费。启动时冻结新的独立3小时截止，不改历史D2或权重阶段时钟。预算/截止到达即停止并保留partial，身份/合同失败则停止并保存错误；没有自动重试、替补根、额外查询或后续训练。

本轮预声明选择规则：以每个`root × j`的同状态π0 source和固定G赢家搜索+两次复核标签分类；稳定成功须三个已声明标签均成功。每个查询同时保留学生原完整闭环结果及已保存的根重复结果，报告24槽位的有效/缺失/失败分母，并按8个祖先根汇总，不把24槽位或17世界当独立训练种子。

|观察|可支持的解释|下一阶段选择（本轮不自动执行）|
|---|---|---|
|稳定失败学生根的自身状态被π0 source稳定救回|原保持策略在该状态有可用接续；未证明G必要|优先分析学生偏离旧指导及交接兼容性，不把source解计成G新增|
|稳定失败学生根中，π0 source三次均失败而固定G赢家三次成功|该有限同状态对照存在G指导见证；支持评估自身访问指导分布|仅此时优先设计单因素自身访问示范学习；先冻结合格标签、原keep、完整对照与独立有限训练预算|
|G仅在原本稳定成功学生根找到解|还不能解释当前稳定失败根的转化瓶颈|保留这些结果，不以它们代替失败根的改善证据|
|π0和G均未找到稳定成功或标签发生翻转|该固定oracle/剩余窗口未提供可靠指导，或重复性不足|报告无见证/ambiguous；不推断物理无解、容量不足，也不自动增加搜索量|
|输入/恢复时间/历史或固定赢家复核不一致|诊断有效性尚未成立|先修合同及证据，不做学生能力归因|

若至少一个原稳定失败根18/30/55出现“source三次失败、G三次成功”的有效槽位，可以推荐下一次TRAIN自身访问示范对照；完整8根和全部24槽位仍同时交付，不能只报该阳性。若只出现source可救，下一步围绕保持/交接；若均无可靠见证，暂停这条补标签路线。不同根出现不同类型时并列报告，不强行归结一个原因。即使有可靠G指导，仍只证明老师可救；“学生能拟合并改善闭环”要由下一次独立、预声明预算的学习试验验证。本轮不能判定容量是否不足。

执行工作清单（完成状态只由实际证据更新）：

- [x] 核对原始节点、配对旧损失、老师—学生矩阵、真实更新数和身份；更正weight_ratio报告的R5失败根文字，保留更正说明。
- [x] 在已有retention入口原位接入学生访问状态采集和剩余窗口查询；不另造按日期/根ID硬编码的运行逻辑。
- [x] 通过必要行为测试：j的前后步约定、H16-j与原交接时刻、完整状态/历史/时间、终止与缺失、不复注入扰动、17世界和固定赢家复核、真实费用/超时/无优化更新；静态prepare/audit不得隐式启动GPU。
- [x] 冻结实际π0/G73/BC1000、TRAIN8、配置/物理、代码、完整布局、输入/授权与独立时钟；声明真实run目录和命令，核对不修改其他项目进程。
- [x] 启动单次诊断，保存逐槽位状态和费用、错误/完成watcher；无训练reward，不用event文件或服务器进程冒充训练进度。本阶段报告已完成根/查询、截止与running/completed/error/partial；如输出TensorBoard诊断标量，检查HTTP及本run实际标量。
- [x] 交付全部8根/24槽位同状态矩阵、源/候选/复核、实际XY与原始NPZ、真实失败/终点及active/padding成本；检验固定接口，按上表决定下一试验并停止。
- [x] root在共享锁内更新唯一台账，审查diff、相关测试、commit/push并给出回执；本段没有授权未来流水线自动启动。

月底投稿安排是条件性证据路线，不是成果承诺或无界实验许可：

|日期|目标与必须得到的证据|进入下一步的边界|
|---|---|---|
|10/11–14|完成本轮机制诊断，定位指导分布、保持/交接或重复性中的可操作问题；后续在用户本轮任务范围内只验证一个机制修复|诊断有效且存在可验证指导后才设计学生学习；失败/UNKNOWN保留，不能用更多模块填补证据|
|10/15–20|若机制修复过有界开发门槛，再预声明独立训练种子的等成本G教案学生、Noise教案学生与π0对照|同起点/尾部/物理/动作权限/成功规则，计采集、搜索、复核、学习和评价总成本；报告新/旧损失/保留，G与Noise教案支持差异单列；当前未登记执行配置和预算，不从本表直接启动|
|10/21–25|冻结方法、模型选择规则和最终评价协议后做评价及迁移身份核对|固定模型/normalizer/动作/时序；区分仿真评价、数值复现和新环境迁移。未完成同身份核对不宣称迁移；TEST仍需单独明确授权后才打开|
|10/26–31|按实证强度完成论文、图表、成本/失败/复现附录与引用核验|主张限于实际完成的独立证据；若学生可靠净提升或G等成本优势未成立，收窄结论，保留负结果，不将老师见证包装成最终单策略成功|

当前可写：有限固定root中存在G恢复教案和部分学生吸收，同时持续存在旧能力损失及数值标签翻转；覆盖和权重两次局部干预未建立可靠保持改进。尚不能写：扩散必要、容量不足、最终单策略稳健净提升、未见TEST泛化、自动外循环成功或严格可达边界。D1的G与Noise互有独有见证只支持互补性线索；后续学生优势必须另有等总成本和独立种子证据。

本轮实现验证：44项定向CPU行为测试通过，包含snapshot字段与terminal排除、剩余4/8/12/16窗口、历史默认接口、17世界预算与授权锁、搜索无赢家但复核有成功的歧义保留、部分/重复query矩阵拒绝。两位独立审查者反馈已修正；尚待真实GPU执行，CPU通过不证明恢复效果。sim2sim身份核验见 `../evidence/retention_next_20261010/sim2sim_identity_audit.json`：双引擎正结果属于PhysX训练后01966080而非本诊断4988928源π0，且成功口径不同。


## 2026-10-11 诊断完成后的单因素教案学习

完整24槽位诊断与独立80轨迹审计已完成，实际97801charged（active71025/padding26776），累计975315/2M；12 G增量、6 source可救、4未找到解、2歧义。历史稳定失败祖先18/30/55上的4个合格query为18j4、30j4、55j8、55j12，各自source000/固定G111；这只是有限TRAIN老师见证。旧coordinates仅祖先metadata，实际导入使用真实snapshot/trace；未来导出显式parent_coordinates。原执行90a3ddc及不可变原记录保留。完整结果、全部8张XY与独立审计位于B_self_visit_001/report。

依上节预声明分支进入一次有限学习试验，仍属用户本轮直接推进授权。原525示范帧保留，加4条固定第二复核真实成功suffix228帧，共753；不加入学生失败prefix、不平均动作、不用DEV回流、成功祖先或歧义。8祖先等权，祖先内各轨迹等权，各真实G前缀/π0尾部各半；所有逐帧来源、准入/排除、58输入锁保存。此分布干预包含祖先内新增轨迹的权重分配：本次原教案采样质量19/24、新指导5/24（20.8333%），228/753不是实际采样比例；不能解释为同一采样分布下仅增加帧数。

从原π0/fresh Adam/RNG开始，原网络/normalizer/critic/Adam1e-5/clip1、demo1/keep1、每组batch256、seed1010267101和keep6289帧均固定，只变TRAIN教案支持。最多1000BC，同0/100/500/1000节点和旧控制比较；每节点完整64DEV+7solver+8TRAIN、原B/D>5pp复核停止、7solver选候选/最早平局，候选8TRAIN两次固定复核全部保持。旧对照BC2000只作历史上下文，不将不同更新预算当单因素结果。

新增物理预算645200=306000节点snapshot+76800fullDEV+153600警戒复核+108800选中TRAIN复核；此前975315，最坏1620515/2M，独立3h。零节点及π0/R5基线同身份复用，不采新keep，0 PPO/E/G，不恢复旧外循环、不开TEST、不自动采用。保留退化/歧义/无增益与真实停止点；只有独立学生改善且保持警戒通过，才支持后续预声明多种子和等总成本G/Noise对照。当前状态以共享台账与实际run为准。

执行分工：root负责协调器/冻结配置/有界启动和台账；训练审查负责示范准入、逐帧来源及报告；独立证据审查负责80轨迹/来源身份及学习接口审查。首诊断代码已push90a3ddc；学习阶段须通过定向CPU测试、真实输入file-only prepare/audit、独立审查后另commit冻结再启动。

性能复用：本学习阶段沿用原17世界snapshot/64世界DEV批量入口、warmup_actor实际batch256；进程间共享本阶段持久JAX编译缓存，复用身份一致π0/R5/零节点物理证据。STTW的DirectCommandPPO/scan(50)/几何vmap属于不同控制器与任务，不移植其数值语义。本阶段不引用历史加速倍数，实际耗时由launch/status和逐子阶段日志记录。
