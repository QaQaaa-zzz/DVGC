# JIT 双向学习 v1.1：实现、迁移审计与准备报告

本交付是在现有 JIT 上新增的可选实现，没有原来的 v1.0 实现需要迁移。已按顺序阅读更新指令、完整合并报告及同版 YAML，并额外采用用户提供的奖励与损失补充。代码位于隔离分支 `agent/generative-bridge-v1-1`，基础版本为 `c98701a`。历史训练、配置、模型、结果和当前运行指针未改写。

2026-09-28 继续实现：用户要求完成并开启训练，随后授权由实现方确定合适来源和预算。现已补齐源绑定的生产编排、老师 32 候选搜索/重放、G 预训练、学生 PPO/采用及 G 回流更新入口。首轮采用已核验的 lineage_repair_0093；保留全部 162 个原 pending 的补训资格，只对预声明 32 个新根开展本轮老师搜索/独立学生评价，其余根记录 not_scheduled、student_label=null，不虚构评价结果。

声明上限为 698,400 次物理交互（4,800 工程验证＋12,800 真实补采＋680,800 首轮模板上限）、G 20,000 次预训练＋2,000 次增量更新、学生 PPO 128,000 步、含排队在内 12 小时。到限停止，不自动追加。运行按 GPU 资源门禁顺序执行，不中断其他项目。

CPU 相关回归已通过 109 项（19.89 秒，[本轮 JUnit](cpu_production_tests.xml)）；真实来源绑定和 37 个源策略保持锚点的加载已核对。GPU 物理验收与研究性能在此报告更新时仍未产生结果，不能用 CPU 通过代替。启动状态以新运行的 status.json 为准。

## 2026-09-28 来源复查中断修复与恢复

pilot_0001 停在 teacher_search：历史 pending 的 32 个固定评价根，当前源 Actor 复查为 19 失败、13 成功、0 unknown。旧 guard 将合法的源成功当作错误；当时学生 PPO 尚未开始。改为 not_scheduled，reason=source_recheck_succeeded，保留全部 162 个原 pending 及原采样配置；未知结果仍属于工程异常。历史标签与当前复查标签分别保留。当前源已成功的根不能记老师救回，也不能领取学生新增 0→1 的 +2；失败 −0.1、真实扰动物理失败 −2、其他奖励权重均保持原值。采用检查继续使用当前源复查结果作基线。

新增显式 recover 入口，只迁移这个已诊断、尚无学生执行的失败状态。恢复写入独立目录，冻结旧凭证及其文件哈希，复用 17 个已完成阶段凭证、2 条已验证教案与 G update_04000；不搬用未提交的 teacher_search.running。继承物理交互 34,931、G 更新 20,000 及原 started_unix；总上限仍为 698,400 / 22,000 / 12 小时，排队与修复时间不清零。原配置、结果、失败矩阵及全局来源指针不覆盖。

独立审查发现旧 CLI 用 YAML 解析 JSON，导致 1e-05 的数值类型与文件不同；恢复精确保留原执行合同，并校验原合同 SHA，今后 JSON 按 JSON 解析，不做宽泛类型归一化。真实旧运行的 CPU 恢复预检已通过：成功导入烟测、语料、G、源 nominal、core/protected 与 teacher_source，未重复任何物理或优化操作。

CPU：116 项相关回归通过（20.17 秒，[JUnit](cpu_recovery_tests.xml)），包含当前源成功无教案、unknown 拒绝、奖励增益/失败项和恢复输入/凭证篡改检查。GPU：旧 pilot 已完成源/桥接/独立重放语义烟测，2 条教案独立重放成功；本次修复尚无新 GPU 结果。研究性能：学生未训练、未采用，不能据此声称能力提升。新恢复状态与成本以运行目录 status.json、costs.json 为准。

## 2026-09-28 同批量独立重放修复

recovery_0001 在第18个根、候选11停止：31路搜索成功（47步），单路重放物理失败（39步），身份、初始观测、前16步动作相同。改为保持原31候选、顺序、lane、前缀与续接Actor进行独立完整批量重放，然后只核验原先选定的候选，不重选直到碰巧成功。旧单路失败和invalid记录保留，不降低成功标准，不将invalid伪装为正常无解。

真实GPU诊断 replay_diagnostic_0001 原样重复搜索批量：候选11成功（48步）。仍有候选10从成功变为失败，因此本修复只消除批量布局变化这一额外差异，不宣称确定性、逐位等价或鲁棒成功率；后续选定候选若重放失败仍停止。新增检查覆盖31路完整性、lane顺序、诊断执行命令/输入锁和已保存轨迹成本。CPU相关回归121项通过（19.95秒，见 [JUnit](cpu_matched_replay_tests.xml)）。

恢复支持上一代恢复目录：复用9条已验证教案、原G和已完成阶段，保留失败记录。诊断的12,400计费交互并入累计85,164，恢复起点为97,564；G更新仍20,000，原开始时间及12小时截止点不变。剩余9个源阴性根最坏搜索+重放223,200，另预留学生/采用258,400，合计579,164≤698,400。已消耗和预留不能混称真实交互。

真实运行CPU预检调用完整teacher_search，成功复用旧结果并通过第18根，然后在teacher_0018首次新GPU调用前人为截断：零新增物理执行。教案原proposal路径显式继承，避免下一代恢复丢失祖先凭证。本次只处理generative_bridge；其他项目未修改、未启动。

## 2026-09-28 三轮固定Actor、连续G实验

用户要求再做多轮，并明确接受建议：先固定同一个源Actor，G连续学习；采用不通过仅记录，不停止后续轮次，不自动采用退化学生。三轮学生都从lineage_repair_0093初始化，老师尾部同一Actor；即使某轮通过采用，也先作为候选保存，不改变这组固定Actor实验的后续来源。

实施计划：先一次性预声明3组互不重叠的32个新TRAIN祖先根，保持全部162个原pending支持和采样；继承上一完成轮的G完整状态与合格语料；每轮搜索/独立重放、PPO128,000、原四面板采用检查、G最多2,000更新；拒绝采用不停止循环，工程错误/冲突/到限停止，不自动追加。core/protected/nominal源Actor基线、语义烟测和固定G开发噪声复用已提交凭证。旧结果不覆盖，正式TEST不打开。

预算为新增物理每轮最多1,060,000、总3,180,000；学生总384,000步；G总6,000更新，无重复20,000预训练；新三轮整体墙钟12小时（含排队）。旧物理331,449及监督22,000单独列为历史成本。每轮最坏实际声明阶段合计1,000,400，在每轮上限内。

代码series.py管理有界串行轮次、预选根、累计成本和结果索引；production.py接入连续G、固定开发夹具与全组历史继承。旧teacher_new/actor_new转换为下一轮history，学生原采用凭证保留；未采用学生仍不准入G。首轮启动历史为20条原历史+19条老师轨迹，共39条；96个新TRAIN根经过真实来源预检。

CPU相关125项通过（19.87秒，[JUnit](cpu_series_tests.xml)）；含拒绝学生仍完成3轮、数据组完整继承、根去重/祖先split、启动失败如实落盘。独立审查未发现阻塞性问题，启动错误状态记录建议已修复。三轮不同根难度不同，不能单凭各轮成功率升降证明G变强；固定Actor和固定开发数据消除部分混淆，等总预算对照仍未完成。

## 修改清单

| 位置 | 实现与作用 |
|---|---|
| `generative_bridge/contracts.py`、`outcomes.py` | 锁定原 pending 全集与采样契约；分开记录 verified_solution、searched_no_solution、not_scheduled、incomplete、invalid；学生结果不能倒写老师结果；输出结果矩阵。 |
| `data.py`、`feedback_data.py` | 真正执行动作和动作前后观测、TRAIN/祖先 split、完整成功、精确 Actor/normalizer 采用凭证准入；无解/未搜索老师不限制合法学生回流。history 中保留旧学生采用凭证；三组按祖先→轨迹→窗口采样，空组重新归一。 |
| `student.py` | 复用原 Actor 恢复及原 PPO 数值保护；同一 loss 挂接可选 demo 和保持项。整轮空 demo 不创建目标、采样器或示范 RNG，继续 PPO 和声明保持项；真实执行计数区分 JIT 跟踪和消费。 |
| `network.py`、`diffusion.py` | H=16、76D 条件、4D 动作、3,131,716 参数 U-Net；100 步 epsilon MSE、DDIM20；增量更新最多 2000 步，固定开发噪声，incumbent 参与选择，平局保留 incumbent。完整 params/EMA/optimizer/RNG/normalizer 保存恢复。 |
| `proposals.py`、`teacher.py`、`rollout.py` | 32 候选生成、整段真实动作差分代价、完整成功优先、独立重放状态校验；第 16 步转同一续接 Actor 的接口。source-only 重查成功记录 not_scheduled，保留补训资格，不能记老师救回。由 production.py 连接到身份绑定的 GPU 执行及独立重放。 |
| `protocol.py`、`artifacts.py` | 基于回调和持久凭证的阶段编排；预留 PPO/评估预算后安排老师。G 失败可显式重试、记累计成本，不重复已提交 PPO；恢复重验语料和实际 G payload，最后提交完整轮次指针。 |
| `rewards.py`、原 `pulse_exploration.py` | 只在显式启用新契约时应用用户奖励；原 pending 构造和原补训规则保留。增加独立成功轨迹原始观测录制开关。 |
| 原 `pulse_exploration_runtime.py` | 关闭新开关时保留原行为；开启时记录真实 pre/post observation、动作及来源代码，允许显式桥接前缀后无重置续接。该物理执行路径尚未做 GPU 验证。 |
| 原 `training/formal.py` | 在原训练器中接入一个可选联合损失适配器，拒绝重复叠加旧 retention。 |
| `cli/run_generative_bridge.py`、配置 | 审计、准备、显式来源绑定、run-pilot、独立 G worker；原模板和禁执行奖励 overlay 保留。 |

原 Actor 76D/256×3、critic 106D、动作顺序、控制时序、物理模型、原任务奖励及后扰动重置语义没有借此更改。补训资格取原支持全集，老师子集仅提供附加教案。老师正常无解或本轮不安排搜索时可以没有示范；工程错误与执行不完整不伪装成正常无解。

## 奖励与损失

用户补充对应 `conversion=0`、`adoption_bonus=2`、`novelty=0.02`、`repeat=0.02`、`failure=0.10`、`pulse_failure=2`、`teacher_success_bonus=0`。+2 要求学生 0→1 且该精确模型最终通过采用；局部成功但整轮拒绝不拿 +2。已执行学生失败为 −0.1；实际施加扰动导致物理失败为 −2，即使没有可量化几何 cell 也不能漏记；unknown 不当失败。该反馈与原 Actor 任务奖励是两个接口。

G 使用全元素平均 epsilon MSE，不把分组采样权重再乘入损失。PPO 保持原目标，示范系数按本次训练转移从 0.2 线性衰减至 0.05，保持项 0.2 使用冻结源模型及其 normalizer。空示范情况下示范损失关闭；G 没有新合法窗口时参数、优化器和 RNG 都不推进。

## 首批来源和迁移审计快照

详见 [来源审计](source_audit.json) 与 [迁移审计](v1_to_v1_1_audit.md)。只读检查了用户 IDE 指定的 current_source.json，**没有把它自动选为新实验来源**。

- 审计指针指向 `lineage_repair_0093`；Actor 153,352 参数，critic 159,233 参数。Actor、critic、normalizer 哈希通过核对。
- 此来源 pending_fraction 是 **0.5**，重置为 20% 固定跳跃起点、80% 完整快照；不是套用其他历史版本的比例。
- 审计第 95 轮 162 个有效 pending：学生成功 80、失败 82；整轮采用为 false（旧正例保持 10/12，低于 0.9）。这 80 条局部成功不能冒充已采用成功。
- 被审计的旧后缀 NPZ 缺少原始动作前观测。本次可准入 G 的真实新窗口为 **0**；该结论仅针对列明的审计文件，不推断全部历史档案均不可用。
- 尚缺指定的新实验源 Actor/G、锁定 TRAIN/dev 祖先划分、core/protected/solver 开发面板及执行预算。没有自动补采、打开 final TEST 或选择“最新”模型。

## 验证与回流示例

首批 CPU 相关回归：**95 passed，17.58 秒**，无失败/跳过。原始 [JUnit](cpu_tests.xml)、[分类测试结果](test_results.json) 可复查。覆盖新模块以及原探索奖励、策略保持、蒸馏、PPO 数值保护、探索和分批评估接口；不是整个仓库所有测试。

空数据测试包括：空教案保留 PPO/keep、不实例化示范采样器；零附加权重回归原 loss；真实安装的 Brax PPO loss/原网络在构造 Transition 上做受保护优化；无新 G 数据完全 no-op。恢复测试包括完整训练状态和下一步等价、G 失败后不重复学生补训、编号重试累计预算、语料及已提交 G payload 篡改拒绝。

[数据回流示例](fixture_evidence/generator_corpus_update.json) 是明确标记的 **CPU 构造夹具**：老师 searched_no_solution，学生 actor-only 的 18 步完整成功，通过精确采用凭证后形成 3 个 H16 窗口。它展示准入路径，不是物理成功轨迹，未加入生产训练数据。另有 [15 格状态矩阵](fixture_evidence/result_matrix.json) 和 [逐根记录](fixture_evidence/bidirectional_root_outcomes.jsonl)。真实老师成功、真实老师无解但学生成功两类新物理示例均未采集；不能把夹具当研究证据。

| 证据层 | 本次状态 |
|---|---|
| CPU 逻辑、损失、架构、持久化恢复 | 已通过上述 95 项测试；架构做实际初始化、前向和梯度检查 |
| GPU 物理：恢复、H16 切换、重放、PPO | 未运行；CPU 测试不能证明物理等价 |
| 研究性能：生成器价值、回流收益、独立种子、最终测试 | 未运行；没有性能结论 |

## 模型版本和成本账本

[模型/依赖版本](model_versions.json) 记录审计模型身份、固定架构、基础代码版本和环境版本；本次没有新的训练模型版本。实现版本以包含本文件的 Git 提交为准。

[新实验成本账本](preparation/cost_ledger.json)：新实验物理交互 0、正式监督更新 0、GPU 物理/研究实验 not_run。CPU 测试确实执行了夹具优化步骤，**不计成实验更新，也不宣称它们没有计算成本**；本次累计 CPU 夹具更新数与全过程墙钟未统一计量，最终测试墙钟由 JUnit 记录。

[五臂对照计划](preparation/comparison_plan.json) 给出每臂 680,800 次物理交互上限的示例拆分；PPO-only 将省下的搜索预算分配回原 PPO（547,200）与原采集（3,200），共同诊断/采用为 130,400。其他臂为 PPO 128,000、搜索 422,400、共同诊断/采用 130,400。所有臂保持同一源任务/重置及保持项；新旧 G 比较必须固定同一个续接 Actor。

这只是物理交互预算对齐的准备表，**不是已完成等总成本实验**：监督更新、开发评估计算和墙钟预算仍须共同锁定。示例数字不构成启动授权。G 增量墙钟上限采用步骤边界检查，单个计算/开发评分可能越过截止点；生产启动器另设持久起始时刻和全局截止信号；子进程门禁捕获超时后终止自己创建的进程组，预算不因恢复重置。

## 当前执行顺序与限制

1. 已固定源 Actor/normalizer、完整支持池、TRAIN/dev 祖先划分及预算；G 从声明的真实轨迹补采后冷启动预训练，未继承不存在的旧 G。
2. 已完成源绑定的生产调度：老师 32 候选执行/选中重放/教案导出、初始 G 预训练日程、真实采用面板及各阶段凭证连接。已补齐的 `production.py`/CLI 现连接这些阶段；实际 GPU 通过情况仍须核对运行记录。
3. 先在声明的 4,800 交互上限内做 GPU 物理语义验证，再进入声明训练；核验源恢复、原 pending 采样、H16 无重置交接、空 demo PPO、真实 TRAIN 回流和重启恢复；启动时按 JIT 规则启用错误/完成通知。
4. 工程验收后再按锁定的等总预算方案实施对照；新旧 G 使用同一续接 Actor，分别报告物理交互、监督计算、开发评价和墙钟。未经声明不自动进入大训练。

可复现 CPU 命令（仓库根目录）：

```bash
PYTHONPATH=JIT/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest -q JIT/tests/test_generative_bridge_*.py JIT/tests/test_discovery_conversion_reward.py JIT/tests/test_policy_retention.py JIT/tests/test_policy_distillation.py JIT/tests/test_ppo_numerics.py JIT/tests/test_pulse_exploration.py JIT/tests/test_pulse_evaluation_batches.py JIT/tests/test_probe_training_action_pulse.py
```

准备命令使用新的输出目录，不覆盖既有产物：

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python JIT/cli/run_generative_bridge.py prepare --spec JIT/configs/generative_bridge_v1_1.yaml --output /tmp/jit_bridge_prepare_new
```


## 生产编排补充与审查

- 新增 `production.py`：按祖先锁定补采 TRAIN/dev、32 个新根、64 core、64 protected 面板；原支持池字节不变。采用判断固定名义成功、core 无新增失败、protected 保持≥0.98、净新成功≥1，unknown 不放行。
- `worker.py` 在隔离 GPU 子进程预训练/更新 G；CPU supervisor 负责生成候选与调度，避免占用 CUDA 导致自己的资源门禁永远关闭。
- 预训练新增 warmup1000＋cosine 至 1e-5；累计已计费更新独立于被选中的 checkpoint 年龄。全状态恢复与失败尝试分别记账。
- 增量失败只能显式 `--retry-generator`；凭实际失败尝试收据核销预算后使用剩余额度，不重做已提交的学生训练。无新合法数据跳过 G 子进程，保留 incumbent。
- 生产 Git commit 和受跟踪源码文件哈希均锁定，资源排队期间及每个子进程启动前重验；代码变化会停止，不跨版本混跑。
- 所有 162 个原 pending 输出状态；面板以外 130 个保留训练资格但不把未评价当失败。错误/执行不完整另存失败结果矩阵。
- 本轮是已有 TRAIN 根上的桥接/补训试点，不新增原探索器训练。用户奖励按原来源 arrival ledger 生成明确标记的反馈诊断；Actor 任务奖励和 G epsilon MSE 保持既定契约。
- 五臂等总预算对照仍是后续预声明计划，本次单轮不能证明 G 优于对照。新旧 G 性能比较须另用同一冻结续接 Actor，不挪用训练结果作为独立测试。

## v1.2 源替换与累计学习（2026-09-28，启动前审计）

用户明确要求停止旧训练并执行新包。旧 series_0001 已 SIGINT 停止；第1轮完成，第2轮中止，原始错误/结果与898,597 charged-or-reserved成本保留。停止原因单独写入 stop_request_v1_2.json / stop_verified_v1_2.json。

新源严格固定 Phase U `phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928`。新绑定保留Actor/normalizer，64个固定观测动作最大差0；原源训练4,988,928步，新绑定训练0步。原奖励、XML、动作映射、H16、256×3 Actor与U-Net均保留；完整任务仍25步稳定恢复。绑定模板里的旧支持只用于运行时结构，不导入新实验标签。

增量功能：新源独立TRAIN/G-dev/student-dev采集；完整起点0/1/2步脉冲及快照禁用；逻辑episode随机键；累计学生教案；独立保持Teacher；Actor-only预热与物理开发选择；实际PPO三项Actor梯度、KL和固定动作probe；完整任务三臂开发检查；四组合重复诊断；采用后学生真实TRAIN成功回流G。旧 fixed-Actor series 默认行为不变。

A0上限1,600物理步；A1上限1,820,000（含64个独立G-dev、采集额外3步、重新录制名义支持、整批老师核验）；A2上限2,600,000。总4,421,600物理步、24小时，G最多22,000、预热最多2,000更新。PPO三臂各128,000；单种子9281201诊断，不声称等总预算优势。预热选择只用8个预先声明solver-dev快照续接，不冒称学生完整起点成功。A/B是对照，C为预先指定主候选；仅C采用轨迹进入主G反馈。

CPU检查与实际GPU验证分开：CPU源动作一致性通过；名义GPU任务目前在资源门控队列中，未声称成功或新学生已经学习。首损失KL超过声明0.05（扣除安装版self-KL偏移）将中止并保留诊断，不静默调整奖励/学习率。推理checkpoint不恢复优化器和episode计数。

Stage B尚未启用；学习探索器优化与新tail跨轮累计迁移不包含在本次A阶段执行中。仅在新源语义、示范吸收、至少一位学生采用以及教案对照收益门槛通过后另行启用。最终TEST保持未开启。

## v1.2 后续外循环准备边界

A阶段生产运行冻结于16d4641，运行目录campaign_0001；A0名义物理检查仍在等待GPU资源，物理与优化计数均为0。TensorBoard http://127.0.0.1:6017 HTTP已核验，尚无真实训练标量，不能据此称训练已实际开始。

新增可选的采用谱系校验与累计教案迁移，保留原始teacher tail/normalizer身份；接受凭证必须绑定真实checkpoint及连续谱系。新增有界promotion编排协议和RSL探索采集/更新准入：每回合20%概率外部均匀覆盖，均匀分支不提供伪造策略log probability，只有当轮学习分支进入探索器PPO，更新凭证禁止重复使用。默认旧流程不变。

这些是Stage B准备模块，未接入完整生产callbacks，未执行Stage B GPU训练；完整生产适配仍需完成。不得把CPU协议测试称为外循环已运行。新增模块不改写正在排队的code_a1快照。

本次准备模块验证：相关全套187项CPU测试通过（31.96秒）；随后episode ID 0边界修正及旧series回归11项通过；探索准入与真实RSL CPU更新补充18项通过。各测试集有重叠，不相加为总数。生产适配仍缺真实explorer admission/NPZ到coordinator的凭证验证桥接、采用Actor到累计教案的tail receipt桥接；Stage B保持关闭。

G条件来源现在可通过generator_reference_frozen_policy的path/sha256独立绑定：worker训练与teacher候选生成共同使用该固定来源，校验manifest、真实checkpoint的Actor/normalizer哈希及同XML；老师续接仍使用当前source_frozen_policy。默认不指定时保持原行为。该接口不等于已完成Stage B生产适配，也不改变排队运行。

上述G来源解耦后，相关完整CPU回归191项通过（32.26秒），包含worker和teacher实际调用路径选择测试。GPU验证仍待资源门禁。

## 2026-09-29 名义验证启动错误与显式恢复

campaign_0001 在 A0 失败，未进入 G/PPO。配置 num_envs=4、pulse_steps=3 与历史 nominal_source_rollout 的单条/全horizon合同不符；日志在reset/step之前抛ValueError，无prefixes轨迹。新增显式nominal_repeats，仍要求全400步、零残差、无探索器checkpoint；旧默认保持单条。prepare_campaign在创建运行前CPU预检该合同。

相关CPU回归181项通过（29.54秒），含四次名义重复、错配次数/非零扰动/3步拒绝。执行代码254d670；独立code_recovery_0001快照与recovery_0001/campaign已启动，先做GPU名义验证。源transition_4988928、奖励/物理/PPO预算未改。原失败工件保留，旧1,600预留与新4,420,000上限合计不超过4,421,600；截止沿用旧A0开始+24小时，不重置。

恢复入口为JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0001/INDEX.md；TensorBoard端口6019，HTTP200已核验，尚无训练标量。通知监视器有实际心跳、无投递错误。GPU成功与学习收益仍须实际结果，不能用本次CPU检查代替。

继续核验：recovery_0001的A0实际GPU四次均成功，计费1600（active309/padding1291）。随后seed_support继承nominal_repeats4却自行改num_envs1，被同一合同拒绝；新增真实调用回归先复现，再令seed_support显式nominal_repeats1。30项相关CPU回归通过。执行代码2554c75，recovery_0002已启动并实际进入GPU fresh_seed_support，复用锁定A0结果不重复仿真。旧失败预留1600+80400单列，新尝试物理上限4,339,600含复用A0计费，合计仍4,421,600；旧24小时截止保持。当前TensorBoard6020，通知和实时状态在recovery_0002/campaign。G/PPO尚未开始，后续阶段未验证。

## 2026-10-08 完整有限对照翻转隔离与恢复

recovery_0002在第二个老师根失败：选中候选6搜索和重放均成功，仅source-only lane0从失败翻转为成功。旧代码把选中候选label改成None并停止整轮。现保留候选真实label，将完整有限的source翻转单独隔离为invalid/reason=source_control_repeat_conflict；不准入该教案、不计老师新增收益，原pending训练资格保留。只有完整有限重放且选中候选仍成功可进入此分支；未知、不完整、真正老师重放失败仍停止。采用面板冲突source保持unknown，未解决前不伪造采用通过。

新增recover-v12只迁移经证据验证、学生尚未开始的这一失败路径。校验旧合同/数据轨迹/G/开发夹具/教师凭证，复用已完成采集、G update_20000、第一条有效教案、旧search/replay阶段；旧失败记录不修改。真实CPU恢复预检已抵达teacher_0002前的首次新GPU请求，旧冲突根隔离、旧教案保留；新增物理与G更新均0。185项相关CPU回归通过（30.47秒）；CLI数值预算兼容修正后18项相关回归通过（3.00秒），测试集不相加。

当前执行recovery_0003/campaign，冻结代码d26af57。已实际进入teacher_0002。继承物理计费/预留541504，全部尝试总上限4421600、剩余3880096；继承G20000，余G增量2000、BC2000，三臂PPO各128000未改。原窗口已过期，用户本次明确续跑授权下重新声明24小时窗口；历史成本不清零。TensorBoard6021独立新运行目录，不把继承G日志冒充新更新；通知心跳正常。学生尚未开始，GPU后续/采用/性能结果待运行。

## 2026-10-09 worker completion and continuous metrics

The last-valid incremental G worker can finish training and write its final
state, then abort inside CPython finalization with `remaining subinterpreters`.
The CLI now uses an isolated-process completion path: normal worker return,
async-effect barrier, full checkpoint/parent/update/billing checks, filesystem
sync, an explicit `process_commit.json`, then process exit without native Python
teardown. Failures before commit propagate. This contains the observed teardown
failure; it does not establish which native library caused it.

`finalization_recovery.recover_finalization` is an explicit, narrow metadata
publication recovery. It accepts only the diagnosed final G shutdown signature,
validates the same-round stages and full learner/G state, retains all original
failed artifacts and billed costs, and creates a new publication-only container.
It does not rerun physics or optimization and cannot be passed to the training
runner. A continuation inherits only its audited whole-round bundle.

`export_tensorboard.py` accepts distinct source `name` and optional shared `run`,
plus nonnegative `step_offset` and optional `step_offset_receipt` pointing to
the actual learner initialization field. A missing future receipt defers export;
an unequal or noninteger offset is rejected. The cumulative student axis is local training
transitions plus the saved learner initialization offset; it is not total
physical interactions. Duplicate coordinates across different sources are an
error, rather than silently merging different attempts. Historical files remain
unchanged. R3 starts this learner accounting lineage at zero; earlier historical
training is not fabricated onto that axis.
