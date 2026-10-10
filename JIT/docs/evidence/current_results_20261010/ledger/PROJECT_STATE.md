# PROJECT_STATE.md — DVGC-JIT / STTW_CONTROL 研究总台账

> 这不是成果宣传材料，而是两个项目的可持续研究记忆。
> 本地唯一事实台账；2026-10-08 已做有界只读校准，范围与未知项如下。
> 保留初稿历史、用户目标及反例；当前事实以本次核验块为准。未重跑训练、物理或完整旧评价。

## 0. 元信息与快速总览

```yaml
schema_version: 1
state_revision: 182
initialized_on: 2026-10-07
last_updated_on: 2026-10-10
last_local_audit: "2026-10-08T09:10:13+08:00"
last_result_ingestion: 2026-10-10
history_coverage: PARTIAL
deployment_status: INSTALLED
research_hub: /home/qy/STTW_CONTROL/research-hub
sttw_repository: /home/qy/STTW_CONTROL
jit_repository: /home/qy/DVGC
jit_subdirectory: /home/qy/DVGC/JIT
```

上述路径本次已确认。共享规则为 `/home/qy/STTW_CONTROL/research-hub/AGENTS.md`；本文件 `/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md` 是唯一研究事实维护入口。后续单项目更新不得顺带刷新另一项目核验时间。

| 项目 | 长期目标 | 当前明确方向/待核验边界 | 最近结果核验 | 当前阻塞 |
|---|---|---|---|---|
| STTW_CONTROL | 稳定控制与驱动—转向协同 | 无alpha局部底层best300首次7/7通过；冻结V5.2两个upper best | 2026-10-10：300完成、39,321,600转移；Q1.739090，负向稳态长期偏差消除；R196同流3/7 | 350退回6/7；best300仍7/7且旧状态恢复已过；连续合格未满足，剩余至400澄清；未采用/未组合 |
| DVGC-JIT | 有效扰动探索与策略改进，提高最终跳跃策略泛化 | 原200轮连续任务series0011；实验继续采用不等于正式能力采用 | 2026-10-10有界快照：67/200，R71完整、R72状态running；R71 DEV 228/288，P0 186/288，初始C 228/288 | 能力保持与重放有效性仍需核查；完整历史/图片索引见JIT-AUDIT-20261010-HISTORY |

两个项目并行、分别管理证据，不互为完成前置。规划章节、Sim2Sim 或新算法路线不自动纳入本轮工作范围。

历史投稿安排：用户 2026-09-24 提出 DVGC-JIT 在 2026 年 11 月前、STTW_CONTROL 在 2026 年 12 月前投稿。本次未重新确认期限，不能伪装成已验证可达的进度承诺。[S6]


### 0.1 本次核验与仓库映射

本轮 ID：`HUB-INIT-20261008`；核验时间：2026-10-08T09:10:13+08:00；主机 `qy-MS-7E06`。检查 Git 状态、实际作用域规则、当前相关源码与冻结配置、运行 status/receipt/评价 JSON、必要模型身份；没有执行研究训练、物理回放或打开 final TEST。`artifact_checked` 仅限所列文件，保存的 audit 通过不等于本次重新复现。

| 路径/角色 | 分支 | HEAD | 修改前状态 |
|---|---|---|---|
| `/home/qy/STTW_CONTROL`（STTW主仓库） | `feat/residual-recovery` | `4cafb531522f94f4bc79696225f635134947a388` | 未跟踪 .planning/、.vscode/；无跟踪改动 |
| `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3`（STTW当前实现工作树） | `experiment/direct-command-policy-v3` | `fb8f22908fa15b924a8dc1e91d01fdac05e2edeb` | 干净 |
| `/home/qy/DVGC`（DVGC主仓库（含JIT）） | `agent/two-phase-soft-tube` | `c98701a1a84363435977736728d6d94229d427d9` | 未跟踪 .vscode/、FETCH_HEAD、docs/TWO_PHASE_REBUILD_GUIDE.md、typescript；无跟踪改动 |
| `/home/qy/DVGC/runs/worktrees/generative-bridge`（JIT当前实现工作树） | `agent/generative-bridge-v1-1` | `d26af57cf38136624cdd768b96d59bcb7b6de856` | 干净 |
| `/home/qy/DVGC/runs/worktrees/generative-bridge-results`（JIT历史报告工作树） | `agent/generative-bridge-results` | `f131247d7f123020d84560cb2474ae2e4a4d189f` | 本轮只读；状态未做完整审计 |

已检查 `/`、`/home`、`/home/qy` 到上述仓库/工作树及 JIT 层的规则；所查路径未发现 AGENTS.override.md。相关源码/文档目录未发现更深层 AGENTS。主仓库 Git worktree 映射已读；未逐一核验所有历史工作树的算法/状态。不可变运行代码快照不修改。已有所有未提交文件保持原状。

规则差距：STTW当前工作树AGENTS末尾还继承旧“固定端点、alpha不入Actor”实验条款，但当前V3配置明确 one_actor=true、alpha入网；本轮保留旧条款并添加日期/适用范围说明，不能把它当V4已执行的证据。DVGC旧入口的“扩散未实现”和等待队列描述均是旧阶段快照。用户本轮管理要求不授权重启这些运行。

计划与执行：规则/版本定位完成 → 当前合同及结果核验完成 → 有界历史补录完成 → 文档入口/保留性/路径检查见第10节。不另维护 task_plan 或新 CURRENT_STATUS。

## 1. 来源登记与解释规则

| 来源 ID | 内容/定位线索 | 本次核验范围 |
|---|---|---|
| S0 | 用户 2026-10-07 当前任务：两项目并行、各有小论文；三头方案失败；当前尝试上层输出指令、底层跟踪；希望持续维护历史和结果 | 当前用户说明；目标属于 user_requirement，实验表现属于 user_report |
| S1 | `STTW_Codex_FrozenLower_IndependentAlpha.md`，标题“冻结强底层、两个固定 α 独立训练（V4）”；文中代码核对到 `b10c138e0a7abea12c4a05e1fed3870e9564c2bf` | 已查阅关键章节文本；未核验其引用的源码、事件文件和 checkpoint |
| S2 | `STTW_Codex_Direct_Command_V3.md`，V3.0；文中历史 HEAD `90d6a8435da1f1cbcec585e8cf51a9536ffdc053` | 已查阅规格关键内容；该规格明确不是已训练结果，且独立端点决策已取代其中共享网络要求 |
| S3 | `JIT_Codex_下一阶段指令_v1.2.md`，2026-09-28；历史代码 `5941d6bb87f46cb11ce20e46a51dd9b6f0b77f3f`；历史结果 `0147bd1e76a577a6607563846ce579c0d49d14df` | 已查阅关键章节与结果摘要文本；未读取本次服务器状态或原始结果 |
| S4 | 用户 2026-10-02 关于 JIT 探索网络、扩散恢复和向跳跃策略蒸馏的说明；用户要求判断模块必要性和论文逻辑 | 可见历史对话摘要；不是本次代码审计 |
| S5 | 用户历史对话摘要：三头效果差、250 轮模型继续训练表现、扰动后立即让跳跃策略控制、恢复空间与 α 的讨论 | 二手上下文线索；具体日期、run 和原始证据须逐项补充，不得补造 |
| S6 | 用户 2026-09-24 两项论文和中期研究安排 | 历史计划；当前执行情况未知 |

S0–S6 全部为附件初稿继承的来源登记；其中“已查阅”描述的是初稿作者，不是本轮读取了完整旧对话。本轮日期为2026-10-08。S1/V4与S3原全文未定位，保留摘要为 history_document/user_report，不提升为用户新实验授权。S2实际副本位于当前STTW工作树 `docs/direct_command/STTW_Codex_Direct_Command_V3.md`。本轮直接来源见E1–E7。

研究目标以最新明确用户决定为准；实际完成情况以对应版本、运行和结果为准。`history_document` 里的“已验证”只表示历史作者的陈述，不自动升级成本次 `artifact_checked`。


### 本轮直接证据索引（均为本地真实路径）

| ID | 路径与核验范围 |
|---|---|
| E1 | `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/direct_command_frozen_lower500_20260929`：读取 manifest、frozen_config、status、training_summary、review/metrics、review/audit JSON；checkpoint存在；lower actor.msgpack/identity.json SHA实算匹配 |
| E2 | `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/learning/src/sttw_control/`：direct_command_env.py、direct_command_ppo.py、direct_command_audit.py、frozen_lower_controller.py；核对优化器归属、raw/governed和身份断言 |
| E3 | `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0002/campaign`：读取status、production、explorer_provenance、pretrain_result、source_swap_audit、teachers/0001_result与0001_layout_repeat JSON；读取相应冻结code_recovery_0002的campaign/production/student代码 |
| E4 | `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_1_20260928/recovery_0002/actor_acceptance.json` 与 `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_1_20260928/series_0001/round_0001/actor_acceptance.json`：原始采用计数；series_0001/status.json；旧学生两份独立候选均未采用 |
| E5 | `/home/qy/DVGC/runs/worktrees/generative-bridge-results/JIT/docs/generative_bridge/CORE_RESULTS.md`、CORE_RESULTS.json：历史完整对照摘要；老师吸收率只核对报告和其结构化摘要，未逐条重放 |
| E6 | `/home/qy/STTW_CONTROL/runs/discrete_alpha_three_control_20260919/analysis/REPORT.md`、summary.json及训练status；`/home/qy/STTW_CONTROL/runs/asymmetric_priority_comparison_20260918_continue250`的status；仅有界历史检查，不等同三头失败身份确认 |
| E7 | `/home/qy/DVGC/runs/worktrees/generative-bridge/JIT/docs/generative_bridge/REPORT.md`、`/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/audit/binding/initialization_receipt.json`与source_swap_audit；实现沿革、源身份与零训练重绑定；未重新推理 |

## 2. STTW_CONTROL

### 2.1 研究目标、任务定义与论文边界

研究执行器约束下单轨双轮平台的稳定控制及速度—转向任务取舍。希望在正常指令下兼顾两目标，在冲突/扰动下按偏好让步，并在协议规定的恢复阶段回归任务。

`upper_alpha=0`：路径/转向任务优先，必要时允许减速；`upper_alpha=1`：速度优先，允许一定转向/轨迹偏差。安全要求不因 α 改变而放宽。中间 α、共享网络和连续偏好插值属于后续问题，不把独立端点结果当成已证明连续可调。

重要缺口：用户长期希望路径跟踪，但 V4 历史规格主要评价转角、速度和航向恢复，未要求完整 XY 几何路径回归。当前应分别报告“转向指令跟踪”“航向恢复”“几何路径误差”；没有后者证据，论文不写完整路径恢复已解决。[S0][S1]

候选论文主张是偏好明确的驱动—转向协同及其有效机制，而不是“加了一个网络”。抗扰与恢复空间仍是长期主线；是否纳入当前最小实验包，由当前协议和证据决定，不自动扩成多个新研究模块。

### 2.2 当前路线：先修无alpha局部底层，再门控冻结V5.2组合比较

当前权威尝试为 **STTW-LOWER-NOALPHA-20261010**。用户固定lower repair4e956f0、upper V5.2 2eeef42，保护best0新增200/best1新增250。新210/211局部Actor/Critic不带alpha/path/return，从零训练；ECBC/ESO/物理/执行器不变。完整200→300续训已完成，300接受更新/39,321,600控制转移；best300及磁盘重载局部7/7合格，R196同协议3/7；250仍未合格，所以连续两次后期合格条件尚缺一次。负向稳态长期7.835s偏差消除，但旧B0完整状态未验证。旧6.315s完整状态2秒恢复比较已过（.445秒进入并保持）；350已完成，last350退步6/7，连续后期合格未满足；best仍300（重载7/7）。下一步仅剩余350→400澄清50更新，累计52,428,800到达硬上限后停止；不得把300和400跨越350失败算连续通过，冻结上层组合仍门控未运行。尚未注册/安装新底层，禁止本轮上层训练。

**历史H009快照（不再是当前活动路线）：** R196冻结底层lower_alpha=1；SmoothV4独立上层两端fresh150→250，best250、qualified=false，已停止；结果与远端证据保留。

**以下为09:10初始化时的历史路线核验快照，保留溯源；“独立端点仅为目标/共享上层是当前实现”已被本次真实独立实现取代。**

```text
原始速度/转向参考 + 当前观测/历史
                 ↓
上层参考指令网络：输出参考或参考修正
                 ↓
调整后的速度/转向参考
                 ↓
共同冻结强底层：已有残差 Actor + ECBC + ESO
                 ↓
底层控制输出 → 执行器 → 车辆 → 观测反馈
```

上层控制意图与底层执行分开，禁止写成“上层直接输出前轮角速度/后轮电机动作”。底层不是裸网络，也不是没有姿态反馈的纯指令直通。

最新可定位的后续决策为 V4：先训练两个独立上层，固定 `upper_alpha=0/1`；共享同一个冻结底层 checkpoint，而不是共享上层 Actor/Critic/优化器。底层原有 `lower_alpha=1` 与上层 α 分开命名。此处是已明确的方案，不代表新训练已跑完或效果已达标。[S1]

历史指定底层线索：

```text
checkpoint:
/home/qy/STTW_CONTROL/runs/path_rsl_4096_20260915/training/checkpoints/update_0200

declaration:
/home/qy/STTW_CONTROL/runs/path_rsl_4096_20260915/evaluation/alpha_2/seed_49001/force_right/residual/declaration.json
```

身份、归一化、接口、实际冻结状态和是否有更新的用户指定模型须核查，不按“最新文件”自动选模型。新旧决定存在冲突时记录，不静默换底层。

本次实际合同（E1/E2，artifact_checked）：

| 字段 | 已核验实现与差距 |
|---|---|
| 上层 | 一个共享α条件Actor，345维输入，128→128→64 ELU，2个Gaussian latent `[speed, steer]`；独立Critic346维。不是两套固定α独立学习器 |
| 输出/频率 | 相对raw的速度修正[-1,+0.25]m/s与转角修正±0.2rad，限速/限幅后给governed；上层50Hz，底层200Hz；底层最终执行前轮角速度/后轮轴速，均rad/s |
| 底层内容 | 指定update_0200冻结残差Actor＋ECBC＋ESO＋执行器；280维（27字段×10帧＋mask），256→128 ELU。下层不在上层Torch优化器参数中；两个α共用同一个加载对象及身份 |
| 下层身份/归一化 | `/home/qy/STTW_CONTROL/runs/path_rsl_4096_20260915/training/checkpoints/update_0200`；actor.msgpack SHA `950efa65fa5828c0a11119ff41ce0ab3b756be74a31b63ca7973cd487f9083b4`，identity SHA `15c64eca1aba7720c4d0f3855f3b204e0ce6b35b1c013cda99d6cc50d2d15040`（本次实算）；字段顺序/尺度/历史由声明与sidecar绑定，上层不启用running normalization |
| 两个α | upper_alpha=0/1进入共享上层及偏好奖励，每回合固定；lower_alpha恒为1（用于旧底层观测/恢复状态），不随上层切换 |
| raw/governed | raw为slew后原指令，用于外部速度/转角奖励、原航向参考与恢复判据；governed供底层跟踪，并因果积分为旧几何Actor的历史路径，末端使用切线延拓；不拿governed替换外部任务成功标准。B0旁路上层但仍含冻结残差 |
| 物理/权限 | manifest物理步0.0002s×25→控制0.005s；总残差±1.5/±10rad/s。当前XML与旧底层源XML不同，不能称无分布偏移；配置明确无外扰、无XY位置目标 |
| 奖励版本 | frozen_config schema `sttw_direct_command_v3.0`，config SHA `2a8a73b5736e27d1b5b2d115ff71c7d56319c4806f8fb99a0cf68f593685320d`；Huber速度/转角、门控航向、roll/roll-rate、超速/低速、修正与命令变化；scale .1，总cost cap 100，失败替换区间奖励；未发现已落地的V4去总裁剪版本 |
| 当前模型 | 新初始化上层Actor/Critic/optimizer，seed73，500/500；评测 `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/direct_command_frozen_lower500_20260929/pilot/checkpoints/update_0500.pt` 为last completed，不能改称best或采用模型；正式采用凭证 UNKNOWN |
| 当前结果 | main与random各B0/α0/α1，6条16s完整轨迹、无物理失败；main_preference=criteria_not_met，B0和两策略final_hold均false。是速度/转角/航向开发评价，不是完整几何路径恢复或泛化证明 |
| 当前运行 | 该run已complete；本轮未发现所查项目路径下相关Python训练/评价进程。远端/未登记调度作业 UNKNOWN，不从进程缺失推断成功 |

**冲突明确保留：** S1/V4要求两个新独立上层与共同冻结底层；E1仍为shared，且cap100与同步reset设置未按V4改变。当前工作树继承的旧独立残差条款也不是本次V4上层实现。不擅自用任一方覆盖另一方，不启动新训练来补齐。本次16s run是已存在冻结合同，与旧通用10s规则不同；仅如实归档，不据此授权后续默认16s。

### 2.3 研究尝试账本

#### STTW-H001 — 基础控制与学习残差

问题：传统/基础控制在温和指令和困难转向下能力不一致，较弱底层可能妨碍上层学习任务取舍。

尝试：使用 ECBC/ESO，并引入残差补偿；后续选择“已有残差网络＋ECBC＋ESO”作为冻结强底层。[S1][S5]

已知观察：用户/历史上下文报告过温和工况可控、困难条件不足和残差改善线索。本次未核验成组原始结果，不能概括为所有工况均优于基线。

当前结论：保留强底层路线，不无理由退回纯 ECBC。具体改善范围、失败范围和模型身份仍待补证。

来源：history_document + user_report；核验：not_checked（原始实验）。运行 ID/配置/成对结果：UNKNOWN。

#### STTW-H002 — 三头网络未达到预期

问题：希望通过多头结构表达不同 α 偏好并提升控制表现。

尝试：采用三头网络；具体三个头的参数化与完整训练配置待查。[S0][S5]

已知观察：用户明确报告效果不佳/失败，未得到满足预期的偏好控制表现。

当前结论：不作为当前主线；保留负面结果，不删除该尝试。不能把表现不佳自动归因于网络容量不足，也不能宣称三头结构原则上不可能有效。

失败原因：UNKNOWN；奖励可辨识性、训练设置或结构问题只能作为待验证解释。

重新尝试条件：出现区别于原试验的明确诊断、新证据和有界验证目标；不能仅换名字再次训练同类方案。

来源：user_report；核验：not_checked。历史检索线索包括 `feat/residual-recovery`，但本次未建立它与具体 run 的一一对应。

#### STTW-H003 — 长时间续训未必改善

问题：继续增加训练量是否能够解决现有表现问题。

尝试与观察：用户曾报告在已训练 250 轮模型上再续 250 轮效果差，而仅续约 20 轮的版本相对更好。[S5]

边界：这里的轮数单位、checkpoint、优化器/critic 是否恢复、评估场景和可比性均待核实。不能据此断言所有续训无效，也不能把仅加载 Actor 权重叫作 exact resume。

当前结论：不再默认“多训就会更好”；后续每次运行写清 scratch / weights_only / exact_resume、实际新增样本和固定评价。

来源：user_report；核验：not_checked。因果解释：UNKNOWN。

#### STTW-H004 — 上层改为输出参考指令：V3 规格与后续演进

问题：把任务偏好与底层执行分开，让网络直接决定速度/转向参考，而不是再增加解析分配器或直接输出电机动作。

尝试/规格：V3 设计共享 Actor 接收 α 和历史，输出相对原始指令的两个参考修正，通过 ECBC/执行器控制；规格禁止候选搜索、参数网络和额外蒸馏等并行扩展。[S2]

状态：V3 文本明确为待实施规格；后续有代码/训练分析线索，但本次未拼接其完整运行历史。不能仅凭这份规格写“V3 已成功”。

路线变更：保留“输出参考指令”的关键方向；其中共享上层和较弱底层安排已被 V4 的冻结强底层、独立端点决策取代。[S1]

来源：history_document；核验：已读规格文本，原始运行 not_checked。后继：STTW-H005。

#### STTW-H005 — 冻结强底层，两个固定 α 独立上层

问题：先确认端点偏好与真实控制效果，再讨论共享结构；避免底层能力不足和共享训练耦合混在一起。

明确选择：相同冻结低层，两套新初始化的上层 Actor/Critic/优化器，分别固定 α=0 和 α=1；上层输出速度/转向参考修正。[S1]

配套要求：保留用户认可的 KL 过大更新回滚；检查总成本裁剪造成的失真和环境采样同步问题；不额外加入多头、蒸馏或新的解析控制层。

当前状态：CURRENT_DIRECTION；实际代码、两组运行状态、结果和是否被新指令取代均待本地校准。禁止写成“两个网络已训练成功”。

关键判据：相同物理条件和共同底层下，真实速度、转向、姿态与恢复表现符合所声明的端点偏好；不能只看 reward 更高或曲线更平滑。

来源：history_document + 与 S0 一致的当前意图；核验：原始实施 not_checked。

#### STTW-H006 — 奖励周期波动与全局成本裁剪的历史诊断

问题：奖励抖动是否等于学习崩溃，以及奖励是否能区分所希望的改善。

历史分析：V4 文档对用户事件日志报告了环境采样阶段同步和旧总成本 `min(C,100)` 长时间触顶的情况，指出全局裁剪可能掩盖速度/转向偏好及航向改善。[S1]

当前结论：这是已有的具体诊断线索，不能重新把全部问题泛称为“学习率过大”。但本次只读了分析文本，尚未重核事件文件；修复是否实施、是否改善控制，另需对应运行证明。

不允许的替代：平滑显示曲线、单纯增加训练步数或改用不同奖励尺度后直接比较 reward。

来源：history_document；核验：原始日志 not_checked。相关后继：STTW-H005。

#### STTW-H007 — 2026-10-08 补证：共享指令层与冻结底层500轮

- 问题：强底层是否能使上层形成端点取舍与回正；关联STTW-C1/C2。
- 尝试：V3共享上层从零训练500更新、32,768,000策略转移；下层update0200冻结；源码eb86825，E1。不是V4独立端点实验。
- 观察：main冲突[2.5,4.5)s，B0/α0/α1速度RMSE=.00937/.13613/.13750m/s、转角RMSE=.12914/.09146/.09149rad、全程峰roll=.3753/.4373/.4377rad。α0与α1全程速度差RMSE=.00401m/s、转角差=.000902rad；主/随机均未恢复。训练cap率约37.6%，末20轮约48%，0/40960已结束训练回合物理失败。
- 结论：支持上层改变行为及以速度换部分转角误差，不支持端点分离、共同姿态/恢复达标或完整路径恢复。触顶是诊断线索，未证明唯一因果。
- 证据：E1原始metrics/status及派生training_summary、保存audit；6/6轨迹审计通过为已存收据，本次未重新执行。详细图索引 `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/direct_command_frozen_lower500_20260929/INDEX.md`。
- 下一步：从已存B0回正和raw/governed误差定位底层迁移失配，再决定V4需解决的具体问题；不默认续训。若底层不能独立达到恢复要求，先不能用上层独立端点结果解释协调收益。

#### STTW-H008 — 有界历史补证与尚未建立的对应关系

- 问题：保留三头失败和续训反例，但避免误配运行。
- 尝试/观察：E6中的“discrete_alpha_three_control”实际是ECBC1、ECBC0.8、direct三种控制臂，各200轮；末25轮direct平均0.7691s且100%失败，ecbc08仅25/200更新保留。其名称中的three不等于三头网络，不能拿这份结果证明H002的网络结构失败。
- 续训工件：asymmetric_priority_comparison_20260918_continue250/rho34p2为250更新完成、32,768,000新增转移；rho10暂停于36更新，总流水线paused。该运行与H003“250再250 vs20”报告的一一对应仍 UNKNOWN；不混算成全部已完成。
- V3直接前序：`/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/docs/METHODS_AND_RESULTS.md`及`docs/evidence/direct_command_fresh500_20260929/README.md`报告fresh500采样456轮后硬KL回退，偏向保速度而损失转向；仅本次核验报告，未重新核算其轨迹。
- 结论：H002/H003原用户报告保留，原始身份未充分补齐；阶段同步在E6报告中已解决，不能把它套为当前所有失败原因。
- 下一步：有明确旧run/结构配置线索时再映射三头和续训；不为补录重训。source_type=repo_artifact/history_document，verification_scope=所列status/summary与报告，原始物理复现=not_checked。

#### STTW-H009 — SmoothV4：降低无益上层抖动并保留跟踪与方向恢复

- **最终回收（2026-10-08T23:20:58+08:00；repo_artifact/artifact_checked）**：execution_status=completed；evaluation_status=complete；adoption_status=not_adopted（不作为六场景合格控制器）。两端250/250有效PPO更新；各16,384,000策略转移＋131,072初始Critic转移；最终status=saved_and_stopped，无自动续训。best均update250，仅在完整评测60/250中选优、qualified=false。原10秒末速度/转角/航向共同保持：B0 2/6、alpha0 3/6、alpha1 5/6；新12条最终轨迹无摔倒，但fast_turn两端侧倾超0.30rad且10秒航向未恢复。延长16秒两端末保持通过，不消除10秒失败或此前越界。
- **收益/反例**：alpha0直行保护后offset二阶差分RMS35.0357→0.3531（−98.99%），实际轮角变化率RMS0.13268→0.00937rad/s；急转/反转转角RMSE较旧alpha0退化5.91%/13.35%，反转进入容差延迟0.250→0.705秒。alpha1急转速度RMSE0.03407较B0的0.08265改善，但比旧alpha1的0.01736退化。直行两端仍加约+0.031/+0.033m/s无益offset；急转alpha0比alpha1更保速度、转向更差，不支持预期偏好已学会。单seed77001开发面板，不是独立泛化证明。
- **训练与交付**：两端硬拒绝/非有限0；最大KL0.01113/0.01621，末20批航向RMSE0.248/0.244rad，训练奖励仍有周期波动，不宣称收敛。前序fresh及续训账本15485.4秒（含结转fresh尝试，不含更早取消warm-start），无额外墙钟截止；本次只做已保存NPZ后处理，无新仿真。60/250共24条件奖励重建误差<6.4e−8。TensorBoard6010为1–150、6011为151–250，HTTP核实两端最后250；上传原生事件各含完整1–250奖励。远端[精简证据](https://github.com/QaQaaa-zzz/STTW_CONTROL/tree/98032bb7f0c73063c413e27d9011e0fab5c8a0a1/docs/evidence/r196_smooth_v4_250_20261008)，commit=98032bb7f0c73063c413e27d9011e0fab5c8a0a1，14文件9,029,539字节，远端HEAD已核实。必要PNG/NPZ/JSON及原生事件，无重复PDF/CSV/ZIP/模型。完整本地入口`/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_extend250_20261008/publication/bundle/README.md`。下列running/PENDING/150合同为历史快照，已被本段覆盖。

- **最新覆盖决定（2026-10-08T21:21:30+08:00）**：每端总轮数250；150评价移至250；取消额外墙钟/计算预算截止，保留耗时记录。续训队列已运行等待前序完成，详见文末extension250登记与 `/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_extend250_20261008/INDEX.md`。下列150与秒数上限均是被覆盖的历史合同。

- **最新阶段回收（2026-10-08T20:08:28+08:00，repo_artifact/artifact_checked）**：两端均60/150且update0060已保存，worker1821330仍存活；当前evaluate60/fast_turn。已保存straight_hold、speed_changes、gentle_positive、gentle_negative、steer_reversal五场景，fast_turn正在完成16s配对评测；evaluation_status=partial，尚无完整面板结论。源码run循环确认：本次评价、奖励审计和出图正常结束后自动顺序训练α0到150、α1到150，再做150评价并停止；预算/错误/停止条件仍优先，不能保证一定达到150。没有人为暂停、重启或新增实验。

- **稳定身份 / 核验**：experiment_id=`STTW-EXP-20261008-R196-SmoothV4-fresh150`；project=STTW_CONTROL；owner=Codex本会话；recorded_at=2026-10-08T19:04:14+08:00；source_type=`repo_artifact`、verification=`artifact_checked`。本轮只读核实运行产物并更新台账，没有新增仿真或改变训练。
- **问题与假设**：旧R196上层直行有无益干预，α0转角修正存在显著高频变化，急转后的方向恢复不完整。假设新的分项奖励保护、20ms offset差分和同回合Actor时间正则能缓解问题；假设尚未获得新面板验证。
- **方法与不变项**：原始发布指令limited_command仍是评价目标；proposal是保护前请求，governed是保护后下发参考，actual是真实运动，黑实线不是网络输出。保留R196＋ECBC/ESO、200Hz底层/50Hz上层、原物理/动作范围/slew及受限残差。α0调整必要欠速价格，α1保留速度目标与过度少转约束；增强小航向恢复和近目标偏航阻尼、普通稳态修正代价；用20ms offset速率/二阶差分替换电机差成本，Actor正则仅同回合邻接观测；删除总成本100裁剪，分项上限合计800并同步失败吸收。
- **最新用户决策**（`user_requirement`）：附件warm-start60被用户覆盖为Actor/Critic/Adam/std全部scratch，各150批；初始std0.1、seed81，不加载旧策略。512环境×128策略步，每批65536策略转移，每端最多9,830,400策略转移；另2批value-only共131,072转移/端，不计为Actor更新。无运行统计归一化，观测顺序/固定尺度沿当前network配置。20轮测评取消，60/150六场景物理评价；不额外扩训。
- **版本、配置、证据**：独立分支`experiment/r196-smooth-v4`，代码`e1d5c702c09aa5090b95cc3ffa986a1689bf6808`（本地提交，未push），真实实现父6cdb3ce；d269f71仅旧证据。工作树 `/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4`。权威解析配置见run中`alpha0/frozen_config.json`、`alpha1/frozen_config.json`；身份/物理/准备bank见`manifest.json`，两端`initialization.json`均parent_checkpoint=null、previous_policy_loaded=false、std约0.1，Critic/Adam全新。复用准备物理bank不等于续训旧Actor。
- **运行身份与资源**：run=`/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_fresh150_20261008_run03`；host=`qy-MS-7E06`；PID=1821330，launch_epoch=1791457135.416266，实际cmdline已核对；共享GPU，不停止其他任务。启动命令见`launch.json`；TensorBoard [6010](http://localhost:6010/#scalars)，当前真实奖励HTTP核验凭据`tensorboard_verified.json`。运行入口[INDEX](/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_fresh150_20261008_run03/INDEX.md)，[状态](/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_fresh150_20261008_run03/status.json)，[预算](/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_fresh150_20261008_run03/budget.json)。
- **预算与停止**：训练4500s/端、编译600s、评价累计1200s、总10800s；先到保存停止，无自动追加。已结算计算626.89s（含旧fresh尝试结转，不含正在执行批次未结算时间），compile=130.52s，train0=496.33s。硬KL>.03回滚epoch/减Actor LR/重新采样，连续3批硬拒绝停止；非有限立即停止。旧作业和它们的成本/失败记录不删除。
- **实际进度**（2026-10-08T19:04:14+08:00）：execution_status=`running`；upper0=7/150、upper1=0/150；α0已完成2批Critic预热。最新完整训练日志batch=7，accepted_policy_updates=7；最近持久checkpoint=0（每10批保存，不能把已完成批次全称为已保存模型）。最近批次KL=0.003140，接受4/4 epochs，Actor/Critic裁剪前梯度=0.7476/1.8092；当前已读7批累计硬拒绝0、非有限0。这些只是训练运行证据。
- **本次可报告结果与边界**：旧NPZ离线审计直行quiet窗口α0保护后修正139次有效反转、变化率RMS0.6097rad/s、二阶差分RMS35.034rad/s²；真实轮角变化率RMS0.1327，对照B0 0.0092、α1 0.0081。来源`/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_20261008/offline/old_commands.json`；旧电机差成本贡献另见同目录legacy_cost.json。这是旧数据后处理，不能当新网络改善。最近训练片段速度/转角/航向RMSE=0.0716m/s/0.0442rad/0.1619rad，来自随机训练片段，不能替代同场景对照；片段结束回合数0，物理失败计数0，无有效分母时不把日志failure_fraction=0解释为成功率。
- **评价 / 采用状态**：evaluation_status=`not_started`，adoption_status=`pending`；六场景速度/转角/方向共同保持成功数、分母、改善幅度、新旧平滑比、bestmodel均`PENDING`，不是0。当前未出现evaluation目录或best_model.json，符合还没到60轮。共同末保持阈值速度≤.1m/s、转角≤.04rad、航向≤.05rad持续.5s，物理失败不可合格，工作界越.30rad另列；fast_turn保留10s结论再报告16s尾段。
- **best合同**：候选仅实际完成完整六场景的60/150检查点，优先失败/工作范围/共同保持，再本端主目标、次目标、平滑度；不按训练reward选模。bestmodel.pt和best_model.json保存模型及排序范围，后续使用选中模型；同协议实测已有则复用对应真实轨迹。此best不是150轮每轮都参评的全程最优，也不保证合格。
- **中断与更正留痕**：warm-start原试验用户取消于α0第4批；首fresh试验发现初始化std被旧函数覆盖为.2而停止；后一次在首Actor更新前按取消20面板修订重新启动。新run没有继承任何这些上层权重，fresh尝试已用时间由prior_attempt_budget.json累计结转。13项相关检查及独立奖励重建是工程证据，不是控制效果或实车证明。
- **为何保留 / 下一步**：继续当前明确预算内的运行，待固定面板决定是否保留候选；没有改善、失败或超预算均记录，不追加训练来追求通过。下一次会话先读status、60/150指标、reward_audit和best选择记录，再更新同一尝试；不因本Markdown存在就声称有后台台账监控。

### 2.4 当前优先行动

1. **STTW-NEXT-01：保留best250及本次冻结证据。** 两端均为候选，不升级为全场景合格控制器；训练与评价已停止。
2. **STTW-NEXT-02：后续如获新实验指令，针对alpha0响应/转向损失和普通场景无益offset提出可检验改动。** 不直接追加轮数或改变物理权限。
3. **STTW-NEXT-03：后续对比继续沿用固定六场景及原10秒门槛。** 延伸恢复单列，独立泛化仍未验证。

旧“先重新证明底层能力再提出实验”优先项已由用户本次明确R196/SmoothV4执行要求取代，历史诊断保留。未授权的规划器、搜索器、参数网络或连续共享策略仍不纳入当前运行。

## 3. DVGC-JIT

### 3.1 研究目标与当前流程

在已有跳跃策略上，通过受限扰动探索获得有价值的状态/轨迹，以接续训练、生成式桥接教案及学生学习等手段提高最终策略能力，并刻画经验覆盖/跳跃包线。[S0][S3][S4][S5]

```text
已有跳跃策略 + 合法起点/历史
                ↓
受限扰动探索 → 扰动后的真实状态
                ↓
基础策略接续评价 / 候选样本筛选
                ↓
需要时：生成式/扩散老师提供短段桥接方案
                ↓
经完整接续检验的教案 → 学生补训/蒸馏
                ↓
独立评价：新能力、旧能力保持、完整任务、泛化
                ↓
采用或拒绝候选；再决定是否进入下一轮
```

这是研究链条的组织方式，不代表各环节当前均实现或均有效。用户 2026-10-02 表述中已有探索网络和扩散恢复，并希望向跳跃策略蒸馏；当前精确实现仍须核实。[S4]

不得误解的时序：扰动后就由跳跃策略接续，补训起点也是扰动后的真实状态；不是先等到快摔倒再取“抢救状态”。“状态修复”是控制效果描述，不代表扩散模型一定直接输出状态；历史桥接方案生成短段动作，当前输入/输出以代码为准。[S3][S5]

### 3.2 模型身份、阶段和已知历史来源

必须区分：`baseline_actor`、`explorer`、`generator_or_teacher`、`teacher_tail_actor`、`student_initializer`、`retention_reference_actor`、`candidate_student`、`adopted_actor`。同一对象可以兼任某些角色，但必须显式声明，不能靠变量名猜。

2026-09-28 用户指定基础模型线索：[S3]

```text
/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928
```

文件夹名含 `9977856`，不改变所选检查点为 `transition_4988928`。若不存在后续明确变更，不能擅自换成文件夹中最新检查点。绑定新全任务环境、标签、归一化和评估终点必须核实，不能只改路径。

历史分支/证据线索：

```text
较早探索阶段：agent/two-phase-soft-tube
生成桥接代码：agent/generative-bridge-v1-1
生成桥接结果：agent/generative-bridge-results
相关报告：JIT/docs/generative_bridge/CORE_RESULTS.md
相关来源：JIT/docs/generative_bridge/source_audit.json
```

以上不是当前 HEAD 结论，不能自动 checkout 到这些分支或覆盖本地工作。

本次当前合同（E3/E7，artifact_checked）：

| 角色/字段 | 当前实现、执行程度与边界 |
|---|---|
| baseline_actor | 指定Phase U transition_4988928；零新增训练重绑定为 `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/audit/binding/bridge_baseline/checkpoints/transition_0`，名称0不表示随机初始化。initialization_receipt保留原Actor/normalizer身份 |
| 身份 | Actor `a06acbc81aead7120a291cf82359189bff36ccfecde132615ae0f264ee3126e8`；normalizer `db84aad7750991d6b720e9f065ed7e49f862893c7753f9839c593d7951b6ea86`，本次读绑定凭证，未重算参数树哈希；不将jump_ori环境别名自动当作本策略 |
| adopted_actor | v1.2无学生训练/采用完成凭证；当前保留baseline。v1.1两候选均拒绝，仍保留其源lineage_repair_0093；这两个阶段不互相覆盖。“全JIT最新统一采用策略” UNKNOWN，未审计所有旁线 |
| explorer | 本阶段 fixed_uniform_collection_only；historical_explorer_imported=false、optimizer_updated=false、stage_B_enabled=false（原始explorer_provenance）。旧学习探索器与Stage B准备代码不能称当前已更新 |
| 扰动后时序 | 完整起点第0/1/2控制步施四通道脉冲，0.02s控制；真实前缀结束的完整状态/历史用于接续与支持，不等待临倒。快照训练不再注入该外部脉冲，不能将任意状态修改当真实可达前缀 |
| G/老师输出 | H=16，76维条件，4维动作序列，U-Net/DDIM；执行短动作桥接后无reset交回source Actor继续完整任务，不直接生成物理状态。当前G为 `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0002/campaign/generator_pretrain/update_20000/manifest.json`，pretrain_result记录20,000更新 |
| tail/保持参照 | 当前v1.2 teacher_tail、retention_reference均为同一4988928绑定来源；generator reference与tail可在后续源码独立绑定，不代表Stage B已执行 |
| student_initializer | A/B从源Actor＋normalizer开始，Critic/optimizer新建；C预定从源Actor的BC预热候选开始，再PPO；三臂均未执行，不是exact_resume；不是连续已采用学生外循环 |
| 教案累计 | v1.2有累计有效教案bank/manifest与采样代码，但本次在teacher_0001 invalid处终止，未进入学生导出/消费阶段；不能写成多轮累计教案已训练。v1.1实际只有G累计、学生按轮从同源重新训练 |
| 最新执行 | recovery_0002/campaign phase=failed：`teacher replay invalid; never use empty-demo fallback`；计费459,504，G20,000；PPO 0/384,000、BC 0/2,000（阶段收据无学生阶段） |
| 完整评价/采用 | A0固定条件4次名义重复成功（非4个独立条件）；新三臂完整学生评价未开始。最新已定位的完整老师/学生采用评价仍是v1.1 pilot＋series round1，见H003本次补证；不将A0代替泛化 |

当前失败需保留：teachers/0001_result.json选中candidate6，verification.full_success=false且teacher_status=invalid；同目录0001_layout_repeat.json却记录candidate6的search/repeat标签均1，changed_candidate_ids=[0]。两份原始文件存在解释差距，原因 UNKNOWN，本轮未静默选一方消除错误、未重放或修代码。

### 3.3 研究尝试账本

#### JIT-H001 — 扰动探索拓宽候选轨迹

问题：如何相对随机扰动更有效获得可利用的扰动后状态和轨迹。

尝试：探索网络根据观测/历史输出受限扰动，在已有跳跃策略基础上探索；历史上下文中存在候选成功率和覆盖改善线索。[S5]

当前结论：这是已有主线，不是每次对话重新从零设计。具体探索器版本、采样预算和随机对照须补齐。

证据边界：候选成功率、候选轨迹集合和最终单策略能力不是同一指标；不能仅凭探索结果声称最终跳跃成功率或泛化提高。

来源：user_report / 历史摘要；原始结果核验：not_checked。

#### JIT-H002 — 扰动后接续与补训

问题：把拓宽得到的状态转化为策略可掌握的能力，而非只画更宽的 tube。

尝试：从扰动后的真实状态接续，并使用这些状态开展后续补训。[S5]

当前结论：保留这一真实训练起点，不再提出基于错误“临摔倒才开始训练”假设的解释。

待补证：快照来源、观测历史、是否重复施扰、旧能力保持和从完整自然起点的最终评价。快照恢复成功不能替代完整任务结果。

来源：user_report；具体运行和结果核验：not_checked。

#### JIT-H003 — 生成桥接老师与候选学生的历史试验

问题：老师找到的恢复方案能否真正转化为学生能力，同时避免旧能力回退。

历史观察：2026-09-28 文档报告两个独立候选学生，而不是一个学生累计两轮。[S3]

| 历史面板 | pilot：旧→候选成功数 | 追加第 1 轮：旧→候选成功数 |
|---|---|---|
| new_roots | 13→19 / 32 | 11→21 / 32 |
| core | 58→54 / 64 | 58→55 / 64 |
| protected | 57→61 / 64 | 57→55 / 64 |
| nominal | 1→1 | 1→1 |

pilot 的 core 丢失 9 个旧成功；protected 保留了旧成功中的 54/57，保持率 94.74%，不是用候选总成功 61/64 代替保持率。该表现未满足历史 protected 至少 98% 的采用条件。

同一历史报告中，老师对两批根找到 19/19、21/21 方案，学生对应掌握 7/19、10/21。不能汇总成“一名最终策略累计学会 17 个”，也不能写成 40 个独立训练种子。

当前结论：存在老师—学生转化和能力回退问题；这些数据不能证明最终单策略净提升及生成模块必要性。数值/布局敏感性仍需分辨，不能把每一次标签翻转都直接称为真实遗忘。

来源：history_document，S3 引用上述历史代码/结果 commits；本次只核对报告文本，原始结果 not_checked。当前模型是否已被后续版本替换：UNKNOWN。

#### JIT-H004 — 累计教案、学生谱系与主动探索接入的改进方案

问题：生成器累计历史，不代表学生也学到累计历史；每轮重启学生不等于连续迭代。

历史审计陈述：S3 指出当时学生只导出当前轮教案、各轮从相同源 Actor 重启，只有 G 累计语料，且 `explorer_optimizer_updated=False`。

改进规格：使用累计有效教案；区分长期基线、老师尾段、学生初始化和保持参照；正式外循环从最近已采用 Actor 继续，候选未采用则保留原来源；再单独接入主动探索。

当前状态：历史下一阶段方案，不是自动已经落地的事实。须查实现和 run manifest 才能升级状态。不能把修复谱系问题等同于整个方法已验证有效。

来源：history_document；原始源码与实施结果 not_checked。后继实验 ID：UNKNOWN。

#### JIT-H005 — 扩散模块必要性与向跳跃策略蒸馏

问题：扩散/生成模型为何适合提供恢复教案，是否有相对更简单方案的可测优势，学生部署时是否真正获益。

当前背景：用户 2026-10-02 明确要求分析这一路线是否只是增加模块，以及如何组织成小论文。[S4]

需要保存的研究判断：老师端有效不等于学生端有效；同预算更有效恢复、更好的教案覆盖和最终学生独立收益，应分别确认。最终评价若仍使用老师介入，必须标明是组合系统，不能声称单一跳跃策略获得了同样能力。

当前结论：OPEN；不预设扩散一定必要，也不未经证据就放弃。具体同预算比较方案属于后续实验设计，不由本管理文件自动启动。

来源：user_requirement / user_report；当前运行结果：UNKNOWN。

#### JIT-H006 — 2026-10-08 回收v1.2失败与累计学习边界

- 问题：从指定4988928源建立新标签/累计教案，能否改善学生吸收和旧能力保持；关联JIT-C2/C3。
- 尝试：A=PPO+keep，B=累计教案+PPO+keep，C=累计教案+BC预热+PPO+keep；仅C为预先指定主候选。每臂128k，原总物理预算4,421,600、24h、G22k/BC2k；恢复版限4,339,600（含A0复用）＋旧失败预留82,000，不重置截止。来源变更后不沿用0093标签。
- 观察：A0四次同条件名义通过，fresh support完成；G预训练20k后，teacher_0001重放invalid导致failed，459,504计费，尚无学生PPO或采用结果。旧report“等待GPU/G未训练”已被本次原始status纠正。错误文件与layout repeat间标签差距见3.2，原因UNKNOWN。
- 结论：有G训练和部分桥接工程证据；不支持累计教案提高最终学生、学习探索器联合迭代、学生真实回流G或最终单策略增强。invalid不等于无解，不允许empty-demo fallback绕过。
- 证据：E3/E7及 `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/ACTIVE_RUN.json`，指针实际为recovery_0002，不是campaign_0001；本次只读核验。
- 下一步：优先对照已存search/replay轨迹、candidate/lane、选中凭证和判定代码解释冲突。解释并固定有效性边界前，不重启A2或自动进入Stage B。

#### JIT-H003/H004 本次核验补记：保留能力与谱系

E4原始actor_acceptance确认两次adopted=false：pilot core保留49/58、丢9，protected54/57（94.74%）；round1 core50/58、丢8，protected50/57（87.72%）。采用门槛为core无新增失败、protected≥98%、净新增≥1。因此即使new_roots 13→19、11→21，两候选仍拒绝，不能只报新增成功。

E5报告老师19/19、21/21，学生独立7/19、10/21；G语料39→60，真实学生回流0。这些老师逐根计数本次只核对结构化报告，未重算每条轨迹。源均0093，学生分开初始化，G连续继承；不是一名学生累计学会17个。series状态为failed/KeyboardInterrupt、已完成1个追加轮；停止沿革见E7用户切换v1.2记录，不写成自然完成三轮。下一步仍是有效教案吸收与保持的独立证据，而非把G MSE下降当学生成功率。

### 3.4 当前优先行动

1. **JIT-NEXT-01：解释v1.2老师选中候选重放冲突。** 优先只读比对teacher_0001与layout repeat的candidate6、轨迹、判定路径；解决有效性问题前不把invalid记成无解，不进入学生或Stage B。
2. **JIT-NEXT-02：定位一处实际转化瓶颈。** 用已有数据区分老师有解但学生未学会、老师未解、旧能力回退或评价不稳定；给出一个有界诊断，并标注对小论文哪项主张有作用。

扩大网络、增加新时机判别器、重写物理、无上限老师搜索不自动列为下一步。

## 4. 活动实验与结果回收

历史检查快照（不代表当前无训练）：2026-10-08T09:10:13+08:00。本机/proc中按两个项目实际cwd及Python任务筛选，未发现相关训练/评价进程；只说明本轮筛选结果，不覆盖远端、容器、其他cwd或调度队列。旧status与收据优先；本轮没有停止任何任务。历史服务/远端队列是否可自动启动 UNKNOWN，下一次有资源操作前须再查。

| experiment_id | 项目 / 负责人 | 分支 / 代码快照 | run 目录 / 命令 | host / job 身份 | 状态 / 检查时间 | 结果及下一次检查入口 |
|---|---|---|---|---|---|---|
| direct_command_frozen_lower500_20260929 | STTW / 原运行负责人UNKNOWN，本轮仅归档 | eb86825；工作树HEAD见0.1 | `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/direct_command_frozen_lower500_20260929` | qy-MS-7E06；未见存活worker；历史launch.json可查 | completed / evaluation complete / adoption UNKNOWN；本次核验 | `cat /home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/direct_command_frozen_lower500_20260929/status.json`；review/metrics.json与INDEX.md；不自动续训 |
| generative_bridge_v1_2_recovery_0002 | JIT / 原运行负责人UNKNOWN，本轮仅归档 | code_recovery_0002，2554c75 | `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0002/campaign` | qy-MS-7E06；未见存活worker；started.json/notifications/launch_receipt.json可查 | failed / evaluation partial / no new adopted student；本次核验 | `cat /home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0002/campaign/status.json`；teachers/0001_result.json；先解释重放invalid |
| generative_bridge_v1_1 series_0001 | JIT / 历史 | 5941d6b | `/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_1_20260928/series_0001` | 原启动身份留在run | interrupted语义；原status=failed/KeyboardInterrupt；完成追加1轮 | actor_acceptance、status；用户切换来源后停止，不能重启旧队列 |
| STTW-EXP-20261008-R196-SmoothV4-fresh150→250 | STTW / Codex | experiment/r196-smooth-v4 / e1a6cc1 | `/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_extend250_20261008` | qy-MS-7E06 / worker2356405已结束 | completed / evaluation complete / not_adopted；2026-10-08T23:20:58+08:00 | 两端250/250，best250；见H009和[远端](https://github.com/QaQaaa-zzz/STTW_CONTROL/tree/98032bb7f0c73063c413e27d9011e0fab5c8a0a1/docs/evidence/r196_smooth_v4_250_20261008) |

每个真实运行出现后填写一行，详情可引用原 manifest。进程标识至少结合主机、启动时间或调度作业身份，不单凭 PID；禁止干扰另一项目任务。

结果回收顺序：核对运行身份和完成状态 → 检查完整产物 → 汇总基线/候选物理指标 → 检查评价有效性 → 更新对应尝试与采用状态 → 更新论文证据与下一步。结果失败或没有改善同样必须登记。

离线运行结束而 Codex 未活跃时，记录为下一次会话首要检查项；本文件不是后台监控程序。已存在的获授权收集流程可提供结果文件，但科学结论仍须基于完整证据整理。

## 5. 决策记录

决策是“选择怎么做”，不是实验成功证明。保持旧决定与后继关系。

| ID | 决策 | 来源 / 时间 | 影响与后继 |
|---|---|---|---|
| DEC-001 | 两项目并行、各自小论文；建立持续维护的唯一研究台账 | S0，2026-10-07 | 本次管理初始化；不授权额外训练 |
| DEC-002 | STTW 不继续把三头当默认主线，改为上层输出指令、底层负责跟踪 | S0，2026-10-07；更早起始日期待查 | 保留 STTW-H002 的失败记录；当前路线见 H005 |
| DEC-003 | STTW 共同冻结强底层，两个固定 α 独立上层；当前不做共享 Actor、多头或蒸馏 | S1；原决定准确时间待查 | 取代 S2 中共享上层安排；不是否定未来共享研究 |
| DEC-004 | JIT 扰动后直接接续/补训，不能错写成临摔倒抢救 | S5；原决定准确时间待查 | 约束状态来源、时序与论文解释 |
| DEC-005 | JIT 指定 transition_4988928 为新阶段来源；教案累计与模型角色分离 | S3，2026-09-28 | 后续是否有明确变更待查，不静默选最新模型 |
| DEC-006 | 把扩散的作用和学生收益作为待证明问题，而不是预设论文结论 | S4，2026-10-02 | 引导证据整理，不自动授权新模块或新实验 |

## 6. 论文主张—证据矩阵

以下是应积累的证据，而不是声称已具备全部结果；未被当前阶段授权的实验只作为缺口登记。

| 主张 ID | 希望支撑的结论 | 需要的证据 | 当前状态 / 不能越过的边界 |
|---|---|---|---|
| STTW-C1 | 上下层分工能在既定约束下产生明确端点取舍 | 同底层、同指令/场景下 α0/α1 的速度、转向、姿态与恢复对比 | E1共享上层500轮不支持：两端点几乎重合、恢复不达标；独立V4尚无已查执行证据 |
| STTW-C2 | 所提机制带来改善，而非仅来自更强底层或更大动作权限 | 匹配底层和权限的对照，以及针对当前机制的必要消融 | 证据缺口；不自动开展大面板 |
| STTW-C3 | 抗扰与可恢复范围扩大 | 固定状态/扰动定义、明确预算、完整恢复判据及失败边界 | 历史线索待补；经验采样不等于全局安全保证 |
| STTW-C4 | 具有平台验证价值 | 与仿真结论对应的实机协议、日志和限制 | 实机证据待定位；未核验不写已完成 |
| JIT-C1 | 探索网络提高可利用样本获取效率 | 同预算随机/学习探索，报告有效样本与覆盖，并计入评估成本 | 候选数据线索待核验；不直接推论学生能力 |
| JIT-C2 | 生成式/扩散桥接有独立价值 | 问题、输入输出和角色清楚；与适当简单方案比较恢复/教案收益及计算成本 | OPEN；不预设复杂模块必优 |
| JIT-C3 | 最终学生学得更强且保留旧能力 | 单个已采用模型；新旧能力、完整任务、未参与调参的评价与谱系 | E4两候选均拒绝，旧能力回退；E3新阶段未到学生训练，不能证明最终独立收益 |
| JIT-C4 | 包线/覆盖描述与真实部署能力一致 | 按模型版本区分候选轨迹并集和单策略经验能力范围 | 定义与证据待校准；不把 tube 直接称认证可达集 |

每幅拟用于论文的图须能定位：claim_id、run_id、原始指标、绘图入口、数据版本、样本数及筛选规则。尚无原始数据的示意图明确标注“示意”。

## 7. 历史补录与尚未解决的冲突

已覆盖：附件全部H001–H006/H001–H005历史种子完整保留；本轮补E1–E7及H007/H008/JIT-H006。当前STTW共享冻结底层500轮、JIT v1.2失败、v1.1采用负结果优先核验。

未覆盖：完整旧对话、S1/V4及S3原文、所有历史工作树/旁线、全部旧轨迹复算、远端/容器/调度作业；三头具体run与“250再250 vs20”严格配对仍UNKNOWN。已核验两主仓库和当前实现工作树版本/改动、所列运行，不代表全面历史审计。

检索入口可以包括仓库已有 `AGENTS.md`、`JIT/AGENTS.md`、`CURRENT_STATUS.md`、`PROJECT.md`、`CODEX_HANDOFF.md` 及其实际变体。找到后登记真实路径，不批量新建同名副本。

| 缺口 ID | 待解析事项 | 处理原则 |
|---|---|---|
| GAP-01 | 已确认不一致：已查实现为共享上层，V4独立端点未落地 | 保留V4意图；不把共享负结果归给未执行V4 |
| GAP-02 | 三头失败和续训对比对应哪些 run | 保留用户报告，原始证据待补；不重跑来伪装历史 |
| GAP-03 | 累计教案代码存在；v1.2在老师阶段失败，学生未开始，探索器未更新 | 代码存在不等于多轮外循环运行；Stage B仍关闭 |
| GAP-04 | 当前路线评价已定位；STTW采用凭证与全JIT跨旁线最新采用仍UNKNOWN | E1为last；E4均拒绝；E3未训练学生，不从最新文件推定采用 |
| GAP-05 | STTW“路径优先”在当前实验中究竟测了什么 | 保留路径目标，同时准确说明当前转角/航向/几何路径证据边界 |

## 8. 新尝试记录模板

```markdown
### <STTW/JIT>-EXP-<YYYYMMDD>-<短标签>

- 问题与关联论文主张：
- 假设：
- 相对哪个实验，主要改变什么：
- 日期 / 负责人 / 当前阶段：
- 代码 / 配置 / 数据 / 模型来源：
- 初始化与续训身份：
- 预算、停止规则、采用规则：
- run_id、运行目录、日志及结果入口：
- 执行状态 / 评价状态 / 采用状态：
- 观察：成功数/总数、关键物理指标、基线差异；无结果写 PENDING。
- 结论：支持、不支持、不确定或结果无效；说明适用范围。
- 失败原因：已验证原因与待检验解释分开。
- source_type / verification_scope / verification_level：
- 为什么保留或暂停、什么条件下才重试：
- 下一步：
```

## 9. 最近更新

### 2026-10-07 — 初始化快照，revision 0

建立两个项目的目标、当前路线种子、已知失败/演进、JIT 历史结果摘要、决策与论文证据矩阵。当前仅为管理文档初稿；尚未安装到用户工作区，未访问本地仓库、运行训练或核验最新结果。

下一次更新应优先完成“当前状态校准与既有结果回收”，不是继续扩写管理体系，也不是为了补齐表格启动新实验。

## 10. 本次初始化交付与接续

- 已完成：附件增量落地、规则/仓库定位、当前事实校准、有界历史保留与补证、运行结果回收。验收通过：6个实际代理入口均含两条真实绝对路径；8份旧文档仅增量加入口，原正文逐字保留；所有初稿历史ID保留；两份共享文档的字面绝对路径全部存在；4个被修改工作区git diff --check均通过。
- 新建：`/home/qy/STTW_CONTROL/research-hub/AGENTS.md`、本文件；`.write.lock`为串行协调锁，不是第二台账。
- 接入：两主仓库AGENTS、JIT子目录AGENTS，以及STTW当前指令层工作树AGENTS、JIT当前生成桥接工作树的根/JIT AGENTS。旧规则正文全部保留；可变工程状态文档顶部增加唯一来源标识，冻结结果与源码不改。
- 本轮只做文档和只读检查；未训练、未物理重跑、未停止任务、未新增后台监控、未commit/push；已有未提交文件保留。源码/权重不因本轮变更。
- 接续顺序最多三项：①新会话先读共享规则与本台账，核对第4节登记运行是否出现新工件；②STTW解释冻结底层回正失配并明确V4与共享实现差距，只有匹配底层和权限下分离/恢复证据才保留协调优势主张；③JIT先解释重放invalid，再判断有效教案能否被学生吸收且保留旧能力，未通过采用不替换基础模型。
- 初始化错误记录：一次通用JSON摘要脚本遇list而非dict，改为按文件schema读取；未改原始工件。猜测的direct_command_lower.py不存在，已定位真实frozen_lower_controller.py。不将工具检索错误当实验失败。

### 2026-10-08 — revision 1 更正摘要

共享上层500轮已经完成且不达标；V4是目标而非已执行独立端点。JIT最新恢复不是等待GPU，而是老师重放invalid失败；G已20k但学生PPO/BC未执行、探索器未更新。旧“当前”说明降为日期快照。补录时保留三头/续训user_report和老师—学生失败谱系，不编造完整旧对话。


### 2026-10-08 — revision 2 文档验收

按 verification-before-completion 核验本轮修改：当前STTW状态/配置/评价判据、底层两个SHA与JIT失败计费/探索器状态均再次与磁盘JSON匹配。未执行算法测试或研究重跑，因为没有算法修改；保存的历史测试与轨迹审计只作为已有证据引用。共享文档无未替换的安装路径占位符；第8节的尖括号是有意保留的新尝试模板。

实际创建/修改清单（临时取证脚本在/tmp，不属研究入口；锁文件不入Git）：

- `/home/qy/STTW_CONTROL/research-hub/AGENTS.md`
- `/home/qy/STTW_CONTROL/research-hub/PROJECT_STATE.md`
- `/home/qy/STTW_CONTROL/AGENTS.md`
- `/home/qy/DVGC/AGENTS.md`
- `/home/qy/DVGC/JIT/AGENTS.md`
- `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/AGENTS.md`
- `/home/qy/DVGC/runs/worktrees/generative-bridge/AGENTS.md`
- `/home/qy/DVGC/runs/worktrees/generative-bridge/JIT/AGENTS.md`
- `/home/qy/STTW_CONTROL/PROJECT.md`
- `/home/qy/STTW_CONTROL/docs/METHODS_AND_RESULTS.md`
- `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/PROJECT.md`
- `/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/docs/METHODS_AND_RESULTS.md`
- `/home/qy/DVGC/PROJECT.md`
- `/home/qy/DVGC/JIT/docs/CURRENT_STATUS.md`
- `/home/qy/DVGC/runs/worktrees/generative-bridge/PROJECT.md`
- `/home/qy/DVGC/runs/worktrees/generative-bridge/JIT/docs/CURRENT_STATUS.md`
- `/home/qy/STTW_CONTROL/research-hub/.gitignore`


### STTW-EXP-20261008-lower-tracking — 已登记，待执行

- user_requirement：单独确认冻结下层在变速/变转角几个场景的能力；不训练。
- code_revision：fb8f229，experiment/direct-command-policy-v3；已有文档改动保留，无控制源码变更。
- protocol：straight_hold/speed_changes/gentle_positive/gentle_negative/steer_reversal/fast_turn，共6场景×ECBC+ESO及冻结残差2方法；同prepared_bank[3]、seed77001、lower_alpha1、upper bypass。每条10s，共12回合最多24000底层ticks；1800s硬墙钟预算，一GPU禁预分配。
- model/config：沿用direct_command_frozen_lower_500.json中update0200身份、物理/ECBC/ESO/权限；无权重或优化器更新。manifest保存指令和SHA。
- evaluation：诊断用途；最终0.5s速度±0.1m/s、转角±0.04rad、航向±0.05rad分别及联合报告，工作侧倾±0.3rad单列。无外扰，不推论旧任务或泛化。
- run_directory：/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/lower_tracking_probe_20261008；command：timeout1800s /home/qy/mujoco_playground/.venv/bin/python evaluate.py；owner本会话，host qy-MS-7E06。
- execution planned / evaluation not_started / adoption not_applicable；原始数据、图、审计随后补齐。


### 2026-10-08 — STTW-EXP-20261008-lower-tracking 完成

- source_type=repo_artifact；verification_level=reproduced，仅上述新声明12条诊断轨迹；无训练、无控制源码/权重/物理更改，未修改JIT条目。
- execution completed / evaluation complete / adoption not_applicable。24000真实控制ticks；仿真含编译324.24s，小于1800s预算。原始数据、完整六场景图、分项CSV及12/12独立奖励/参考审计见`/home/qy/STTW_CONTROL/runs/worktrees/direct-command-policy-v3/runs/lower_tracking_probe_20261008/INDEX.md`。
- 12/12完整10秒、物理失败0/12；末0.5秒速度±0.1、转角±0.04、航向±0.05联合保持：ECBC+ESO5/6，冻结残差0/6。仅一确定准备状态诊断，不作为随机成功率。
- 直行2.3m/s、0转角反例：冻结残差末0.5s速度RMSE0.321m/s、转角0.173rad、未包裹航向2.925rad；基线约0.046m/s/近零转角与航向。普通场景出现持续偏转，无法称当前冻结底层为可靠通用指令跟踪器。
- 正结果：fast_turn [1,6)s速度RMSE0.1401→0.0344m/s，峰值侧倾0.3848→0.2884rad，工作区超限2.32→0s；代价转角RMSE0.1011→0.1120rad，末段未恢复。其余5场景两方法均无工作侧倾超限。
- 结论边界：现有旧Actor+因果参考适配器+V3物理组合的问题已重现；不能归因为旧权重本身，未验证原任务是否仍正常。上层alpha完全旁路，故本轮问题不由上层修改指令触发。
- 下一步建议（未执行、无自动训练）：用同一旧checkpoint原任务重放与当前适配器对照定位接口失配；在普通指令稳定跟踪证据建立前，不把更多上层PPO当作底层问题的解决办法。


### STTW-EXP-20261008-lower-random-retrain — 用户授权新底层训练

- user_requirement：重新训练底层残差，随机速度指令，尽可能跟踪速度和转角；1024环境、先200轮，奖励及其它超参数由代理决定，启动后交付可用TensorBoard。取代此前仅诊断的下一步建议。
- 独立工作树/分支：/home/qy/STTW_CONTROL/runs/worktrees/lower-command-tracking，experiment/lower-command-tracking；父fb8f229，未提交新实现已保存source_snapshot及manifest补丁；旧任务与权重不覆盖。
- 方法：ECBC+ESO+有界残差，200Hz；Actor210/Critic211，10帧历史、ELU256/128、2高斯潜变量tanh后前后轮残差，原±1.5/10权限。无alpha、无路径适配，无上层。当前实际速度为仿真辅助观测。
- fresh Actor/Critic/optimizer/std；复用已有完整ECBC准备状态bank，SHA/物理/控制器核验；未加载任何旧策略/优化器。
- 指令：v1.5..3m/s；70%±.12rad、20%±.30rad、10%直行，连续随机target/hold/slew，10s回合末2s直行；当前指令可见、未来指令不可见。
- 奖励：速度/转角指数精度项+无全局裁剪Huber误差项、工作侧倾/动作幅度与变化惩罚、物理失败罚；无航向/路径目标，不把底层转角跟踪等同全局方向恢复。精确系数见冻结config。
- PPO：1024×128×200=26,214,400控制步；4epochs/8minibatches；Actor lr1e-4，Critic3e-4，gamma.9975，GAE.95，clip.2，entropy.001，std.15限.05.. .5，softKL.01/hardKL.05回滚停止，seed830081。
- 预算：短测8×16×2后正式从零初始化；总墙钟9000s，最多200轮，无自动续训或扩评；数值/硬KL错误停止。一个GPU禁JAX预分配，不停止其它任务。
- run_directory：/home/qy/STTW_CONTROL/runs/worktrees/lower-command-tracking/runs/lower_random_commands_1024_200_20261008；learning/cli/train_lower_command.py；host qy-MS-7E06；pid/starttime见launch.json。
- execution planned / evaluation not_started / adoption pending；5个新行为测试及7个原PPO测试通过；启动含独立零残差物理与ESO状态一致性门槛，短测真实梯度门槛。控制效果尚无改善证据，不以训练reward代替。


### 2026-10-08 — lower-random-retrain启动核验

- execution running / evaluation not_started / adoption pending。正式6/200更新，786432/26,214,400控制步；精确实时状态以run/status.json为准，非终局结果。
- 进程身份：timeout PID157082；独立watchdog161889；TensorBoard157083端口6006。详情launch.json含启动epoch；其它项目6021服务保持。
- 工程8×16×2通过，零残差全物理/控制器/ESO/执行器状态一致；正式update0000优化器state为空，权重与独立fresh初始化一致，不继承smoke。
- HTTP http://localhost:6006/#scalars 的training/train/mean_step_reward已经加载实际正式标量。已核验Actor/Critic真实有限梯度。
- 可读入口：/home/qy/STTW_CONTROL/runs/worktrees/lower-command-tracking/runs/lower_random_commands_1024_200_20261008/INDEX.md；source_snapshot、manifest、tensorboard_verified.json留证。
- 当前仍在训练；下一步仅回收200轮或停止状态，随后按用户请求配对诊断跟踪；无自动续训/扩大评测。

- lower-random-retrain启动后源码已提交 `3e847f3a0671ff966dbbe5c85db44adabd743443`（experiment/lower-command-tracking），未自动push；运行source_snapshot保留启动源码，提交不修改在训算法。


### STTW-EXP-20261008-lower-random-review — 分析及有界配对评价

用户要求分析训练结果。已核验200/200，26,214,400步，3652.73s完成，训练中0/12288已结束回合物理失败；不能作为控制效果结论。训练reward best为update156（第157轮采样），最终update200。对两checkpoint固定确定动作，同旧lower_tracking_probe6场景/seed77001/完整bank[3]、slew.5/.3、10s评价，共12新增回合最多24000ticks、1800s墙钟，不训练；核对物理/控制器/初态/指令后复用原6条ECBC+ESO基线，并按新奖励重评分。指令窗口[1,6)、末段[9.5,10)与此前相同，不以航向当本次训练目标。工作树lower-command-tracking；入口原run/review_20261008/evaluate.py，指标/图/审计在同目录。execution planned / evaluation not_started / adoption pending。


### 2026-10-08 — STTW-EXP-20261008-lower-random-review 完成

- source_type=repo_artifact；verification_level=reproduced（仅新12条固定场景评价）；训练日志/权重为artifact_checked。execution completed / evaluation complete / adoption pending，未新增训练。
- 训练200/200、26,214,400步、3652.73秒，0/12288已结束回合物理失败；31轮soft KL截短，708epoch接受，无hard KL/nonfinite停止。训练best为update156（157轮采样），last200。
- 同前次6场景/初态/指令/权限，复用匹配基线6条并按新奖励重评分；新增两checkpoint共12条、24000步、153.44秒。原始指令、动作权限、独立奖励/航向积分审计通过，PNG/PDF、逐步与累计奖励分项、XY/测速/姿态/实际残差、NPZ/CSV全部覆盖。入口：/home/qy/STTW_CONTROL/runs/worktrees/lower-command-tracking/runs/lower_random_commands_1024_200_20261008/review_20261008/INDEX.md。
- 12/12完整10秒，无物理失败或>0.3rad工作侧倾；末0.5s速度±.1/转角±.04共同保持12/12，航向±.05保持0/12。后者未列入本轮奖励，不能据此说已学会方向恢复，也不能把末段保持替代全程跟踪。
- last200对ECBC+ESO：六场景速度RMSE改善19%–42%，四转向场景转角改善7%–49%；fast_turn速度.1401→.0817m/s、转角.1011→.0621rad、峰值侧倾.3848→.2794rad，工作超限2.32→0s。直行则引入约.0044rad偏置、末速约2.269对2.3请求，10s航向约-.212rad；仍有左右不对称欠转。
- last200转角六场景均优于best156，但后者部分速度/侧倾更好，不称全面支配，不改training best身份。所有环境回合时钟同步且无失败，第157轮采样跨前回合尾.16s+新回合首.48s，均直行；训练reward选best受阶段影响。
- 结论：新底层比旧迁移残差明显改善，尚非精确通用跟踪或方向恢复完成；没有证据归因为网络容量不足或无梯度。
- 下一步（未执行）：1)回合相位和选模公平性；2)零转角偏置/左右欠转诊断；3)改善后再决定上层是否启用。保持当前模型/物理，未自动续训、扩评或push。

### JIT-EXP-20261008-student-evaluator-recovery — 当前恢复核验

- source_type: repo_artifact；verification: artifact_checked；更新 2026-10-08T10:54:05+08:00。仅更新DVGC-JIT，本记录覆盖较早的“recovery_0002仍是当前运行”判断；原失败历史保留。
- 研究阶段：固定G/固定P0的教案吸收三臂诊断，探索E与Stage B关闭。用户已明确要求修复并继续；本轮收到新的报错。
- recovery_0003已完成老师搜索：25条verified教案、3条有限source-only翻转隔离、4条原Actor已成功；demo1679实际转移。G预训练20000继承，BC2000完成。
- 预热候选0/500/1000/2000在8个solver-dev均8/8，预声明同分最早规则选0；不能宣称预热带来提升或C已使用更新后Actor。
- 新故障：student_A在Brax evaluator初始化时，128个TRAIN逻辑slot误用于8个eval环境；训练status environment_transitions=0，learning_probe executed_loss_calls=0，无transition checkpoint。不是PPO发散或任务奖励问题。
- 修复：独立eval环境包装器与student_eval随机键空间/8个slot；TRAIN128个slot/随机键/奖励/物理保持。CPU179项通过(30.14s)，含128/8真实JAX reset及随机键不交叉；真实旧工件恢复预检到student_A前，0新增物理/监督更新。
- experiment_id: generative_bridge_v1_2_recovery_0004；owner: 本线程JIT代理；code_revision: 6f1335b；不可变快照code_recovery_0004。
- resolved_config、身份、命令与预算：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/campaign/production.json、recovery_audit.json；源transition_4988928；weights_only P0、新critic/optimizer，不是optimizer exact_resume。恢复复用fresh_source_phase与warmup凭证，不重做G、采集、老师和BC。
- 预算：累计计费/预留811224(保留失败student_A的136000预留)，全部尝试总上限4421600；沿用recovery_0003的24小时截止，不延长；PPO各128000、G仅剩incremental2000，BC不再更新。
- execution_status: completed via report recovery_0005；stage: A2；evaluation_status: completed development；adoption_status: rejected A/B/C。新学生是否产生真实更新以当前status/metrics为准，CPU通过不证明GPU通过。
- 运行目录：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/campaign；入口：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/INDEX.md；通知/TensorBoard收据见该目录。final TEST未打开。
- 当前下一步：核验学生A真实loss、步数与TB；若出现新工程错误按真实边界停止；保留源冲突unknown，不通过伪造标签促成采用。原JIT-NEXT-01的candidate6误归因已更正为source-only lane0翻转，证据见recovery_0003/reused_preparation及teacher记录。

- 本次GPU启动验证：student_A formal_report/status均completed，128000/128000训练步，TRAIN panel728交互，transition_128000/identity.json存在；首损失KL gate passed。当前A_dev_038运行，不宣称独立开发评估已完成或Actor采用。TensorBoard http://127.0.0.1:6022 HTTP200，实际episode/sum_reward末点step128000、462.987060546875已从服务API取回；loss在student_A，奖励在episodes/student_A_episodes。完整证据：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/campaign/monitoring/http_verified.json。

- 2026-10-08 报告恢复核验：recovery_0004已完成A/B/C各128000步、全部声明开发评估和G incremental2000，最终汇总错误引用本地warmup/selection.json；实际预热结果继承自recovery_0003。原失败status保留。
- 修复提交8e8b619：按warmup凭证checkpoint位置解析selection，校验checkpoint/receipt及所选update；分离纯报告写入，新增finalize_bridge_campaign.py，禁止重复训练/物理/G更新。CPU回归181项通过，新增保护测试后报告专项4项通过；真实历史工件报告恢复成功，所有记录输入哈希未变。本次无新GPU物理测试。
- 当前入口：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0005/INDEX.md；reports/status.json=completed；根ACTIVE_RUN指向该结果。实际GPU训练384000步，累计物理计费/预留2290226/4421600，监督更新24000；本次报告恢复新增交互/更新均0。TensorBoard6022的A/B/C真实episode/sum_reward均已加载。
- 开发结果：A/B/C core lost=20/30/9（旧正236），protected lost=16/27/21（旧正234），new gained/lost=21/0、25/0、22/1，每臂new unknown4。三臂均未采用，C另有four-combination冲突/unknown隔离；不能宣称能力整体提升。保留P0，G增量已完成并选update0500；反馈history962、teacher_new25、actor_new0。最终TEST未打开，探索E/Stage B关闭。
- 本条取代本块较早的运行中/待检查学生A状态。下一步：分析固定G下三臂的新根收益与旧能力损失；任何新增轮次另明确预算，不因汇总报错重复已完成训练。代码已本地提交，本次未push。

### 2026-10-08 — STTW训练场景与指令语义核对

- source_type=repo_artifact；verification_level=artifact_checked；只读核对冻结config和command_rows/step实现，无训练/物理执行或路线更改。
- 最新lower_random_commands_1024_200_20261008每环境/回合重新采样速度1.5–3.0m/s与转角目标（70%±0.12rad、20%±0.30rad、10%零）；随机保持0.8–1.8s并限变化率，10s回合8s起转角目标归零、速度为该回合采样值。初段保留准备状态速度/零转角，不能说从首步到末步都完全随机。网络当前200Hz输出两路残差，参考指令与电机动作不同。证据：runs/worktrees/lower-command-tracking/runs/lower_random_commands_1024_200_20261008/config.json及对应learning/src/sttw_control/lower_command.py。
- 用户打开的旧bend_training_20260913/constrained_8192/frozen/task.json则是固定弯道几何、速度请求2.1m/s，随机外扰；沿路径产生的转向参考仍随进度变化。不能把固定路径、固定电机动作、随机任务指令混为一谈。
- 本次只回答训练分布问题，不改变已登记实验状态、采用决定或下一步。


### STTW-EXP-20261008-tracking-candidate-review — 已授权几何候选复查

用户要求读取METHODS_AND_RESULTS及本对话，找到跟踪最好的ECBC+残差并实际仿真。历史任务不等价，先筛选几何候选soft_budget best0244（路径略优）与precision best0196（超速峰值较小），不按不同reward排序或称全局最优。先执行soft_budget五场景alpha0/.5/1、seed49001，最多15残差+4新基线（单弯匹配基线复用），<=37400控制步、每回合<=10s，不训练、不改物理。检查point与模型/normalizer身份已通过当前加载器核验；代码主分支4cafb53，保留现有用户文档改动。原path_rsl R200仅特定几何扰动任务、最新lower随机指令仅转角任务，不与五场景几何回报混排。计划与输出：runs/tracking_candidate_review_20261008/plan.json、soft_budget/status.json；existing learning/cli/five_scene_review.py。owner本会话，execution planned / evaluation not_started / adoption pending；执行错误停止；新结果按同窗速度/路径/姿态与基线报告，非holdout。


STTW-EXP-20261008-tracking-candidate-review补充冻结：为避免仅按单弯挑选soft_budget，precision best0196也执行同五场景/seed49001/三alpha；两模型均身份匹配。合计30残差+8新基线<=74800ticks，单弯各自基线复用；两奖励的恢复状态逻辑不同，此次沿用原工具各自重建物理/计分链而不静默跨配置借用基线，其余场景基线重复一致性事后核对。CPU两个有界评估worker，无新增训练或参数更新。execution running / evaluation partial / adoption pending。


STTW-EXP-20261008-tracking-candidate-review用户追加两张图：已视觉核验图1=path_projection/checkpoint_comparison/force_right_trajectory.png，对应R75/R150；图2=timed_random_reward_only/inrange_best192的reversal__nominal XY图。追加有界原图原任务复测：R75/R150各四场景force_right×3alpha，R192四个inrange nominal×3alpha，共36残差回合<=72000ticks，10s/回合；复用各自匹配基线，无新训练。旧XML从b623a93提取并核验原source_xml_sha256=a3dcc59e...；保持原meshes，加载器完整身份断言，不忽略当前XML差异。输出legacy/R75、R150、R192，replay_legacy.py；只是原任务复现，不跨旧/新物理或不同命令幅值宣称总冠军。统一指令跟踪语义（速度+路径或速度+转角）已向用户提出澄清，同时执行不依赖该选择的原图证据核验。


### 2026-10-08 — tracking-candidate-review双目标确认

用户明确冻结底层主指标为速度+前轮转角指令跟踪，并追加要求也做路径贴合比较。两任务分别评价，不能用XY贴线排名替代底层跟踪排名。原图R75/R150/R192的36条物理回放已经生成，整理图时发现顶层priority_alphas缺失及场景声明未缩小到实际四场景；已按实际轨迹修复本次派生声明，不改源记录。precision缓存拒绝原因是旧tracking声明缺少新dataclass默认字段；已断言规范化完整配置相同后仅补齐本次缓存声明。soft_budget五场景15条件完成；precision与旧模型图尚待补齐，不宣称完整对比完成。下一步按共同初态/物理/指令测试冻结策略，分别登记原任务与迁移适配边界，无新增训练。


STTW tracking-candidate-review共同指令测试冻结：R75、R150、R192、precision196、soft244，每个六场景、alpha=1，共30新增回合<=60000控制步，10秒/回合，每模型1800秒墙钟上限，当前真实物理与执行器、prepared_bank[3]/seed77001、slew .5m/s²/.3rad/s、200Hz，沿用lower_tracking_probe_20261008协议。核对相同model/controller/actuator/bank哈希后复用既有ECBC+ESO、旧R200及新直接底层best156/last200配对轨迹；任何身份不符停止。旧模型通过本次run目录command_replay.py的显式因果适配输入路径/时间误差，权重不改、不泄漏未来指令；R75/R150与既有FrozenLowerController输入/输出/状态一致性门槛通过，五模型字段/身份/有限状态检查通过。R192时间路径与新几何模型保留各自输入语义；这测试完整适配系统，不声称原生速度/转角策略比较。主指标速度/转角RMSE、失败、末0.5秒保持；同指令积分XY贴合为独立描述指标，非新增路径闭环或训练成功标准。无新增训练。原任务图已全部补齐：36旧回放+30新几何回放；precision/soft五场景配对基线物理逐点相等，均0/15失败、12/15末段保持，tight_turn均不达保持。下一步完成共同指令测试并汇总模型台账。


### 2026-10-08 — STTW tracking-candidate-review双目标评估完成

- source_type=repo_artifact；verification_level=reproduced（本次66原生路径残差回合+30共同指令迁移回合）；复用24条共同指令轨迹为artifact_checked。execution completed / evaluation complete / adoption pending。无新增训练或当前模型替换。
- 入口：runs/tracking_candidate_review_20261008/INDEX.md、model_inventory.md、commands/INDEX.md、verification.json；docs/METHODS_AND_RESULTS.md已补模型/训练/结果专题。两图身份R75/R150同一个path_projection训练，ELU256→128/280维/10帧、固定四路径抽选+随机持续外扰；R192为timed_random_reward_only/0192、ELU128×3/310维/10帧、随机速度+yaw-rate/时刻，图为inrange小幅场景。不是原生前轮角度随机训练。
- 原生路径原图回放：R75失败3/12、末保持0/12；R150失败0/12、保持3/12；R192失败0/12、保持3/12（不同任务，不横排成功率）。新共同五几何场景：Precision196/Soft244各15，0失败、保持12/15、超时9/15；Precision路径RMSE在4/5场景较小，Soft单弯略好但侧倾峰值更高。66/66有独立逐步奖励重建，最大差2.16e-7以内，分项/速度估计/误差/跨alpha图与PNG/PDF齐全。
- 共同指令：六场景、seed77001、bank[3]、相同当前model/controller/actuator/权限/slew/原始指令，α1旧模型因果适配，30新+24身份匹配复用，共九方法54条均完整10s、0物理失败。原旧XML与当前不同，迁移明示；不是控制了训练分布/物理/网络的单因素实验。fast_turn对R192属于训练范围外推。
- 六场景[1,6)s池化速度/转角RMSE：ECBC .07469m/s/.05759rad；R75 .15705/.10201；R150 .31352/.13123；旧R200 .06221/.07640；R192 .05300/.03641；Precision196 .04387/.03599；Soft244 .03775/.03227；新直接156 .04571/.05106；新直接200 .04802/.03465。完整逐场景/阶段数据见commands/metrics.json；这里只是描述性池化，不以跨奖励回报选冠军。
- Soft244平均指令误差最小，但fast_turn末速度/转角保持失败（5/6），峰值roll .43518、累计>.3rad 2.595s；Precision/R192保持6/6但峰值roll .41314/.41975。新200保持6/6、峰值roll .27940，无>.3rad，但直行偏置和路径漂移不能隐去：共同指令积分路径全程几何横向RMSE2.2168m，差于ECBC1.2022m；Precision .55145m为本次最小。末转角容差通过不等于方向或路径已保持。无安全保证/实车/独立holdout结论。
- 结论：没有全面支配候选；Precision196保留为路径候选，new lower200保留为原生速度/转角候选，Soft不能只因较低平均误差直接封版。用户要冻结底层，但本次只提出候选建议，不自动修改已采用配置。
- 下一步仅建议（未执行）：1)定位并修正零转角偏置；2)在匹配急转条件检查姿态与欠转权衡；3)小范围冻结复验通过后再决定上层共同底层。停止本轮评价，不自动追加训练或扩大面板。


STTW tracking-candidate-review交付：专题方法结果追加已提交15b8761，仅本轮追加段落入Git，保留用户原有AGENTS/PROJECT/台账入口改动；本轮未push。运行脚本/图/NPZ/CSV与身份检查均保留runs/tracking_candidate_review_20261008，原冻结权重和物理源文件未改。共享台账已同步完成状态，不再有本轮活动仿真。


### 2026-10-08 — 用户保留R196与R244为后续候选

- source_type=user_requirement + repo_artifact；verification_level=artifact_checked。用户明确“保留R196和R244”，后续候选集合固定为Precision update0196与Soft-budget update0244；覆盖上一轮建议保留new lower200的候选优先级，不删除历史模型、不声称已经部署或恢复合格。
- 原checkpoint/Actor哈希与identity、best指向重新核验；候选登记runs/tracking_candidate_review_20261008/retained_candidates.json，原权重未改、未复制、未新增训练。
- 两组独立fresh（resume_checkpoint=None），均1024×128、ELU128×3、30字段×10帧+mask=310，200Hz，完整ECBC+ESO，残差1.5/10rad/s；alpha0/.5/1每回合等概率固定；3.5s零残差准备后10s任务，真实预推进分散训练相位。Precision共200轮/26,214,400步，197轮采样选196；Soft共250轮/32,768,000步，245轮采样选244；不是从R196续训到R244。
- 同一结构化动态训练混合：40%普通任务（速度1.7–3.0、yaw0/±.35、1s开始4s回正），40%左右短急转（1s速度2.5/yaw±1.8，1.95s回正），20%缓弯侧力（yaw±.35从1至6s、速度2.3、3–3.5s方向相关±2N）；有限任务抽样，不是扰动力/时长/全部切换时间连续随机化。速度/yaw经slew并积分固定世界参考，路径误差与速度构成学习目标，非原生前轮角度目标。
- 奖励主差别：Precision欠速权重.12→8/尺度.10→.05、路径权重4→.1，超速8/.05共同，总代价cap100；Soft欠速8alpha、路径8(1-alpha)、超速8/.05共同，保留越带/姿态/动作成本，以100C/(100+C)平滑总代价，1s宽限后收紧回归带及超时持续成本。两者逐步普通奖励均dt*.1*(1-有效代价)，首次超期额外-5、失败整步-200。
- 本次仅身份保留及训练说明，无新试验；此前急转姿态和保持反例继续有效。下一步由这两个冻结候选继续对照，不将训练reward best称全场景能力第一。


### 2026-10-08 — R196/R244固定底层别名与调用文档

- 用户要求说明当前α并在全局可定位Markdown记录，供STTW_CONTROL底层调用。新稳定文档docs/FROZEN_LOWER_CONTROLLERS.md与机器注册表learning/configs/frozen_lower_registry.json；根AGENTS/PROJECT及共享AGENTS均有绝对路径检索入口。
- 别名STTW_R196_ALPHA1、STTW_R244_ALPHA1固定原checkpoint和lower_alpha=1；权重/identity SHA钉住。原训练每回合0/.5/1，原生路径复测三α，共同指令对照固定1，不能混淆。上层alpha不隐式传入；动态α历史不得回填改写。
- 文档包括310维原始观测顺序/alpha索引18、10帧mask/归一化、实际可运行Actor加载代码、残差单位和权限、闭环状态与reset、符号/速度估计/定位依赖。现有通用FrozenLowerController仅支持旧280维；310维Transfer是已测试诊断实现，2001点缓存限10s，尚非无限时长生产控制器。没有自动换装当前上层或启动训练。
- 验证：直接提取文档Python代码，两个别名分别加载真实观测4帧，8次输出与已保存下一步动作完全一致（最大差0）；错误α被拒绝。记录runs/tracking_candidate_review_20261008/registry_verification.json。验证覆盖Actor加载/身份/调用示例，不等于新闭环部署或实车验证。


R196/R244接口文档交付提交1aa2e63；仅新增接口段落/注册表/文档入Git，原用户未提交内容保留；未push。共享检索入口与状态同步完成。


### STTW-EXP-20261008-frozen-endpoints250 — 用户确认四组启动

- user_requirement：两个冻结底层STTW_R196_ALPHA1（用户TTW拼写按唯一别名解析）/STTW_R244_ALPHA1，各自upper_alpha0/1独立网络，每组250；用户已明确确认2×2共四组。lower_alpha固定1，绝不传入upper_alpha。
- 方法：原上层V3参考修正/奖励/原始指令评价，345Actor/346Critic/2高斯latent，128/128/64 ELU，50Hz；每组独立fresh Actor/Critic/optimizer，alpha作为常量context且模型不共享。下层注册表SHA固定310输入/128×3ELU/tanh，200Hz，源归一化/字段/时钟/真实世界yaw与body yaw分开；权重不训练。
- 已验证诊断Transfer迁入registered_lower_controller.py；与原实现逐步obs/action/state对照一致。容量2001→3201仅支持原16s上层有限回合，无无限时长宣称。物理/ECBC参数/最终权限不改。
- 组合遵守冻结下层标准：ECBC在governed参考上输出，再加±1.5/10残差；ESO每tick提交一次，raw独立积分与reward不变。区别于旧500轮raw中心共享总修正限幅，不能把新旧结果称只换权重的单因素实验。
- 代码分支experiment/frozen-lower-upper-endpoints，父3e847f3，未提交启动代码保存在run/source_snapshot；计划见独立PROJECT，24项接口/PPO/原行为测试通过。短物理与8×16×2工程门槛在正式训练前执行，每组正式从零重建。
- 配置：512×128×250=16,384,000政策转移/组，65,536,000底层ticks/组；四组合计65,536,000政策转移/262,144,000底层ticks。原PPO gamma.997/GAE.97、4epochs/4minibatches、Actor1e-4/Critic3e-4、softKL.01/hard.03保持。seed73一致，命令66001一致，失败/数值/预算停止，不重试扩训。
- 资源预算：串行4worker，每个总16800s（compile900/engineering600/training14400）；每底层两组结束后配对评测1200s；总上限72000s。评测主场景+随机各B0/alpha0/alpha1=6回合/底层、12物理回合总计，分别按alpha重评同基线奖励；不扩大面板。
- run_directory：/home/qy/STTW_CONTROL/runs/worktrees/frozen-lower-upper-endpoints/runs/frozen_R196_R244_upper_endpoints_250_20261008；入口learning/cli/train_upper_endpoints.py。host qy-MS-7E06，pid/time在launch.json；queue_status.json+各组status.json提供进度。TensorBoard新6007，不停止旧6006或其它项目服务。
- execution running / evaluation not_started / adoption pending。用户“看看效果”由队列末配对评价回答，启动不称效果达标。


STTW-EXP-20261008-frozen-endpoints250 启动核实（2026-10-08T13:13:01.292682+08:00，repo_artifact/reproduced）：R196_upper0正式3/250，196608策略转移，state=running；其余三组排队。每个worker重做接口/短测/正式fresh初始化。第一组2×4s接口、非零上层/ESO单提交探针、8×16×2工程检查已通过；初始正式优化器空，参数等于seed初值且不同于smoke训练后权重，512环境upper0/lower1。首正式轮Actor梯度.6292/Critic2.9184，KL.003383，16/16 minibatches接受，无非有限/硬KL停止；只证工程运行，不证控制改善。

TensorBoard http://localhost:6007/#scalars 已HTTP读取R196_upper0/tensorboard/pilot的train/mean_step_reward更新1；凭据tensorboard_verified.json。队列PID799733、首worker799737、TB799734、watchdog799735，6006与其它任务未停止。代码提交2a7fcdd，未push；run/source_commit.json登记，报告索引链接修订不改变运行算法。唯一实时入口：/home/qy/STTW_CONTROL/runs/worktrees/frozen-lower-upper-endpoints/runs/frozen_R196_R244_upper_endpoints_250_20261008/INDEX.md。只读进度：cat /home/qy/STTW_CONTROL/runs/worktrees/frozen-lower-upper-endpoints/runs/frozen_R196_R244_upper_endpoints_250_20261008/queue_status.json 以及R196_upper0/status.json。

下一步（自动有界队列已配置）：1. 完成R196 upper0/1各250后主/随机六回合评价；2. 完成R244同预算同协议；3. 用同alpha配对奖励和速度/转角/侧倾/航向/实际残差报告收益及失败，停止不扩训。未达到预算或触发错误时保留实际完成checkpoint与状态，不声明250完成。


### JIT-EXP-20261008-experimental-closed-loop — 允许遗忘的有限闭环

- source_type: user_instruction + repo_artifact；owner: 本线程JIT代理；2026-10-08用户明确要求容许遗忘、以更强扰动/更多状态泛化评价过程与最终策略，并构成闭环继续。该指令覆盖Stage B旧入口中“零核心遗忘/必须正式采用”作为实验续接门槛的限制；旧A2拒绝记录保持不变。
- 方案：起点为recovery_0004的C Actor+normalizer、recovery_0005所指已保存G update0500；E新建对称mean0/std.6。每轮冻结当前Actor/E/G，采128条真实TRAIN(80% learned E/20% uniform)，原Actor续接标注、最多32根老师搜索，累计教案+全pending学生PPO128000，真实学生TRAIN回放，实验采用检查，E仅当前on-policy有效样本PPO，G增量2000，原子发布下一轮三模型。E是实际梯度更新，不只是重新采样。
- 实验采用：完整有限checkpoint+4次nominal全任务成功；遗忘统计不阻断实验续接。formal_adopted=false单独记录，未通过nominal保留前一实验Actor。老师与学生成功分开，source-repeat冲突unknown不获得伪转换奖励。教师H16、Actor/G结构、奖励、物理、动作映射、reset不变；保持项仍锚定P0，系数.2。
- 新预算：2轮/学生256000步，G<=4000次，E每轮最多4epoch(仅本轮样本)，新物理<=3300000，24h，不自动扩轮/重试。旧累计2290226另计；不覆盖旧实验，不停止其他项目GPU任务，沿用空闲GPU门控。
- 评价：P0、起始C、每轮学生固定相同完整初态、随机键和9条件（动作扰动幅度.25/.4/.6×onset0/10/20；每条件32，共288回合/模型）。逐条件记录成功/失败/未知、gained/lost；是固定开发压力集，不是最终TEST。没有等总预算对照，不据此宣称G或E因果优势。
- 代码：/home/qy/DVGC/runs/worktrees/generative-bridge；新增closed_loop.py/CLI，复用现有teacher/student/E admission/G更新；工程计划JIT/docs/generative_bridge/CLOSED_LOOP_PLAN.md。CPU190项回归已过，真实C身份/1679条累计教案迁移/62个原pending资格校验通过；复核补了预算超额和启动失败状态保护，最终回归待完成。本次GPU物理闭环尚未执行。
- 当前状态：queued_gpu；最终CPU193项通过(30.15s)，复核修正后专项7项通过；独立复核已完成。代码提交4ef24ab，本次未push。

- 启动实核 2026-10-08T15:25:50+08:00：experiment_id=generative_bridge_closed_loop_20261008/series_0001；监督PID1170961，code snapshot=/home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/code_4ef24ab；run目录=/home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/series_0001；入口INDEX.md，配置plan.json、预算cost_ledger.json、轮次current_source.json在首轮完成后发布。
- 第1轮baseline_stress_00执行状态=waiting，GPU有其他计算进程，门控尚未放行；原项目进程未改动/终止。训练0/256000、G新增0/4000、E未更新；首阶段12800是预留，不是测量交互。GPU物理闭环验证/新研究性能均尚无结果。
- 可视性：http://127.0.0.1:6023 HTTP200；独立新日志尚无reward，不冒充已加载。通知PID1171065心跳正常且无投递错误；真实reward观察器PID1174888将核验episode/sum_reward HTTP数据并写monitoring/http_verified.json。原v1.2顶层ACTIVE_RUN已指向新series，历史局部状态不改。
- 只读进度：cat /home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/series_0001/status.json；cat /home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/series_0001/round_0001/execution/baseline_stress_00/status.json。下一步：资源释放后自动执行首阶段并观察错误通知；首轮完成核验E optimizer_updates与三模型身份确实进入下一轮；两轮结束按9条件配对比较泛化收益和遗忘，不自动追加轮次。


### JIT-EXP-20261008-neighborhood-closed-loop — 邻域输入已启用

- source_type: user_instruction + repo_artifact；verification: artifact_checked；audit_time: 2026-10-08T15:56:48+08:00。用户明确要求启用邻域。supersedes JIT-EXP-20261008-experimental-closed-loop的series_0001队列（其余两轮设计/奖励/预算不变）。
- 旧队列停止前已暂停并核对监督PID身份、GPU门控waiting、stages=[]且无训练子进程；仅终止旧JIT监督与其监视服务，其他项目未动。旧run保留queue_cancellation.json，status=cancelled，actual_new_physics=0；12800旧计费项为未执行预留，不计成新消耗。
- 新版本663e4c1，本地提交未push；不可变快照/home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/code_663e4c1。新增TRAIN历史适配器，复用邻域编码器；旧source-only邻域模式不变。scope=train_history_v1随E checkpoint保存；旧106维E不得冒充兼容权重。Actor C和已保存G保持，E fresh symmetric mean0/std.6。
- E观测386维=106+16×17+8，既有邻域编码器压到64，策略/价值头输入178。12维相对物理坐标、当前Actor成功/失败、历史Actor成功、已验证老师成功、valid；近/远两尺度统计。历史Actor/老师成功不改写为当前Actor能力。不是完整或认证可达域。
- 初始1116条证据、1024个TRAIN到达context：原1024条TRAIN、C的32条TRAIN评估、25条verified老师、35条source复查。每个实际评估策略身份保留；同策略0/1冲突归unknown，包括teacher侧source重复冲突。邻域跨Actor保留到达信息，当前能力标志仅使用相同Actor实际评估。
- 地图每轮采样前冻结，config/map SHA绑定collection behavior receipt及PPO更新；用已记录386维obs学习，随机20%分支和unknown不进E PPO。本轮source/student/teacher结果只进入下一轮，开发与最终TEST不入图。奖励权重/原Actor/G结构/物理/reset不变。
- 验证：233 CPU项通过30.40s，含真实RSL386维PPO更新、随机/unknown排除、map篡改拒绝、checkpoint作用域校验、source冲突；真实历史数据构图核验及独立代码复核通过。新GPU物理闭环/泛化效果尚未验证。
- experiment_id: generative_bridge_closed_loop_20261008/series_0002_neighborhood；output=/home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/series_0002_neighborhood；入口INDEX.md、plan.json、initial_neighborhood_history.json、launch.json；两轮学生256000/G4000/物理3300000/24h，不追加原两轮之外的轮数。
- 当前监督PID1258478；第一阶段baseline_stress_00=waiting，GPU门控检测到其他计算进程。实际学生0/256000、G新增0/4000、E未更新，旧Actor/G复用已核验。通知PID1258582心跳正常无投递错误；TensorBoard http://127.0.0.1:6023 HTTP200、tags={}，尚无真实新reward，异步核验器在产生后记录monitoring/http_verified.json。顶层ACTIVE_RUN已切换新队列。
- 下一步：GPU空闲后自动开始；首轮核验neighborhood.json/386D tape/E optimizer_updates并确认下一轮三模型和地图身份；两轮完成按既定9条件报告泛化与遗忘，不自动扩轮。


- 2026-10-08T16:47+08:00实时回收：监督1258478存活，无新错误；第1轮正在P0固定压力基线评价，baseline_stress_00至04完成（5/9条件，160回合/64000实际控制步）。.25×onset0/10/20分别32/16/24成功；.4×onset0/10分别28/12成功，均分母32、unknown0。尚无C/新学生同面板对照，不能称闭环提升。baseline_stress_05等待GPU，额外12800仅为预留。
- 启动约54分钟，已完成5个物理子进程合计90秒，其余主要是GPU门控等待；Actor/E/G新更新均0。时间估计依据旧真实回执：老师搜索/重放58子阶段合计1374.52秒，学生128k单臂83–87秒，G2000更新232秒。新邻域两轮在GPU连续可用时预估1.5–2.5小时（含采集、老师搜索、训练和评估），属于估计而非承诺；排队额外计，24小时为声明停止上限。

### STTW-EXP-20261008-frozen-endpoints250 — 后续R244预算缩减

- user_requirement：后续两组从250改125。已核实队列当前R196_upper1，R196_upper0完成；R244两组均未启动。因此R196两组保留250，R244 upper_alpha0/1各125；lower_alpha始终1，其他算法/物理/奖励/种子不变。
- repo_artifact/reproduced：R244两份配置及严格加载器参数化、正式训练停止条件、状态declared_updates、最终checkpoint选择/导出标注同步；R196评价仍选择250，R244选择125。R244每组512×128×125=8,192,000策略转移/32,768,000底层tick，训练墙钟7200s/worker总9600s；每底层评价预算1200s不变。6项相关测试通过（含125预算训练终止），评价模块语法检查通过。无新增物理测试。
- 未停止/重启现有父监督或R196训练。父监督后续以新进程读取修改后的源码/配置，当前R196 worker已加载原250配置不受影响。旧父进程内存中的queue_status updates_per_group=250及原完成通知仍是旧统一字段，修订权威为run/budget_amendment.json、各worker frozen_config/status及更新后的INDEX；不会据旧父字段额外训练。
- 代码a6c481b，本地提交未push；原source_snapshot保留，变更存run/budget_amendment_source。下一步按原队列完成R196250评价、R244125评价后停止；当前只更改未来预算，未产生新的控制结论。


### STTW-EXP-20261008-R244125-only — 用户停止当前训练并优先best评价

- user_requirement：终止当前训练、开始后续两个125轮并分析R244；用户称当前R244 alpha1，实时核实为R196 upper1，因此按明确“当前”范围停止R196 upper1。用户随后要求bestmodel优先，无best才最后。
- artifact_checked：仅停止核验过cmdline的旧父监督799733和worker1102852（独立组），保存最后完整update0143；R196 upper0保留250完成。旧队列和R196 upper1状态cancelled，原logs/checkpoints/source_snapshot不覆盖。未停止其它项目、TensorBoard或修改物理。
- 新run：/home/qy/STTW_CONTROL/runs/worktrees/frozen-lower-upper-endpoints/runs/frozen_R196_R244_upper_endpoints_250_20261008/R244_125_only。父PID1298973、watchdog1298974；只R244 upper0/1各125，独立fresh，每组512×128=8,192,000转移，训练7200s/worker总9600s，6回合主/随机评价1200s。无R196评价插队，不自动续训。
- 新队列依配置声明每组budget与updates，避免旧统一250字段；每组短接口+8×16×2门槛不变。best按本组有限训练候选mean_step_reward选择，sampling_model_update明确属于更新前模型；末125未经后续采样不能获得旧奖励。阶段末生成best_model.json，评价优先它、无记录才last_completed；checkpoint缺失报错不静默另挑。并非固定场景best或成功证明。
- 7相关测试通过，含125终止、best优先/无best回退、采样模型轮次归属；无额外物理评估。TensorBoard复用6007、日志独立nested run，新正式reward尚待加载核实。下一步：验证R244启动/真实reward；自动完成两组和best配对评价；交付实际控制图与失败/梯度/KL/成本触顶等，不以训练奖励代替效果。


STTW-EXP-20261008-R244125-only 启动核实：R244_upper0已正式1/125、65536策略转移，R244_upper1排队；接口与8×16×2工程检查通过。TensorBoard http://localhost:6007/#scalars 的R244_125_only/R244_upper0/tensorboard/pilot实际train/mean_step_reward更新1已HTTP加载，凭据run/tensorboard_verified.json。运行中无收益结论。代码2ee6774本地提交未push；训练best优先、无best才last的选择已单测；配对review增加训练失败、roll范围违反、限幅、成本cap、梯度/KL和计算耗时JSON及曲线，原R196日志临时目录后处理测试通过，无新增物理。首正式更新无hardKL/nonfinite停止。下一步自动完成两个125及best配对评价并保存停止，用户之后查看新run/INDEX.md回收结果。


STTW-EXP-20261008-R244125-only 中途结果请求核实：用户要求分析两组，实际alpha0仅6/125、393216政策转移，alpha1尚未开始，父/worker仍存活、状态新鲜，无best配对控制评价。前6轮加权成本cap14.05%、roll工作界违反6.74%、最终执行限幅0.163%、上层参考变化率限幅99.06%。0物理失败但0完整回合结束（尚只768/800策略步），不能报告0%回合失败率。第6轮Actor/Critic梯度.3739/12.9036、KL.001796，无硬KL或非有限停止；可说优化正常推进，不可说控制已改善或两alpha取舍已成立。机器证据run/interim_first6.json。本次只读取既有日志，无新训练/评价/中断；保持两个125及best优先最终评价队列。


### STTW R196两组训练日志分析（2026-10-08，用户请求）

- artifact_checked：alpha0完成250轮、alpha1用户停止143轮；旧队列没有执行R196_comparison。两组均无best_model.json，按用户规则若评测使用最后250/143，不把事后单批最高奖励创造为best。未增加物理评测、不更改正在运行的R244任务。
- 为避免轮次混比，取共同94—143轮（50轮/8个回合周期采样长度）的逐tick加权RMSE。alpha0/1速度.05060/.02770m/s、转角.04548/.05028rad、航向.36640/.49333rad；冲突段速度.12969/.04242、转角.08756/.10727。训练内出现预期偏好：alpha1保速度、转角与航向更差；不是共享网络控制alpha的因果证明，也不是固定checkpoint基线收益。
- 恢复段航向RMSE.16996/.28854rad（9.74/16.53度），方向误差仍明显；没有独立保持评价，不能宣称恢复达标。alpha1前50到末50的转角RMSE.04270→.05028，有被牺牲任务恶化证据。
- 全训练物理失败alpha0为0/20480结束回合、alpha1为0/11264；无hardKL/nonfinite停止，实际梯度非零。末50成本cap分别.00462%/.00115%，roll工作界违反.0334%/.00119%，相对初期显著下降；参考变化率限幅仍97.59%/97.37%，最终执行器限幅仅.446%/.426%。控制效果不能用无失败或reward上升替代。
- 训练耗时107.56/63.44分钟（采样优化各update记录总和，不含工程准备）；训练周期明显，2.56s rollout短于16s任务，单批reward排名有采样阶段影响。
- 证据：run/R196_training_analysis/INDEX.md、tracking_training.png/pdf、training_diagnostics.png/pdf、window_metrics.json、training_summary.json、analyze.py；仅现有日志后处理，图已检查。下一步保持R244125队列；R196若需净收益结论须另作同条件冻结底层配对评价，本次未插队执行。


### STTW固定六场景测试（2026-10-08，用户纠正）

- user_requirement：给具体实验场景、后续统一复用此前测试，不能只分析训练曲线。核实旧lower_tracking_probe和tracking_candidate_review均使用straight_hold/speed_changes/gentle_positive/gentle_negative/steer_reversal/fast_turn六场景；旧证据是底层诊断，不是新R196上层250/143的评价。
- 规范：隔离工作树docs/FIXED_COMMAND_TESTS.md、learning/configs/fixed_command_six.json为fixed_command_six_v1。原指令逐行复用；seed77001，bank[3]、slew0.5m/s²/0.3rad/s，10s观察，指令窗口[1,6)、末保持[9.5,10)，速度.1/转角.04/航向.05分别报告，0.3rad历史侧倾诊断与原工作界分开。10s观察不改16s原始奖励/失败计分。
- 后续默认每个底层三方法B0（仅冻结底层）/upper0/upper1，6×3=18回合；best优先、无best才last。R244当前尚未评价，已将未来review入口从main/random替换为六场景，不叠加旧两场景，原1200s评测预算保留、不为完成面板超预算。训练进程与参数未修改/停止。修订保存在run/R244_125_only/fixed_panel_amendment.json及独立源码副本。
- 8相关CPU测试通过；10s旧轨迹临时后处理验证出图/完整窗口通过，无新增物理回合。六场景新上层GPU评价尚未执行，不声称已获得新结果。历史ECBC/底层结果只有身份一致时才可复用，不能伪装本轮匹配基线。
- 下一步：R244两组结束执行固定六场景；逐场景真实图与数值判定偏好/恢复；R196新上层仍缺这套评价，训练日志不替代。


### STTW R196固定六场景补测启动（用户明确要求立即执行）

用户要求直接测试，已启动R196 upper0最后250/upper1最后143（两者无best文件）的固定六场景三方法评价，原R244训练未停止。run=frozen_R196_R244_upper_endpoints_250_20261008/R196_comparison，18回合/36000底层ticks，上限1200s（首次前置检查未产生物理回合，剩余执行timeout1150s），原固定协议不变。评价显式允许用户已停止的部分训练checkpoint，不伪称143/250完成。

初次检查发现独立保存的训练准备bank非逐叶1e-6一致，拒绝后保存preflight_bank_check.log，未物理评价。核对两组model/controller/actuator/dt/substeps一致；评价统一对三方法复用alpha0的同一完整bank[3]，记录training_bank_parity=false/common_evaluation_bank，而非放宽误差阈值声称一致。物理和奖励不改。正式评价已编译启动；完成后核验18轨迹/奖励审计并交付逐场景图和效果。


### STTW R196固定六场景补测完成及指令图（2026-10-08）

- reproduced：18/18真实10秒轨迹完成、0物理失败；24条（含重评分基线）奖励/参考审计通过。upper0 last250、upper1 last143，无best记录；冻结lower_alpha1、同一完整alpha0 bank[3]。固定六场景/原窗口/物理/奖励不变。无新训练，不停止R244。
- 末0.5s速度/转角/航向三项共同保持：B0=2/6（直行、变速）、upper0=2/6（反向缓转、反转）、upper1=2/6（变速、反转）。上层有局部取舍收益，没有全面提升；单开发初态、不同训练长度，不称泛化/只改alpha因果。
- 急转：B0/alpha0/alpha1速度RMSE .0827/.1180/.0174m/s，转角.0582/.0738/.1003rad，峰值roll.4131/.2532/.2119rad，末航向RMSE.3158/.2897/.4372rad，全部未恢复方向；alpha1保速度姿态、牺牲转角/方向。直行两上层速度和方向退化；正向缓转全未通过；反向缓转alpha0改善恢复；反转两上层速度姿态与恢复改善但转角误差增大。
- 首次串行完成2场景、第三场景4秒后为墙钟效率切换剩余4场景并行；前缀原样保存partial_before_batch。总38400底层tick，2400为工程重算前缀；原1200秒墙钟内完成（完成时距内部1100秒限额约185秒）。batch_manifest、resume_budget、serial_status与两份execution日志可审计。重复前缀最大速度差.0061、转角.00049、航向.00018，批处理浮点轨迹不宣称逐位相同；各场景三个方法共同初态/批次。
- 用户追加原始/网络输出指令图：六场景*_command_references.png/pdf/csv已实测生成并核对governed=limited_command+offsets；原始slew前目标/实际原始指令、tanh映射后的网络请求、保护后governed分别显示，切换、末保持、真实失败终点标记。控制器收到governed，不拿真实车速轮角冒充网络指令。后续固定panel自动输出此图。
- 中文统一交付：/home/qy/STTW_CONTROL/runs/worktrees/frozen-lower-upper-endpoints/runs/frozen_R196_R244_upper_endpoints_250_20261008/R196_comparison/review/RESULTS_ZH.md；完整REPORT/metrics/audit/NPZ/CSV/PNG/PDF同目录。8项相关测试通过及真实18轨迹后处理检验，图已查看；代码6cdb3ce本地提交未push。R244未来评测采用六场景整体并行，原1200秒预算不变；下一步待其125两组与best评价结果，不追加训练。


### JIT-EXP-20261008-neighborhood-closed-loop — 共享GPU续接

- user_requirement：不再等待其他项目结束，允许并行。repo_artifact/artifact_checked：只停止核验过无GPU子任务的本项目旧排队监督；旧series_0002保留并标注cancelled/successor，shared_gpu_handoff.json记录原状态、64000已用交互与12800未执行预留释放。
- successor：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/series_0003_neighborhood_shared；代码da113ef。新gpu_shared仅移除compute PID阻挡，启动前仍要求20000MiB空闲；该条件不是显存预留或吞吐保证。其他项目未停止。
- 沿用原两轮学生256000/G4000/物理3300000及原started_unix的24h期限；5个P0基线结果通过spec/status/candidates/execution哈希及配置核对复用，64000计入当前成本，不新增训练轮数。Actor/G来源与fresh E语义、邻域、奖励、物理不变。
- reproduced：53项相关CPU测试通过，独立代码审查无阻断。共享GPU子阶段baseline_stress_05已启动（监督1437080，子任务1437522）；这仅证明调度通过，尚非闭环学习或泛化结果。TensorBoard http://127.0.0.1:6024 已HTTP200，新奖励尚未产生，独立reward verifier已运行；错误/完成桌面watcher已启动。ACTIVE_RUN已更新。
- 下一步：完成剩余基线/起始C压力评估后自动执行两轮Actor/E/G闭环；核验真实奖励和结果，按原预算停止。运行入口INDEX.md，原结果和冻结配置不覆盖；本地提交未push。


### STTW R196精简证据上传（用户明确授权）

用户要求只上传必要内容、不重复文件格式。已从远端feat/residual-recovery最新059bd3d建立独立证据分支reports/r196-six-scenarios-20261008，未上传未发布训练实现或模型。证据目录docs/evidence/r196_fixed_six_20261008，共15文件/8039118bytes（约7.7MiB）：12PNG（六场景实际运动/网络指令）、evidence.json（协议/指标/训练摘要/模型哈希/审计）、timeseries.npz（200Hz关键轨迹及奖励分项、共同原始参考去重）、README.md；无PDF、CSV、权重、训练日志或ZIP重复副本。18保留轨迹的速度/转角RMSE和奖励总和与原始结果一致，所有内部链接/文件哈希核验通过，完整原始证据本地保留。

commit=d269f71424b5eb393e7c5158333f2231eb687664；git push已成功，git ls-remote再次确认远端HEAD完全一致。远端入口：https://github.com/QaQaaa-zzz/STTW_CONTROL/tree/d269f71424b5eb393e7c5158333f2231eb687664/docs/evidence/r196_fixed_six_20261008。上传回执在原run/R196_comparison/upload_receipt.json。当前实现分支/其它项目未push，不修改运行中训练。


### STTW-EXP-20261008-R196-SmoothV4 — planned

user_requirement: execute supplied SmoothV4 with two independent warm-started Actors (250/143), fresh Critic/Adam/std, two value-only rollouts then at most60 policy batches per endpoint; lower R196 alpha1 and existing physics/permissions unchanged. Source6cdb3ce verified; evidence d269f71 is reports only. Worktree /home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4, branch experiment/r196-smooth-v4; run runs/smooth_v4_20261008. Offline saved NPZ audit complete, no new physics yet. Compute caps compile600/train1800each/eval1200/total5400s; fixed cases at20/60 with separate10s/16s fast_turn conclusion. Execution planned, evaluation not_started, adoption pending. Other jobs untouched. Next: implement/check, bounded training, physical comparison.

### JIT-EXP-20261008-neighborhood-200 — 已核验两轮，新增200轮

- user_requirement：先检查训练结果，无问题则继续200轮。本次按新增200轮（全局3–202），不重置原两轮，正式采用与实验续接继续分开；不放宽nominal采用检查，不改奖励/物理/H16/网络。
- artifact_checked：series_0003两轮完成，物理837540，学生256000，G4000，E优化8+6=14；85已执行子阶段完成，160条训练指标有限。P0/C/R1/R2固定开发压力成功186/228/112/204，各288，unknown0。两候选nominal均3/4未采用，继续Actor仍C；老师4+8回流，学生0；两次G选择incumbent。第2轮较P0增加18但较C少24，尚未证明累计Actor提升、G因果收益或独立TEST泛化。审计：series_0003/review/audit.json及INDEX.md逐条件表。
- 新运行已声明：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/series_0004_neighborhood_200。plan.json锁定前序published bundle/基线面板；代码2950a38，新immutable snapshot code_2950a38。起始Actor C，E续接round2优化器与邻域，G续接选中incumbent，累计数据/support/demo/tail/novelty完整继承；学生每轮Actor+normalizer warm start、critic/optimizer fresh。
- 预算：新增200×128000=25600000学生步，G≤400000更新，E每轮≤4epoch，物理≤300000000（每轮≤1500000），新7天墙钟；每轮前磁盘≥20GiB检查，共享GPU启动前空闲≥20000MiB；原始P0/初始C基线复用无重复仿真，仍每轮288压力回合。错误/非有限/预算/磁盘/截止即停止，不自动重试或扩展。按旧每轮19–20分钟仅估算约2.5–3天，资源竞争/搜索量会变化。
- reproduced：相关CPU261项通过；长轮次TB exporter按真实日志懒创建writer并在completed后关闭，避免预开400个writer，新增测试1项通过；独立续接审查无阻断。GPU本次续接尚待启动检查。准备完成不等于200轮完成。
- 下一步：启动并核验round3真实collection及E继承；持续记录每轮模型/压力评估/回流/成本与奖励；达到200轮或声明上限后停止。

- 2026-10-08T18:43:21+08:00 启动核验：series_0004监督1763377，通知1763482；当前全局round3/source_suffix_batch_0000，0/200新增轮完成。TensorBoard http://127.0.0.1:6025 HTTP200；真实奖励尚未产生，独立核验器1763378在监测。ACTIVE_RUN已更新，通知心跳无delivery_errors；源码2950a38本地提交未push，未改其它项目。原方案每轮完整执行，预算/磁盘/工程错误停止条件生效。


### STTW-EXP-20261008-R196-SmoothV4-fresh150 — user amendment

用户明确改为两端从零初始化，各150轮，批准训练4500秒/端、总10800秒，编译600/评价1200不变。旧warm-start run已按验证PID1778917终止：alpha0完成4批、alpha1未训练；无旧权重迁入新run。保留旧NPZ后处理及旧续训记录，未停止其它任务。新run=/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_fresh150_20261008；branch experiment/r196-smooth-v4。Actor345/Critic346独立端点、固定R196 lower_alpha1；全部上层Actor/Critic/Adam/std重新初始化。准备状态只复用旧完整物理bank。2批value-only单列，最多150批策略更新，20/60/150固定评价；执行planned，评价not_started，采用pending。12项相关检查通过；旧续训首批2条诊断轨迹的独立奖励重建最大误差<1e-8。新训练尚无控制结论。


### STTW-EXP-20261008-R196-SmoothV4-fresh150 — running, eval60/150 amendment

用户明确取消20轮面板，只做60/150完整六场景；要求保存bestmodel，后续使用best评测。以完整固定六场景物理失败/工作范围/共同保持/主目标/次目标/平滑度排序各端候选，best_model.json记录候选范围和是否合格，bestmodel.pt指向选中checkpoint；同协议已有实测复用其真实轨迹，不把last冒充best。20轮不运行、不参与排名。

当前唯一运行：/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_fresh150_20261008_run03；worker PID1821330，TensorBoard PID1821333端口6010（正式正step奖励HTTP加载核验待后续记录）。已核验两端initialization.json均scratch、source_checkpoint=null、previous_policy_loaded=false、std约0.1，空Adam、新Critic；seed81。配置validation_updates=[60,150]。前一fresh尝试发现旧initializer覆盖std为0.2已停止；下一次在首策略更新前因取消20面板修订重启；均没有权重转入当前run，已用时间累计结转：compile86.7489s、train0 239.3466s，其余见prior_attempt_budget.json，仍同4500s/端及10800s总上限。附件warmstart60原attempt取消4批记录单独保留。

13项针对性检查通过，包括std配置、同回合时间正则、value-only Actor不动、reward参考、hardKL回退及best不默认last。未重跑全仓/底层必要性证明。当前预热/训练运行中，evaluation not_started、adoption pending；没有新控制成功结论。不动其它项目/训练。只读进度：cat /home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_fresh150_20261008_run03/status.json；下一步：1. 有界完成各150或触发停止；2.60/150配对物理图与奖励重建；3.回收best和共同保持/平滑代价结论，不自动续训。


STTW SmoothV4 fresh150 run03 launch verified: upper0 2/150, upper1 0/150 queued; state=running. TensorBoard http://localhost:6010/#scalars 的alpha0/train/mean_step_reward正step1已HTTP加载；回执run/tensorboard_verified.json。两端fresh初始化、std0.1和validation_updates=[60,150]已实测读取核实。代码提交e1d5c702c09aa5090b95cc3ffa986a1689bf6808，本地提交未push。best将从完整60/150固定六场景物理候选选择，不将其称为全150轮逐轮评估最优。

### JIT-EXP-20261008-continuous-200 — 用户纠正：取消无扰动门槛，学生连续学习

- user_requirement：取消无扰动检查，不允许因退化回退到C，不允许每轮重新初始化学生学习状态。此前nominal4/4门槛与每轮fresh critic/optimizer不符合此次明确要求，本条覆盖旧方案。
- artifact_checked：只停止series0004监督1763377及其已核验子进程，停止于round3 teacher_0003搜索，学生尚未开始（0新增轮）。旧产物保留，continuous_learning_handoff.json/status记录取消；35470物理步按保守预留计入后继预算，未停止其它项目。
- 实现c30e0ac：continuous模式不执行student_nominal，也不执行source seed_support无扰动资格检查；改为锁定历史真实seed support，保留旧Actor成功标签，当前Actor标签只来自真实当轮评估。每个完整有限学生直接作为下一轮实验训练源，无性能采用门槛；正式采用仍false，实际TRAIN成功学生轨迹可回流。
- 学习状态：从旧R2候选（不是C）Actor/normalizer/critic开始。旧文件没有PPO optimizer/RNG，首次只能显式bootstrap optimizer一次；此后每轮保存完整TrainingState、optimizer、Actor/critic、normalizer、训练RNG和累计步数，强制继承前轮，缺失则工程错误。物理episode按新TRAIN支持重置，不声称恢复仿真接触状态。E及G原保存/选模协议不改。
- reproduced：CPU226项通过；engineering_continuous_0001真实GPU两段各3200步，总物理6754，92.90秒完成，第二段起始Actor/critic/normalizer/optimizer/RNG五项hash全部等于第一段结束，累计learner6400。TensorBoard http://127.0.0.1:6026 实际两段episode/sum_reward已加载。工程验证数据不回流生产，不能作为性能提升证明；独立复审无阻断。
- successor：/home/qy/DVGC/JIT/runs/experiments/generative_bridge_closed_loop_20261008/series_0005_continuous_200；immutable code_c30e0ac。原200新增轮（3–202）/25600000学生步/400000 G更新/300000000物理/原series0004起算7天，carried_costs=42224含中断35470与验证6754，未重置总预算或截止时间；20GiB磁盘和GPU共享启动门槛保留。
- 下一步：启动并核验真实collection、首个完整learner保存；持续200轮或预算/工程停止，记录压力泛化结果但不以成绩阻断训练。

- 2026-10-08T19:03:53+08:00 启动核验：series0005监督1833427运行，round3已完成真实collection，当前source_suffix_batch_0002。production.json continuous_learning=true，源Actor=2d188c3547a4c9abbc138b05fb5e50d87d8398c1725f89b8794e5bf63f12005f（旧R2学生），历史seed只读复用、无seed_support或student_nominal执行目录。通知心跳正常、无delivery_errors；ACTIVE_RUN已更新。TB http://127.0.0.1:6027 HTTP200，生产新奖励尚未产生，独立核验器1833428持续观察。旧series0004 exporter/TB/verifier已按路径核验后停止，未影响其它项目。c30e0ac本地提交未push。


### JIT-EXP-20261008-flowchart-audit — 用户要求流程一致性核验

- user_requirement：按上传E探索→G恢复桥接→学生模仿+RL→成功轨迹回流G/学习进展回流E→多轮迭代图核对，不允许隐含改变。此次仅审计和记录，未修改算法/配置/运行。
- artifact_checked：核对series0005 live plan/round3配置、五个关键冻结源码SHA一致c30e0ac；当前round3老师搜索/回放，0/200新增轮完成。主干、连续学生/完整PPO继承、H16同源尾段、保持项、仅真实TRAIN反馈、无无扰动门槛均对应图；真实新轮学习回流尚待该轮完成，不能把工程测试冒充200轮结果。
- 重要图外规则：G每次增量后按固定开发噪声MSE在incumbent/500/1000/1500/2000选最低，可保留旧G，未采用无条件last-G累积。旧两轮均incumbent。本次明确披露、不改此规则，不称三网都无条件最后状态连续。
- 其它明确细节：80/20 E/uniform，3tick/.25/start0；教师最多32根×(1source+16diffusion+15colored)；G state+history是76维3帧观测而非历史邻域；E有TRAIN历史邻域、source分账novelty；P0保持系数.2/demo.2→.05；无教案PPO继续、无新G数据跳过；主要工程停机含首loss行为KL差>.05，与成功率采用门槛不同。全部列在series0005/INDEX.md流程图审计表。
- 下一步：沿用户已授权连续学生200轮运行并回收实测；G选模若需改须明确为独立改动；逐轮报告新增/遗忘及回流，不以teacher成功替代学生或G因果优势。

### JIT用户停止训练 — 2026-10-08T19:24:05+08:00

- user_requirement：停止训练。只停止series_0005_continuous_200路径核验过的监督1833427及工作/通知子进程，未触及其它项目。repo_artifact/artifact_checked：user_stop_receipt.json记录停止前状态及PID，退出已核验；root/round/execution活动状态标记cancelled。停止在全局第3轮generator_incremental阶段，整轮0/200完成；保留所有已完成checkpoint/learner/评价/日志和未完成G工件，不自动恢复。当前入口series0005/INDEX.md。


### JIT-CHANGE-20261008-learned-only — 移除辅助随机分支及G选模解释

- recorded_at：2026-10-08T19:30:33+08:00；owner Codex本会话。
- user_requirement：接手DVGC/JIT，取消探索20%随机分支及老师32候选中的15条有色噪声；询问G“不是每轮无条件接最后权重”的含义。本轮未请求恢复训练；共享对话URL无可读取正文，因此依据本机源码/台账/原始选择收据接手，不声称已读网页全文。
- repo_artifact/artifact_checked：实际实现工作树 `/home/qy/DVGC/runs/worktrees/generative-bridge`，分支agent/generative-bridge-v1-1，父c30e0ac，本轮未提交补丁仅修改JIT闭环实现/测试/原工程计划；已有AGENTS/PROJECT/CURRENT_STATUS用户改动保留。新prepare/prepare-continuation显式写uniform_episode_fraction=0、teacher_colored_noise_candidates=0，并传到逐轮production/collection；E全部learned，老师1 source+16 diffusion=17，无候选补位。E/G自身随机采样保留。老师搜索和完整同批次回放均使用17 lane，source对照仍在lane0；旧无字段计划保留80/20和32候选语义，不覆盖旧冻结配置/结果。
- reproduced（仅工程测试）：52项针对性测试9.45秒通过；新4项先失败再通过，另验证拒绝把旧80/20记录当新协议使用；覆盖新旧round传播、采样、提案、完整回放、邻域证据及source冲突/恢复。git diff --check通过。未运行生产训练或物理回放；17世界批次未证明与旧32世界批次物理逐位等价，尚无性能结论。
- G选模解释（本轮不改）：`JIT/src/jit_dvgc/generative_bridge/diffusion.py:train_incremental`从incumbent训练最多2000次，每500次及末次保存；固定generator_dev噪声预测MSE（EMA）在旧G/500/1000/1500/2000中取最低，同分保留旧G；下一轮继承选中checkpoint的完整optimizer/RNG/EMA状态，不保证接最后权重。没有新数据跳过更新。噪声MSE不是物理恢复成功率。
- 原始结果核验：series0003 round1 `generator_incremental/attempt_0000/generator_selection.json` incumbent=0.035091135650873184，末2000=0.03627486526966095；round2 incumbent=0.03509169816970825，末2000=0.03657495230436325；两次选择incumbent。不能将执行4000梯度步说成已累计采用4000步模型。
- 活动状态：series0005 status仍cancelled、round3/generator_incremental中断、0/200完整新增轮；只读进程检查未发现对应训练worker。本轮无新运行、无训练成本，不启动/停止其他项目，不自动push。工程细节在 `JIT/docs/generative_bridge/CLOSED_LOOP_PLAN.md` 原位增量维护。
- 下一步：如用户要求运行，按新协议建立独立后继声明并核对停机learner/G来源、剩余预算；先验证17候选真实物理回放，再在明确预算内执行。G是否改为无条件last另由用户决定，本次不代替决定。


### JIT-AUDIT-20261008-core-diagram-and-push — 框图差异、G选模来源与远端交付

- recorded_at：2026-10-08T19:35:31+08:00；owner Codex本会话。
- user_requirement：分析`/home/qy/下载/JIT-core-method1.pdf`与当前实现，解释G新数据更新及固定选模集是否合理；随后明确要求检查完成把当前核心代码推送远端。PDF仅为审计对象，未当成启动/修改授权。
- artifact_checked：已渲染并阅读一页框图；主干E→扰动真实状态→G H16桥接→原pi尾段→老师教案→学生PPO+模仿+保持→TRAIN回流E/G对应。图未规定起扰0..2、幅度.25、每轮128采样/最多32老师根/新17候选、joint loss系数、同批次物理回放及G固定旧MSE选择。G输入76维真实3帧FIFO，不是E历史邻域地图；“回到可控状态”在代码中靠整段恢复成功/回放检验，没有独立可控集保证。泛化DEV回合不用于E/G训练回流。
- G确实以新成功轨迹训练：series0003 R1 history987/teacher_new4/actor_new0，R2 history991/teacher_new8/actor_new0，实际组概率2/3、1/3、0。停止series0005 R3 feedback_corpus已含history999/teacher_new9/actor_new88，概率.5/.25/.25且new_data=true；G阶段中断，无完成选模收据。cost_progress的1655为记录到的已计费更新进度，不声称完成2000或已选新G。整轮仍cancelled。
- 固定G验证集来源：`generative_bridge_v1_2_20260928/recovery_0002/campaign/generator_dev_fixture.npz`（JIT/runs/experiments下）；hash实算匹配d6df6e0fbf7044401e238fb2b6db41317760d949c158691fbecdd0f54bd8e848。早期bridge_baseline Actor a06acbc8，seed9281201；独立generator_dev64回合，起始0的3步uniform脉冲，幅度.1/.25各32；原始suffix成功30/32和31/32。61条成功轨迹构成候选动作窗口池，从中无放回取256个H16窗口，固定加噪时刻0..99与高斯噪声。256不是独立回合分母，manifest61祖先为抽窗口前池；不冒充256独立测试。随后一直继承同一文件，新的困难状态不进入该固定fixture。
- 分析/推断：固定fixture可提供可比旧分布去噪误差、零新物理成本；但唯一MSE选模不直接测H16+当前尾策略恢复，也不覆盖新救援分布，可能抑制适应。已知两轮拒绝新G，只证明权重未被继承，不能证明被拒G真实恢复更好/更差。该集是反复选模用开发验证集，不是最终TEST。框图本身不规定best或last，所以额外规则不是单凭图即可判定程序错误，但与无回退连续学习目标存在实质差别。
- 建议仅提出未执行：连续G使用最后有限完整learner继续，旧DEV-MSE保留监控/单独best档；改善与遗忘以固定旧+新恢复状态、同一当前Actor尾段的物理验证判断。若保留选模，需要预声明与任务相关的新旧DEV协议和成本。最后权重也不保证更好。方法参考Diffusion Policy原论文Appendix H.2的Push-T固定预算last选择，说明旧MSE选择不是扩散必需机制；不把该论文当本项目效果证据。
- 验证/交付：本轮184项bridge/generative_bridge测试62.23秒通过，另72项完整learner/执行门控/TensorBoard/邻域测试9.84秒通过，共256；独立只读复审无阻断，diff检查通过。未新训练或物理回放。工程审计原位写`JIT/docs/generative_bridge/CLOSED_LOOP_PLAN.md`。
- 用户授权push已完成：本地commit `16524224dbaea82bb92dd8744095bab123c40c30`，origin=https://github.com/QaQaaa-zzz/DVGC.git，分支agent/generative-bridge-v1-1。fast-forward由6f1335b到1652422，含此前6个未推核心提交及本轮提交；git ls-remote确认远端HEAD与本地完全一致。只提交9个本任务实现/测试/工程说明文件，既有AGENTS/JIT AGENTS/CURRENT_STATUS/PROJECT四文件未提交改动保留；未提交模型或运行产物。共享台账在STTW仓库本地同步，不声称其随DVGC推送。
- 下一步：等待用户决定是否更改G继承规则；如要求新实验再声明来源/剩余预算并验证17候选真实回放。训练保持停止，不自动恢复。


### JIT-IMPL-20261008-four-onsets-last-valid — 用户授权实现与条件启动

- user_requirement：按下载目录JIT_Codex_implementation_handoff_20261008.md实施四onset单脉冲/可变L和G last-valid完整学习状态继承；先代码、CPU、预算dry-run和两轮有界试点，用户最后明确试点成功后开启200轮。最后一句提供条件执行授权，不把原附件建议自动延长当授权。物理/奖励/H16/成功判据/完整学生/100%E/17老师候选保持。
- artifact_checked：实现HEAD16524224dbaea82bb92dd8744095bab123c40c30，与审阅基准一致；既有四个文档未提交改动保留。工作树仍generative-bridge，分支agent/generative-bridge-v1-1，不动其它项目/旧运行。
- 当前execution planned，软件implementation in_progress，物理pilot not_started。试点配置拟定2轮、每轮学生128000/G最多2000/E4epochs，物理1500000/轮、总3000000，24h；执行前需冻结实际来源/预算/版本、GPU共享门控、通知及TensorBoard。200轮仅在两轮工程验收全部通过后另立冻结计划。
- 恢复边界已知：series0005 cancelled且0完整新增轮、无current_source；其显式父series0003已发布round2。新恢复只选已发布bundle中的Actor/E/G等，不能用未发布round3或把evaluated_student替换已发布Actor。旧legacy完整learner缺失只允许一次明确bootstrap，此后精确继承。
- 工程验收不等于能力提升；按软件规则/实际物理/未验证结论分别报告。工程计划原位CLOSED_LOOP_PLAN.md；本次不重画用户框图。

### JIT-IMPL-20261008 软件冻结与试点准备

- repo_artifact/reproduced（软件）：commit0ff8266，独立代码快照`/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/code_0ff8266`；335项相关CPU回归37.67秒+5项只读验收测试1.15秒通过；独立源码复审无阻断；既有四文件修改不提交不覆盖。
- execution planned：新目录`/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/pilot_0001`，两轮（全局R3/R4），每轮学生128000、G最多2000、E4epochs；总物理上限3000000、24h；逐阶段最大1607808。E128回合每onset32，L3/.25单脉冲，各lane保存x3/8/13/18；100%E、1source+16G。严格CPU时序另覆盖L1/3/5/8。
- 来源：停止series0005的最后完整祖先series0003 round2已发布Actor C（8795fb23）、E/G/corpus/demo/support/map同bundle；学生legacy只允许本次首轮一次optimizer bootstrap，随后必须完整learner。G保留state20500，祖先实际保守累计计费28000，历史物理3521951，成本不清零、不混合中断R3组件。
- 资源：继承gpu_shared/free>=20000MiB，不等GPUidle、不停止别项目；启动前读到free21800MiB，磁盘318GiB。新计划/源码/依赖锁定后执行，异常停止不自动retry。独立通知和TensorBoard将核验。
- 验收`integration_audit.audit_integration`核验两轮完整发布、真实lane端点/mask、E/学生/G全状态父子链、TRAIN角色/成本；ready_for_200只指工程接受，不是能力性能门。G旧devMSE只监测，最后完整state携带EMA推理；不再退回MSEbest。可选G新旧物理配对监测本试点关闭（成本0），不得据此称恢复能力提升；固定学生DEV压力面板每轮一次，finalTEST关闭。

- 2026-10-08T20:34:38+08:00执行启动核验：pilot_0001 supervisor PID2126402，首轮R3 collection正在执行（0/2完整轮）；通知watcher PID2126504心跳正常且delivery_errors=[]。TensorBoard6028 HTTP200、exporter2126400/TB2126401；当前学生未开始，reward未产生，不能称reward已加载。入口pilot_0001/INDEX.md及launch.json；此时2304是collection保守预约成本，尚非完成交互实数。后续实测以状态/成本收据为准。

### JIT 四onset试点第一轮实测（pilot_0001，R3）

- repo_artifact/artifact_checked：R3完整发布，计费物理289294，学生训练128000+内部诊断688，E4epochs，G2000更新。collection四onset各32、快照分别x3/x8/x13/x18；有效1344/padding960/charged2304，最大请求约.24224，全部128为valid_post_pulse。老师10个source失败根实际搜索，4条教案完整验证。
- TRAIN配对负结果：before118/128、after106/128，新增2、丢失14。实验连续采用不表示正式部署通过，也不称能力改善。G新池history999/teacher_new4/actor_new106，实际比例.5/.25/.25。
- G实际receipt：起点20500→最后22500，实际选择update_2000；monitoring_best仍incumbent20500、旧MSE.0350915268、年龄2000，但不回退。累计保守计费30000。完整learner transition128000已发布。R4已从该包开始，第二轮完整学生optimizer/RNG初始化待核验。
- TensorBoard6028已通过HTTP检查round_0003_episodes的episode/sum_reward真实点，凭证monitoring/reward_http_verified.json；通知心跳正常。核心commit0ff8266已fast-forward推送origin，ls-remote核验一致。无最终TEST或G_before/after能力对照。

### JIT 用户取消轮询、直接条件接续200轮

- user_requirement：不用轮询，没问题可直接开始200轮。本次不再手动轮询；当前试点正常收尾，不中断、不把未完成R4拼入新包。
- reproduced：一次性激活脚本先验证Linux进程退出事件、PID出生身份拒绝及200轮固定预算；Python/libc无pidfd包装器，使用本机x86_64内核syscall434，已真实子进程测试通过，不改为轮询。脚本/声明及父试点plan哈希固定。激活PID2301938，文件`/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/handoff_200/activate.py`，入口同目录INDEX.md。
- 行为：等待试点监督退出事件→只读两轮验收→冻结独立200轮计划和验收哈希→GPU共享/free>=20000MiB门→启动200轮。任何失败停止、弹窗、保留收据，不自动retry。接续只用完整发布R4，禁止第二次legacy bootstrap。200轮root为`bridge_four_onsets_continuous_20261008/series_0002_continuous_200`（JIT/runs/experiments下）；300000000物理上限、25600000学生步、G400000更新、每轮1500000/E4epochs、7天，磁盘20GiB余量。
- 当前已核验R4学生128000训练+689诊断完成；初始化Actor/Critic/normalizer/optimizer/RNG五项均等于R3完整learner，offset128000。TensorBoard6028 R4真实reward已通过HTTP读取。未冒称R4 G已完成或200轮已实际开始；实际激活由handoff/status.json、pilot_acceptance.json与后继launch.json证明，并自动增量同步本台账。后继TB6029单独日志；文件事件驱动观察器会在真实reward可取时写HTTP凭证。
- 软件核心0ff8266已推远端；335相关CPU测试+5审计测试通过。G恢复能力提升尚未验证；首轮TRAIN118→106/128负结果保留。不改其他项目代码和历史结果。


### STTW-H009 / SmoothV4：延长到250并取消墙钟截止 — 2026-10-08T21:19:27+08:00

- user_requirement：两端总训练250轮；先批准追加9000s，随后明确“以后不要设置什么上限、不要纠结预算”，因此以最新指令取消额外墙钟/计算预算停止，轮数250终点保留，不能解释为无限续训。150测评移到250，60实测保留，物理best选择不变。
- repo_artifact/artifact_checked：当前parent upper0=150、upper1约120/150，运行中；runtime_control.json已禁用150评价。新阶段 `/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_extend250_20261008`；自动串行等待原worker结束后恢复最后完整学习状态。代码准备完成，15项相关检查通过；没有中断α1或其它任务。
- 初始化=learner_state_resume_environment_reset：恢复本轮fresh训练的Actor/Critic/Adam/log_std/RNG/accepted_policy_updates，不加载历史250/143策略、不用best回退训练、不重复2批Critic预热。旧checkpoint未保存完整物理/ESO/history状态，必须重建环境并选择未用episode key，不能称逐帧exact_resume。若旧硬墙钟使parent不足150，承接其最后完整checkpoint补至总250；不掩盖前序停止。
- 取消墙钟行为经检查：wall_limits_enabled=false时remaining无限、无SIGALRM截止，但compute ledger继续记录秒数；250轮数量终点、非有限/KL异常检查和不自动重试保留。新阶段execution=planned，evaluation=not_started，adoption=pending；250结果PENDING。
- 证据：[衔接入口](/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_extend250_20261008/INDEX.md)、user_amendment.json、parent runtime_control.json。启动后补queue PID和实际状态。下一步：自动安全衔接；训练至250；用60/250同六场景证据选best并记录局限，不追加轮数。


#### STTW-PERF-20261009 evaluation review and first warmup
- repo_artifact/reproduced: baseline_evaluation_100 trial001 completed six cases / 18 trajectories, all12 endpoint numerical reward audits passed. Warmup only: physics819.897549s, output0.260435s, report15.574993s (audit4.384817s, plotting10.931829s), total871.661110s. Remaining warmup/three hot samples/diagnostic still running, PID2180010; no formal training.
- Original-run comparison across every saved5ms field: 11/12 endpoint traces exact; gentle_positive alpha1 max XY4.77e-6m. B0 maxXY4.01e-5m, max per-tick scored reward difference3.75e-6. Physical failure endpoints/active lengths unchanged. Some B0 clipping/endpoint-extension boolean diagnostics differ, so all-field bitwise equivalence is NOT claimed. Receipt: performance worktree/runs/performance/baseline_vs_original_full.json. This is original-vs-replayed-baseline variation, not evidence of a batching regression or proof that the original is wrong.
- Read-only review found three issues: baseline recompile vs candidate cache must be separated from hot physics; equivalence receipt must retain numeric failures instead of passing arbitrary finite differences; candidate batching must retain persistent review compute accounting. All corrected in pending working changes. Focused12 CPU tests passed; actual18-lane equivalence remains pending. B0 numeric audit is additionally required in candidate report.

- Reporting correction (local commit, no push): fast_turn now retains both10s and16s report windows even after early failure; original saved trajectory remains truncated at true endpoint. A failing regression reproduced the old missing16s conclusion;9 reporting/evaluation helper tests pass after fix. Physics, reward and failure scoring unchanged. This is a verified original reporting defect, distinct from tiny floating discrepancies.

- Stage5 implementation committed locally as opt-in case-method batching (GPU A/B acceptance still pending, default sequential unchanged). Corrected timing/compute-accounting/strict numeric receipts, added independent B0 audit and process telemetry; focused11 tests passed, combined131 passed before final harness-only additions. Baseline two warmups completed (physics819.898s and750.990s); three hot samples remain. Sequential A/A safety/terminal/clock fields are exactly equal, while some continuous and clipping diagnostics differ. Candidate queue PID2255021 waits for baseline successful completion before running batched_evaluation_100; no overlapping benchmark GPU jobs or formal training.

- Benchmark reliability commit861cf1a retains each completed sample before a later error and records JAX allocator statistics separately. Three related tests pass. GPU statistics conservation investigation: within summary trial003, weighted family means exceed all-group mean by heading0.000448/speed0.000504/roll0.000474; baseline residuals ~1e-7, counts exact. Default dot precision is a hypothesis, not yet confirmed. GPU-only identical-input regression prepared; CPU run3 pass/1 intended skip. No reward or physics modification; training_summary remains opt-in pending diagnosis.

- Sequential evaluation required protocol completed: two warmups872.003837/773.724593s; three measured panels751.663091/752.835059/848.134005s, median752.835059s. All5 panels have18 NPZ trajectories and12 endpoint audits passed. Only the owned sixth duplicate diagnostic was SIGINT after verifying PID/cwd/argv and all measurement receipts; raw KeyboardInterrupt status is retained, with measurement_completion_receipt.json. No original training/evaluation was stopped. Candidate codebe423f7 avoids this redundant sixth evaluation because phases are measured inside every operation.
- Reproduced family reduction defect on4090D: DEFAULT dot relative error6.26136e-5 versus same-input float64 reference, HIGHEST1.63702e-7. GPU test failed before fix; all4 device-statistics tests passed after. Local commitffba76e changes only family-statistic dot precision, not controller/reward/physics.
- Batched evaluation now launched manually after the queue correctly rejected the optional-diagnostic KeyboardInterrupt status: PID2401558, root performance worktree/runs/performance/batched_evaluation_100. Original update100 checkpoint/lower/prepared identities, six cases,18 trajectories, two warmups/three samples, full audits and5ms plots, no sixth duplicate. No formal training.

- Batched first panel: all18 numerical reward audits passed, all safety/termination/clock arrays and active lengths match sequential reference; reported boolean physical conclusions unchanged. Strict all-field numeric comparison fails: max fast_turn XY difference0.0211668m (already in10s window), larger than sequential A/A; some clipping/projection diagnostics differ. B0 first physical-step speed differs by one float32 ULP before lower-action divergence. This is not claimed bitwise/full-state numerical equivalence or proof the old controller is wrong. Raw comparison: evaluation_AB_first.json and evaluation_AB_fast_windows.json. Default evaluation still sequential.
- Independent commits: fc6e48f adds env outcome receipts/device-stat timing;8e9a1ba switches current Smooth/direct entry from per-run cache to existing trusted local shared cache helper, respecting explicit settings. Six cache/dimension/harness tests passed; actual cache gain pending.
- Declared next engineering queue PID2452500: waits for five batched panels and all18 audits per panel; then cache first/reuse8x16, summary+guard8x16 full-state/reset verification and bounded trace, env N256/512/1024 withT128, candidate env512, two-endpoint8x16 end-to-end baseline/candidate, frozen-rollout PPO trace. Each timing mode retains2 warmups/3 samples; non-evaluation adds a diagnostic repetition. No formal training, no budget adoption or automatic2048. Queue aborts on errors. Runs are serialized and original experiments remain untouched.


### JIT 一次性200轮接续事件

- user_requirement：用户要求不再轮询、没有问题直接开始200轮。两轮试点已完整结束且只读验收ready_for_200=true；证据/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/handoff_200/pilot_acceptance.json。
- execution planned：独立200轮/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0002_continuous_200，源为试点最后完整R4包；G/学生/E完整继承，无legacy bootstrap；代码0ff8266，冻结预算学生25600000/G最多400000/E每轮4epochs/物理300000000/7天，单轮1500000，磁盘20GiB余量；资源门已通过，不停止其它项目、不自动重试/延长。固定旧DEV监测不否决，finalTEST关闭。
- 科研边界：试点工程通过不证明G能力提升；首轮TRAIN118→106/128的负结果保留。200轮评价尚未执行；启动前登记，后续launch.json记录实际PID。


### JIT 一次性200轮接续事件

- execution running：200轮监督PID2307372，已启动；入口/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0002_continuous_200/INDEX.md，TensorBoard6029。首轮训练奖励尚待产生，事件观察器会实际验证HTTP标量并记录，不把服务启动当奖励已加载。


STTW-H009 extension250启动登记（2026-10-08T21:21:30+08:00，repo_artifact/artifact_checked）：串行衔接监督PID2305993实际存活、state=waiting_for_parent，前序α0=150、α1=127，worker1821330仍训练。代码e1a6cc1f5a57b90b780207a69ce5bfa543cc1642本地提交未push；15项相关检查通过。parent runtime_control已禁用150面板，监督自动承接最后完整checkpoint至250，wall_limits_enabled=false。新阶段尚未开始采样，不能称新reward已加载；当前6010仍为前序真实训练，续训开始后6011由监督实际核验正step>150奖励并写tensorboard_verified.json。完整身份见/home/qy/STTW_CONTROL/runs/worktrees/r196-smooth-v4/runs/smooth_v4_extend250_20261008/queue_launch.json、queue_status.json、INDEX.md。最新规则已写共享AGENTS中的STTW训练时长默认规则；不再反复请求墙钟预算，250轮是终点。


### JIT 一次性200轮接续事件

- repo_artifact/artifact_checked：200轮新运行TensorBoard6029的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0002_continuous_200/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。


### JIT 一次性200轮接续事件

- execution failed/stopped：一次性接续任务/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/handoff_200停止，原因RuntimeError('200-round series stopped: {\'charged_interactions\': 870612, \'completed_rounds\': 3, \'error\': "ValueError(\'teacher replay invalid; never use empty-demo fallback\')", \'explorer_optimizer_updates\': 22, \'lifetime_physics_charged\': 5005129, \'phase\': \'failed\', \'supervised_updates\': 6000}')；不自动retry，不启动/延长未通过门控的训练。保留原运行和收据。


STTW-H009续训状态核实（2026-10-08T22:53:41+08:00，repo_artifact/artifact_checked）：extend250 worker2356405与监督2305993存活；α0=250/250已完成，α1=235/250、正在train1，不是最终评估。6011 HTTP实际alpha1/train/mean_step_reward最后step=235，旧6010是已结束的150轮前序，不能拿旧页面停止更新判断新训练停止。继承阶段已真实执行，150评价按要求跳过；250评价将在α1训练完成后执行。无额外墙钟截止，本次只核验和更新台账，没有重启或增加轮数。

### JIT 200轮重放拒绝修复与继续（用户明确授权）

- user_requirement：报错后查看、修改并继续；仍遵循不用人工轮询。两轮试点实际completed，试点G4000更新/物理612566；后继200轮完成3轮R5--R7后，在R8老师0001选中candidate6重放失败而停止。旧状态/失败目录不改。
- artifact_checked：search/repeat均17lane、同snapshot/context/Actor/normalizer/model/seed/动作文件；source0→0、candidate6为1→0，其余标签不变；选中lane首步qvel差3.814697e-6，G16步前缀后tail动作分歧。有限数值轨迹不重复已证实，底层数值差异原因未证实，不宣称修复了GPU确定性。
- implementation commit509920b：显式teacher_replay_failure_policy=reject_finite_candidate；要求同root/context/source身份/lane、完整hash锁定轨迹、有限数组、步数/真实终止或合法horizon、原suffix_label重建吻合；仅source0/0且selected1/0可标replay_rejected，teacher_found unknown、无demo、不进G老师池；独立TRAIN学生资格/反馈保留。未知/截断/非有限/身份异常仍停止；旧无字段行为不变；不挑其他候选替代、不放宽成功标准。
- reproduced（CPU/只读）：264相关回归35.40秒通过；15项新拒绝guard测试独立复核；真实保存17lane搜索+重放严格重验通过，0新增仿真/优化。证据`bridge_four_onsets_continuous_20261008/replay_failure_diagnosis_0001/strict_validation/diagnosis.json`。
- recovery execution planned：`/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0003_replay_recovery_197`；新代码快照code_509920b，最后完整R7唯一父包，学生完整learner累计640000，G状态30500/历史计费38000，E/corpus/demo/map同包；不采用失败R8组件。原200完成3，剩197从全局R8至R204，学生25216000/G394000/物理295500000上限，旧200已计费870612/G6000及全部历史物理5005129保留；合计不超过原预算。original_started_unix=1791465594.1204977，沿原7天截止、不重新计时；不自动retry、不额外延长轮次。
- 准备/预算dry-run已通过；实际launch/通知/TB6030状态随后记录。能力提升未验证，失败教案不作为成功数据。工程计划原位CLOSED_LOOP_PLAN.md，既有四个用户文档修改保留未提交。


### JIT 一次性200轮接续事件

- execution launched：JIT剩余197轮恢复监督PID2648946，入口/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0003_replay_recovery_197/INDEX.md；TB6030独立数据。新reward尚未产生，文件事件观察器会实际验证HTTP后记录；不轮询训练状态。

- 恢复启动健康检查（artifact_checked）：series_0003_replay_recovery_197 phase=running、globalR8 collection、新阶段0/197（原任务累计3/200），supervisor2648946；通知watcher2649046心跳且delivery_errors=[]，ACTIVE_RUN指向当前R8；TB6030 HTTP200。新学生训练尚未开始，未声称新reward已加载；文件事件观察器待真实标量生成后验证。Git远端ls-remote=509920ba90b81de795b83049669cf036587b8446。错误处理经CPU/原轨迹验证，新的物理轮次仍在执行，不声称GPU数值非确定性已消除或G恢复能力提高。


STTW-H009交付纠正（2026-10-08，user_requirement/repo_artifact）：用户指出精简报告漏XY图；已从原NPZ补六场景同图B0/upper0/upper1与原始积分参考，fast_turn延伸单列，无新增仿真。actual_xy/reference_xy等字段补入原timeseries.npz，图为publication/bundle/xy_trajectories.png。根与SmoothV4工作树及共享AGENTS新增XY交付硬检查，旧遗漏如实保留，远端补交已核实commit bb714255f1b70ccd8104841912ab04670dfa9a69，新增XY总图及18条实际/参考坐标；方向恢复不等于回到原路径。


### JIT 一次性200轮接续事件

- repo_artifact/artifact_checked：200轮剩余197轮恢复运行TensorBoard6030的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0003_replay_recovery_197/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。

### JIT-ERROR-20261009 Python退出阶段崩溃检查

- user_requirement：再次报错，请检查。本次只读检查日志、凭证、参数和进程，无源码修改、新仿真、重启或重标completed。
- artifact_checked：`/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0003_replay_recovery_197`已完成19/197，加此前3轮=原200任务22/200；最后完整原子发布R26。R27 supervisor错误generator_incremental_0000:failed，监督2648946已退出。
- 直接错误：2026-10-09T08:01:27+08:00，G worker PID408910退出码-6/SIGABRT；子日志execution/generator_incremental_0000/generator_incremental_0000.log末尾为`Fatal Python error: PyInterpreterState_Delete: remaining subinterpreters`、`Python runtime state: finalizing`。这是解释器退出清理崩溃；具体哪个库/线程遗留子解释器尚未定位。前面的XLA autotuning warnings不作为已证实根因。本次不是此前teacher replay invalid错误。
- R27实际产物核验：学生128000训练+744诊断完成、E更新completed；G selection/cost_receipt/completed.json/incremental_result.json均completed，2000更新，initial68500→last70500，累计计费78000，实际selected update_2000/EMA，旧MSE仅monitor。已实算校验manifest/state哈希并反序列化验证params/EMA/optimizer/RNG/76Dnormalizer/update计数完整且有限。未观察到参数非有限错误；磁盘可用290GiB。
- 必须区分：G工件持久化完成不等于整轮完成。R27 stages/generator_selection.json与current_source.json均不存在，因此仍是失败未发布轮，不能直接把R27零散组件冒充已发布状态包。保留全部证据及已计费成本；恢复需先定位退出清理并核验正式恢复边界，不忽略退出码或手工改completed。
- 原series当前已计费6479343物理、40000 G更新（包括R27），全谱系物理11484472。错误通知已送达、delivery_errors=[]。本轮不做能力结果分析，不声称G提高。下一步：定位子解释器生命周期；验证退出/恢复路径；再在原剩余轮数与预算内接续。


### JIT-RESUME-20261009 有界恢复、过程模型和连续曲线

- user_requirement：继续原200轮训练，核验每轮跳跃模型保存并修复TensorBoard跨轮不连续；不轮询、不扩大预算。
- software/artifact_checked：48796b9实现last_valid G正常返回后完整状态/父状态/累计计费验证、fsync和独立进程退出，训练或校验异常仍失败；原生库具体来源未知。新增严格同轮finalization publication恢复，公开resolver在pending期间拒绝续接，不改旧failed，不跨轮拼接。
- reproduced/CPU：主要回归262通过；最后凭证与发布门变更49项相关回归通过；真实R27 G状态70500完整树/父包/计费验证通过，原轮只读。恢复dry-run通过，实际新发布/续训尚待下项记录。
- 过程模型：R3-R27共25轮125策略checkpoint、1025完整learner状态哈希及继承链已核验；每轮策略0/32000/64000/96000/128000步，learner含Actor/Critic/optimizer/RNG/normalizer，每3200步保存。索引：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/continuation_visibility_20261009/INDEX.md；R27在该索引快照为学生完成、整轮待恢复。
- execution planned：新元数据发布series_0004_publication_recovery_R27，零新仿真/优化；通过所有审计后从完整R27接series_0005_continuous_recovery_177，R28-R204，共剩177，原任务23+177=200。拟用code_48796b9，完整学生累计3200000、G状态70500/已计费78000，E/corpus/demo同轮包。保留original_started_unix=1791465594.1204977及604800秒截止，20GiB GPU空闲资源门与20GiB磁盘门。
- monitoring planned：新TB6031以实际learner初始化凭证偏移拼接累计训练步（R3从0开始），未来不符偏移拒绝导出，原始日志不改；新轮reward由文件事件触发HTTP核验，不轮询训练状态。软件规则/文件保存/CPU通过均不代表G恢复或跳跃能力提高。

### STTW-H010 — R196 Preference V5 planned (2026-10-09)

user_requirement: execute supplied STTW_R196_Preference_V5_Codex.zip, both endpoint JSON semantic overlays. Frozen R196+ECBC+ESO, lower_alpha1, independent fresh upper Actors/Critics/Adam, no V4 upper loading. Branch experiment/r196-preference-v5 at /home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5, exact base e1a6cc1f5a57b90b780207a69ce5bfa543cc1642 verified; report bb71425 not implementation. Preserve lower_reference_centered and all physical/action limits; no automatic push or interference with R244/JIT.
Stage1 max40 updates per endpoint,512x128,5s finite random coordination task; paired left/right gate at20/40. Only if qualified Stage2 adds max80 to cumulative120 with same reward and learner state,16s dynamic/recovery task. No extra wall cutoff. Spec and JSON: worktree/docs/preference_v5/attachment. Run directory worktree/runs/preference_v5_20261009; configuration adaptation and focused checks in progress, no new physics/training yet.
repo_artifact/artifact_checked: existing V4 fast_turn[2,4) upper signed speed requests alpha0 +0.0265375, alpha1 +0.0373853 m/s; request/governed agree within float32; dv<=-.15/-.30 duration0. Supports small positive requests, does not support sustained braking canceled by interface. Full chain/mode-cost audit in progress. Prepared bank metadata confirms 2.6m/s indices5,6,7, zero-roll index6. execution planned/evaluation not_started/adoption pending.


### JIT 一次性200轮接续事件

- execution launched：原200轮已23/200，R28-R204剩177轮监督PID768222；新TB6031跨轮累计训练步，旧日志保留。新R28学生reward尚未产生，由文件事件观察器核验HTTP并记录。入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0005_continuous_recovery_177/INDEX.md


### JIT-RESUME-20261009 实际发布与启动核验

- repo_artifact/artifact_checked：code_48796b9冻结；Git远端agent/generative-bridge-v1-1已核对48796b954bcce297cd42f71f140526a2070d0cce。原4个未提交治理/状态文件保留、不入本次提交。
- publication completed：series_0004_publication_recovery_R27/current_source.json通过公开完整边界验证，recovery_receipt锁定原failed、全部原状态及计费来源；pending门仅在审计完成后移除。恢复零新增物理/学生/G更新，不伪造原series完成。继承学生累计3200000、G状态70500/历史计费78000；原任务新增23轮已全包发布。
- budget dry-run passed：原任务物理已计费7349955，加剩177轮上限265500000=272849955≤原300000000；学生2944000+22656000=25600000；G46000+354000=400000。original_deadline_unix=1792070394.1204977不重置。具体：series_0005_continuous_recovery_177/original_budget_audit.json。
- execution running initial health check：supervisor768222，globalR28，新阶段0/177，原任务23/200；资源门ready，ACTIVE_RUN已指向R28与新系列status；通知watcher768322心跳、delivery_errors=[]。未轮询等待训练完成；不自动扩大轮数。
- monitoring verified：http://localhost:6031/#scalars ，student_continuous/episodes 的episode/sum_reward实际HTTP加载1000点，step3200→3200000、单调连续；证据monitoring/continuous_http_verified.json。新R28 reward尚未在本初始核验中宣称产生，独立文件事件观察器将在实际产生后验证HTTP并写reward_http_verified.json。原日志未改。
- 文件入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0005_continuous_recovery_177/INDEX.md；过程模型索引continuation_visibility_20261009/INDEX.md。每轮策略checkpoint和每3200步完整learner继续保存，无模型清理。
- 结论边界：软件回归262+最终49相关测试通过；本次恢复审计未新增物理，续训监督已实际启动R28。此前R27已有学生128000与G2000真实执行，不能称此次重训；新的G进程退出路径长程稳定性及G恢复/跳跃能力提升尚未验证。

STTW-H010 prelaunch (2026-10-09): V5 implementation commit0a884ea local, not pushed. Focused reward/PPO/task/report checks76 passed; subsequent targeted55 passed after schema/peak gate changes. Single authorized smoke8env x16policy steps x2 updates completed256 policy transitions/1024ticks; no nonfinite/hard rejection. Actual training scalar steps1/2 HTTP-loaded at http://localhost:6012 (smoke/tensorboard/alpha0), receipt run/smoke/tensorboard_verified.json. Smoke used dirty implementation before final report/partial-failure logging fixes; formal source will be separately pinned. Offline chain raw-reference ECBC unavailable (u_nom governed-centered); request/governed small positive, no evidence warrants interface alteration.
Formal planned Stage1 two fresh independent endpoints,seed83,512x128,max40 each (5,242,880 aggregate policy transitions/20,971,520 control ticks), evaluation20/40 left/right5s from same highspeed bank6; B0 reused at40. Smoke excluded from formal budget. Exact lower actor/sidecar match checked via loader, physical/controller/actuator match old manifest. Additional Stage2 max80 each only after gate; no extra wall cutoff or automatic push. Execution pending final report check; no R244/JIT changes.

STTW-H010 launch: source1b7727f, trainer PID788181, run=/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5/runs/preference_v5_20261009, execution running/compiling, completed0/40 each. Formal launch.json records argv/time/source. TensorBoard6012 PID776844; smoke scalar verified, formal first reward PENDING. Stage2 remains gated, not launched.


### JIT 一次性200轮接续事件

- repo_artifact/artifact_checked：原200轮剩177轮恢复TensorBoard6031的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0005_continuous_recovery_177/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。

STTW-H010 user steering (2026-10-09): 用户明确要求“不轮询，结束后我喊你”。停止本会话主动监视和状态轮询，不停止已授权训练流程、不更改预算/gate。最后一次已核验状态为alpha0=13/40、alpha1=0/40，3072完整训练回合、物理失败0、硬KL拒绝0；这是停止轮询前快照，不作为后续实时状态。正式TensorBoard6012已HTTP核验加载alpha0正step奖励。既有流程按20/40左右检查、Stage1 gate及最大120条件预算继续，结束/错误通知保留。等用户再次指令后读取实际完成状态并分析交付；不声称训练/评测已结束。

### STTW-H010 — PreferenceV5 Stage1完成、gate失败停止（2026-10-09）

- **执行身份与状态**（`repo_artifact` / `artifact_checked`）：正式源码`1b7727f6f4ebf42db80f1ac2409f1a66d9c4f997`，分支`experiment/r196-preference-v5`，未push；run=`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5/runs/preference_v5_20261009`。α0/α1各fresh40/40、各2,621,440策略转移；两端各10,240完整训练回合，物理失败0、硬KL拒绝0、非有限停止0。执行`complete`，评价`complete`，采用`not_adopted`。Stage1 gate失败后流程按合同停止，`stage2_started=false`，没有10/16秒任务评价或自动追加训练。
- **五秒左右配对结果**：α0正/负速度RMSE=.6252/.6219m/s、转角RMSE=.01405/.01243rad、子步峰值roll=.2425/.2411rad、相对转弯前实际降速=.6902/.6868m/s且≥.20m/s持续2.5s；满足α0转角和持续减速。α1速度RMSE=.2630/.3093m/s，远高于≤.08门槛；转角RMSE=.03979/.02515rad；正转roll=.2980合格，负转=.3092超过≤.302。端点速度差.3623/.3126m/s、转角差.02574/.01272rad满足取舍方向；四条策略轨迹均完整5s且无物理失败，不能覆盖任务门槛失败。
- **控制链定位**：评测[2,4.5)s α0请求/governed Δv正负转约-.5768/-.5769与-.5422/-.5424m/s，最终后轮约20.83/20.88rad/s，实际速度1.975/1.978m/s；α1仍请求约-.1955/-.1968m/s，最终后轮24.76/24.37rad/s，实际速度2.337/2.291m/s。请求、governed、最终命令和运动同向，未见接口抵消；α1失败属于学习请求仍过度减速。raw-reference ECBC未单独记录，`u_nom`为当前governed-centered合成语义，不反推唯一分解。
- **收益、反例与成本**：α0相对update20进一步降低roll并提高转角精度，但以更大减速换取；α1未收敛至保速，负转仍越工作门槛。普通[.5,1)s α0/α1仍加速+.0622/+.0310m/s，实际速度误差+.0651/+.0341m/s，未消除无益干预。α0冲突成本前三为速度/转角/reference_priority；α1为速度/roll/reference_priority；本固定配对内compatibility未起量，不能外推训练全分布。四条reward独立逐帧重建最大政策步误差<4e-8且无分项触顶。训练reward改善而冲突速度误差可恶化，不作为成功证据。
- **结论边界与证据**：支持“α0学会持续减速保转向，两个端点产生明确方向性差异，执行链真实下发”；不支持“α1保速成功、共同roll门槛、普通无干预、10秒方向保持、16秒恢复、几何路径完成、泛化或实车”。XY首屏、实际控制、motor chain、逐步/累计reward和分项及原NPZ均在`runs/preference_v5_20261009/evaluation40`；综合报告`runs/preference_v5_20261009/analysis/REPORT.md`。保留update40 checkpoint为失败诊断候选，不称best/qualified/adopted，不追加250/500。

### STTW-H010 — 用户覆盖Stage1 gate并启动Stage2（2026-10-09）

- **user_requirement**：用户在已知Stage1 gate失败后明确要求“开始stage2，我们试试”。该指令只覆盖“不通过则不扩训”的门控；Stage1失败、`qualified=false`及其负结果保持不变，不能把Stage2启动记为门槛通过或采用。
- **恢复身份**（`repo_artifact` / `artifact_checked`）：源码提交`4e39bcf3ead681ad9bb2e14227e7a1d457d1e186`，未push。父run=`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5/runs/preference_v5_20261009`；α0/α1来源均为update40、warmup0、accepted_policy_updates40，checkpoint SHA-256分别为`5e60f85235a82110bbd5e264af4040d7d7e54da4b41c844efaeb7596b0d36e1e`、`f7ef272304ab19a40b828c615d32ee26b6302b2cdb06a40bb833e1c3cbe0e84e`。恢复Actor/Critic/Adam/log_std/RNG/accepted-update计数；物理、ESO、滤波与动作历史重置。Stage1/Stage2的reward/action/network/plant/limits/reference/lower/lower_reference_centered/actor temporal字段逐项一致；只按声明切换16秒任务与动态指令分布。
- **执行预算与状态**：新run=`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5/runs/preference_v5_stage2_override_20261009`，训练PID`981949`，Stage2从累计40继续，检查点80/120，每端最多新增80，不追加250/500。首批alpha0 update41已完成，mean_step_reward=-.0130646、KL=.001947、无非有限/硬停止；这是启动健康证据，不是任务成功证据。TensorBoard PID`981948`，`http://localhost:6013/`，API已实际加载当前run的`train/mean_step_reward`。相关测试87通过（1个PyTorch转换warning）。按用户既有要求，完成初始健康/标量核验后不主动轮询，等待用户通知后分析最终10秒/16秒控制证据。

### STTW-H010 — PreferenceV5 Stage2完成，整体未合格（2026-10-09）

- **执行与完整回合**（`repo_artifact` / `artifact_checked`）：两端累计120/120、accepted120、无stop；各新增80批、5,242,880策略转移、20,971,520控制tick、6,144个完整16秒训练回合，训练物理失败/硬KL/非有限均0。update80/120各完成六场景×B0/alpha0/alpha1共18个物理回合；fast_turn同一16秒轨迹分别报告10秒主窗口和16秒延伸。训练完成、评价完成、采用`not_adopted`；Stage1仍`gate_passed=false`，用户覆盖不改写其失败。
- **实际控制链**：update120 fast_turn[2,4)s，alpha0原始2.6m/s/.25rad→governed2.244/.235，修改-.356m/s/-.0149rad，最终后轮23.111rad/s，实际2.177m/s/.233rad；速度请求低于-.15/-.30连续2.62/2.44s，governed连续2.865/2.415s，未被接口抵消。alpha1 governed2.543/.205，修改-.057m/s/-.0452rad，最终后轮26.147rad/s，实际2.469m/s/.197rad：更保速但少转明显。raw-command ECBC仍未记录，不从governed-centered `u_nom`反推唯一分解。
- **安全、普通工况与恢复**：alpha0六场景峰值侧倾均≤.302rad；alpha1 fast_turn=.3150rad且越.3共1.755s，故共同侧倾门槛失败。straight_hold中alpha0/1把2.3请求提高到2.3715/2.3542m/s，实际2.3653/2.3481，速度RMSE.0602/.0433均差于B0 .0166，无益干预未消除。10秒联合速度/转角/航向末段保持alpha0=0/6、alpha1=1/6（仅gentle_negative）；fast_turn16秒两端均失败，末段航向RMSE .6042/.3507rad。fast_turn XY误差10秒9.269/9.356m、16秒18.794/15.169m。转角回零不等于方向或路径恢复。
- **80→120与奖励边界**：alpha1 fast_turn速度RMSE .1257→.0982、roll .3268→.3150，但转角.0440→.0525、航向.4670→.6837；alpha0速度.3421→.3116但航向.5814→.5973。继续训练未解决核心失败。12组策略轨迹逐帧奖励重建全通过，最大分项误差1.145e-5、政策步4.233e-8、分项cap触发率0；训练reward、存活和零摔倒不覆盖任务失败。
- **身份与证据**：源码`4e39bcf3ead681ad9bb2e14227e7a1d457d1e186`未push；最终checkpoint SHA alpha0=`54420b7b89269d8f196c403eb5fd7961097d695a0c494aabc94ecaf529614cc8`、alpha1=`c21d8443233b4a88e26eabae52a0212358df434628f46f5050127935da32560c`。完整实际控制图、XY、六场景表、训练/评价分离指标和原始NPZ入口：`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5/runs/preference_v5_stage2_override_20261009/analysis/REPORT.md`。结论：支持alpha0持续减速保留更多转向及接口链生效；不支持alpha1端点成功、共同安全界、普通无干预、10/16秒方向恢复、几何恢复或采用。不追加250/500。

### STTW-H010 — 精简证据包远端发布（2026-10-09）

- **user_requirement**：精简打包并推送图片、具体数值数据和TensorBoard图，且明确包含前40轮。已发布目录`docs/evidence/r196_preference_v5_20261009`，覆盖Stage1 update20/40和Stage2 update80/120。
- **包内容与复核**（`repo_artifact` / `artifact_checked`）：约20MiB；57个Git树文件（manifest登记其余56个），28张PNG。Stage1含正/负转的update20/40实际控制与XY、gate JSON、12行汇总及两个压缩5ms NPZ；Stage2含update80/120完整数值JSON、42行六场景/10s/16s汇总、控制链CSV、两个六场景压缩5ms NPZ、update120六场景控制与XY及fast_turn电机/reward图。TensorBoard图直接从两个event目录导出连续1--120更新，完整CSV 7,440行并保留stage/alpha/tag/step/wall_time/value。压缩NPZ保留原始命令、governed、实际运动、XY、motor chain、reward及有符号分项和失败标志；checkpoint仅提交SHA身份，不提交本体。
- **验证与远端**：README相对链接0缺失；manifest 56文件大小/SHA全部重算一致；四个NPZ键数246/246/738/738；两个alpha的`train/mean_step_reward`均精确覆盖step1--120；Stage1 12条件和Stage2 42汇总行集合核对通过。发布commit=`6a264f77468cc4090f8e79919027b333740e5dfa`，远端`origin/experiment/r196-preference-v5`已用`ls-remote`核实同commit，raw README可读取。入口：https://github.com/QaQaaa-zzz/STTW_CONTROL/tree/experiment/r196-preference-v5/docs/evidence/r196_preference_v5_20261009 。未改变原始run、checkpoint或结论，未追加训练。

### STTW-H011 — R196 Preference V5.1 strong-primary（2026-10-09）

- **用户要求/方法身份**（`user_requirement` / `artifact_checked`）：执行`STTW_V5_1_StrongPrimary_Codex.zip`及两端JSON，以已发布V5实现`6a264f77468cc4090f8e79919027b333740e5dfa`为精确基线；正式α0/α1上层均fresh。冻结R196 lower_alpha1、ECBC/ESO、`lower_reference_centered=true`、物理、执行权限、345/346网络和动作映射。仅增加按端点选择的实际主目标超容差成本weight40/cap400，以signed `yaw_recovery`替换`yaw_damping`且cap20，并从第一批使用16秒30/40/10/20四训练族。失败组件上界1680→2080，不引入总成本裁剪。每端512×128×最多100，seed87，update25看straight_hold/fast_turn，update100看原六场景；不自动追加或push。
- **实现与限定检查**（`repo_artifact` / `reproduced`）：隔离worktree=`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5-1-strong-primary`，branch=`experiment/r196-preference-v5-1-strong-primary`。附件数学输出与`math_checks.json`一致；V5.1聚焦测试6通过，原V5 reward/task/PPO 74通过。强制heading-family reset核实物理qpos/qvel不变、虚拟参考yaw=e0、冻结lower从actual pose初始化、外部评价rows使e0=0。独立工程run=`runs/preference_v51_engineering_8x16x2_20261009`完成8env×16step×2 accepted updates，256转移/1024ticks，无物理/policy失败或非有限停止；仅工程证据，不支持控制收益。
- **当前状态**：execution=`planned`（正式run尚未启动）；evaluation=`not_started`；adoption=`pending`。正式run将写入同worktree新目录并使用独立TensorBoard；两个正式learner不会加载工程/V5权重。下一步：完成diff审查和fresh验证；提交本地实现；启动100更新、核验当前run reward标量后按用户要求停止主动轮询，等待结束后回收25/100物理结果。
- **正式启动核实**（`repo_artifact` / `artifact_checked`）：实现提交=`071fc6f1366736ea1824389bda3686f58b1232ba`，本地未push；run=`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5-1-strong-primary/runs/preference_v51_strong_primary_100_20261009`。trainer PID1647623、TensorBoard PID1647622，execution=`running`、evaluation=`not_started`、adoption=`pending`；首次核验α0=1/100、α1=0/100，首批reward=-.0387873、KL=.00151062，无hard-KL/nonfinite停止。两端`initialization.json`均scratch、source_checkpoint=null、optimizer_empty=true、std约[.30,.10]、seed87，工程权重和V5权重均未加载。TensorBoard `http://localhost:6014/` 已由HTTP API实际加载当前正式run的`alpha0/train/mean_step_reward` step1，回执`tensorboard_verified.json`。初次普通后台启动随shell退出且未创建manifest/状态，已用setsid重新启动并记录launch.json；没有权重或更新被继承。下一步由有界流程训练到25/100并执行声明评价，任何奖励重建失败均置error；不主动轮询、不追加250/500，等待用户通知后回收实际结果。
- **源码远端发布**（`repo_artifact` / `artifact_checked`）：用户明确要求推送代码。分支`experiment/r196-preference-v5-1-strong-primary`已发布到`origin`；`git ls-remote`确认远端与本地均为`071fc6f1366736ea1824389bda3686f58b1232ba`。首两次HTTPS push分别遇到HTTP/2 framing和HTTP/1.1 empty-reply，远端均未建分支；确认增量仅32对象且最大约170KB后，使用HTTP/1.1及增大postBuffer的普通非强制push成功。未提交或推送run/checkpoint/cache；正式训练进程不受影响。


### JIT-ERROR-20261009-R31 训练中SIGSEGV与待确认重试预算

- user_requirement：报错；沿既定原200轮目标调查修复，保留失败记录及冻结预算，不跨轮混用状态。
- artifact_checked：series_0005_continuous_recovery_177完成R28-R30共3轮，原任务26/200；R31的G进程1339240于2026-10-09 13:03:49退出-11/SIGSEGV，CPU解释器执行期间崩溃，非上次退出阶段-6。内核IP0x1817481定位_PyEval_EvalFrameDefault+18113；没有Python线程栈或可用core，不能据此指定某个库或硬件故障。
- R31：学生128000训练+706诊断完成，G最后计费proposal1168、最后完整保存update_1000/累计77500，完整状态hash与有限性CPU核验通过；缺少completed.json/incremental_result/process_commit，不是整轮完成。原失败成本2000 G保留，无计费退款；不使用finalization recovery放行-11。
- software：2e296f2增加worker导入/执行前faulthandler全部Python线程栈；实际CPU SIGSEGV故障注入由无栈失败到有栈通过，相关20项测试通过，独立审阅无阻塞。此为诊断改进，根因尚未修复。本次没有新增训练/GPU仿真或重启。
- concrete retry proposal：最后完整R30作为唯一父包，重试R31至R204共174轮；原任务学生已消费27*128000，另174*128000=>25728000（比原多128000）；G已计费54000+348000=>402000（比原多2000）。物理上限269615569仍≤原300000000；七天截止1792070394.1204977不变。因用户原要求冻结预算，额外额度未获授权，未启动。
- 证据及待确认预算：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/segfault_diagnostic_20261009_R31/INDEX.md、evidence.json、retry_proposal.json（approved=false/execute=false）。旧模型、日志、状态不改；通知已投递，delivery_errors=[]。无G恢复能力提高结论。


### JIT-RETRY-20261009 用户授权继续及初始化核验

- user_requirement：用户明确“以后都不用管预算的事情 先跑通 再说别的”，覆盖上次因额外128000学生步/2000 G计费而等待确认。今后本任务日常故障重试不再因预算重复询问，仍记录真实消耗、保留失败记录及资源门。当前继续原200个完整轮目标，不据此创建无期限训练。
- execution planned：code_2e296f2从最后完整R30状态包恢复，series_0006_r31_retry_174，R31-R204共174轮；保留失败R31学生/G及全部成本，不跨轮拼装。此次是带faulthandler诊断的重试，SIGSEGV根因尚未确定。
- 初始化artifact_checked：当前E collection_spec无initial_velocity_randomization，pulse_schedule.initial_velocity_randomization各默认半幅0；_natural_reset_sample从固定XML keyframe及配置速度构造，_reset_jump_start_unified固定x=2.5；探索物理初态无额外姿态/速度随机化。学生ProbeEnv.reset每回合bernoulli(.2)：20%固定起跳点，80%从TRAIN支持状态池随机取完整快照；并非在快照上随意加qpos/qvel噪声。该20% reset比例与已取消的探索20%uniform扰动分支不同。网络参数继承，不每轮随机重置。


### JIT 一次性200轮接续事件

- execution launched：原200轮已26/200，R31-R204剩174轮监督PID1474216；新TB6032跨轮累计训练步，旧日志保留。新R31学生reward尚未产生，由文件事件观察器核验HTTP并记录。入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0006_r31_retry_174/INDEX.md

- JIT-RETRY实际启动核验：series_0006_r31_retry_174监督1474216、globalR31运行中，新阶段0/174、原26/200；ACTIVE_RUN指向新R31，watcher1474391心跳且delivery_errors=[]。TB http://localhost:6032/#scalars 实际加载1120个奖励点，累计继承链至3584000（R3-R30）；原失败R31曲线留6031，不混入新继承链。新R31学生reward由文件事件另行核验，当前未冒称产生。全包父G状态76500/累计计费86000（含失败预留），新代码2e296f2，不改初始化/物理/奖励/成功标准。入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0006_r31_retry_174/INDEX.md。


### JIT-RESET-CANCEL-20261009 用户撤销初始化噪声修改

- user_requirement：用户明确“不修改了，继续训练就行，回退刚刚的内容”；撤销此前速度、位置、姿态及hip/knee初始化噪声请求。
- repo_artifact / artifact_checked：仅恢复本次修改的iterative_probe_training.py及移除本次新增测试；原有AGENTS/PROJECT/CURRENT_STATUS未提交修改保留，没有新增噪声规则。
- series_0006已因协议切换准备而停止，0个完整轮次，停止凭证保留。准备series_0007_original_reset_174，以未改动code_2e296f2及原profile从完整R30接续R31-R204；不复用部分R31，不改旧失败/停止状态。原目标已26/200，初始化保持原协议。


### JIT 一次性200轮接续事件

- execution launched：原200轮已26/200，R31-R204剩174轮监督PID1524710；新TB6033跨轮累计训练步，旧日志保留。新R31学生reward尚未产生，由文件事件观察器核验HTTP并记录。入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0007_original_reset_174/INDEX.md


### JIT 一次性200轮接续事件

- execution launched：原200轮已26/200，R31-R204剩174轮监督PID1533173；新TB6034跨轮累计训练步，旧日志保留。新R31学生reward尚未产生，由文件事件观察器核验HTTP并记录。入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0008_original_reset_174/INDEX.md


### JIT-RESET-CANCEL-20261009 原协议恢复已核验

- series_0007监督进程缺少JAX_PLATFORMS=cpu，被启动保护检查拦截，零新增物理/优化；失败目录保留。修正启动环境后在新series_0008_original_reset_174启动，原code_2e296f2和profile均未改。
- 当前status.phase=running，R31，新阶段0/174、原目标26/200；完整R30包继承，监督PID1533173，桌面watcher1533273心跳且delivery_errors=[]。原有文档未提交修改保留，源码/配置/测试diff为空。
- TensorBoard http://localhost:6034/#scalars 已HTTP核验1120个实际历史奖励点至累计3584000步；新R31奖励尚未产生，文件事件观察器待实际产生后核验。未把启动/历史奖励当新训练完成。
- 入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0008_original_reset_174/INDEX.md。旧SIGSEGV根因仍UNKNOWN，本次未修改算法或宣称修复该根因。


### JIT 一次性200轮接续事件

- repo_artifact/artifact_checked：原200轮剩174轮恢复TensorBoard6034的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0008_original_reset_174/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。


### JIT-PERFORMANCE-20261009 加速实现与独立短基准

- user_requirement：按JIT_4090D_acceleration_Codex_handoff_20261009.md优化，不减候选/重放/G2000更新/PPO128000步，不改物理/奖励/连续继承。当前正式series_0008使用旧冻结code_2e296f2，不热改、不自动迁移。
- repo_artifact：实现commit5b2d867，CPU相关回归303通过；随后worker输入锁补充及缓存命中日志修正的33项回归通过。采样索引、轨迹/输入验证缓存、历史语料引用、设备端有限性、紧凑样本来源、opt-in B1常驻教师、诊断CLI已写。B1物理一致性未验证，不得生产采用。
- reproduced：原R31锁定14981文件合计9684819782字节，首次缓存验证4.7363秒，第二次0.0773秒；篡改/恢复mtime回归拒绝。不是整轮加速证明。
- 独立A→B→A物理短基准已启动：performance_benchmark_20261009，冻结2个TRAIN root和17候选原动作，baseline及B1各3请求，最多40800物理步，每臂900秒；不学习、不反哺、不替换正式模型。监督2016783，既有GPU共享门控。详细声明/plan/原始输出在该目录，失败也保留。


### STTW R196 Preference V5.1 update100 完成及评价预算故障修复（2026-10-09）

- execution/result：alpha0、alpha1均完成100/100更新；初次update100评价在straight_hold后因评价tick错误计入已用满的训练上限而停止。训练与update_0100 checkpoint当时已完整。修复commit `77f715f` 将V5.1评价从control-tick预算器分离；补跑仅加载既有策略，完成其余五场景，无新增PPO batch或checkpoint写入。远端分支 `experiment/r196-preference-v5-1-strong-primary` 已推送。
- physical evidence：fast_turn[2,4)s alpha0转角RMSE=.01862rad通过.05；alpha1速度RMSE=.04997m/s通过.08。共同侧倾失败：峰值=.31540/.36761rad，越.302持续2.160/2.445s。六场景10s联合最终保持alpha0=3/6、alpha1=1/6；fast_turn在10s和16s均未恢复航向，16s航向RMSE=.07749/.24961rad。straight_hold速度偏置=.0753/.0442m/s且误差劣于B0。12条endpoint reward重建通过；不以此替代物理失败。结论unqualified、不采用、不续250/500。
- user_requirement：后续训练按明确env/rollout/update规模停止；评价与证据生成不得再用跨阶段control-tick硬预算中断。避免额外SHA/mtime对比，直接完成运行并分析保存数值。
- evidence：`/home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5-1-strong-primary/runs/preference_v51_strong_primary_100_20261009/evaluation100/RESULTS_ZH.md`及`stage2_metrics.json`。

### STTW-PERF-20261009 — RTX4090D execution optimization

- user_requirement: sequential restored-state benchmarks and isolated A/B; unchanged V5.1 method/physics/PPO/evaluation contract; no formal training or continuation, no DVGC/JIT changes or stopping existing jobs.
- artifact_checked: main HEAD1aa2e63 has user documentation changes preserved. Target branch clean at77f715f, newer than audit071fc6f by evaluation tick-budget repair; retained. Historical status complete alpha0/alpha1=100/100, PID1647623 absent. Launch argv and frozen512x128/seed87 config checked; GPU RTX4090D driver580.159.03.
- implementation: worktree /home/qy/STTW_CONTROL/runs/worktrees/performance-4090d, branch performance/r196-v51-4090d. Benchmark reuses SmoothCampaign. Baseline regression96 passed; harness restore-order tests2 passed. No measured speedup or adoption yet.
- engineering budget: each mode2 warmups+3 restored-state samples plus1 instrumented repetition; start env-only8x16, then fixed512x128 baseline. Fresh independent endpoint learners; prepared bank6bb5fc54499ecc4451251bc984c1d4459a32977ba92f09fb53d3382c5bf535d0, frozenR196 lower_alpha1. Stop on numerical/engineering errors, no added training wall termination. Raw output under performance worktree runs/performance/.
- next: establish baseline; sequential equivalence/A-B commits; publish measured and unmeasured stages honestly.


### STTW R196 Preference V5.1 远端结果包（2026-10-09）

- user_requirement：将V5.1已完成结果推送远端，包含数值、图片、TensorBoard图；此前40轮属于V5历史包，V5.1本轮从零训练的指定检查点为update25和update100。
- repo_artifact / artifact_checked：已在`experiment/r196-preference-v5-1-strong-primary`推送commit `d6e5688`。精简包入口：`docs/evidence/r196_preference_v51_20261009/README.md`，含update25两场景和update100六场景的XY/控制/逐步及累计奖励/分项PNG、两份5ms轨迹NPZ、逐工况数值CSV、完整指标JSON、TensorBoard静态图与6200行标量CSV、两端冻结配置与原始错误/补评收据。校验包内31个报告链接、update25六条及update100十八条配对轨迹，均存在；仅后处理，无新增仿真或训练。
- result/adoption：沿用已核实的负结果：两端个别主误差达标，但侧倾和航向恢复失败、普通直行速度仍被无益改变；不合格、不采用、不续训。完整run仍在隔离worktree的`runs/preference_v51_strong_primary_100_20261009`；远端包未提交checkpoint或缓存。
- 远端入口页面已HTTP打开核验；修正轨迹键格式的Markdown显示后，最新远端HEAD为`26c3fed`。


### JIT 原200轮用户暂停（2026-10-09T17:02:22.024237+08:00）

- user_requirement：先暂停原来200轮训练。已核对并终止 series_0008_original_reset_174 监督1533173及其正在运行的R33 student_stress_08子进程2141057；复核均已退出，未停止其他项目。TensorBoard6034及历史日志保留。
- artifact_checked：本续接完成2/174轮，原200轮累计28/200；最后完整发布状态为R32，series/current_source.json与R32/current_source.json内容相同，R32 status completed。R33未完成，不采用其部分状态，不伪造completed。后续仅从最后完整状态包续接，不自动重启。
- evidence：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0008_original_reset_174/user_pause_receipt.json。顶层status记stopped/user_requested_pause。
- acceleration integration correction：performance_integration_0001_20261009仅已准备，资源检查失败，未启动；此次暂停不自动触发该试点或新200轮。

#### STTW-PERF-20261009 progress: first measurements and strict regression failure
- Local commits10d2a91 benchmark,271b48b config dimensions,74883d6 opt-in device statistics. Formal defaults512x128/100/25,100 unchanged; no push yet.
- RTX4090D512x128 restored rollout samples baseline[28.918925,27.907847,31.868537]s; opt-in summary[27.643317,32.112284,27.648829]s. Median reduction4.4% but overlapping ranges, not established robust improvement. GPU metadata/raw timings under performance worktree runs/performance/.
- 8x16 env baseline[5.131330,4.338621,4.378705]s vs config fix[4.245542,4.248834,4.255774]s; all512 active ticks. Config change is correctness, not claimed kernel gain.
- Synthetic episode/dwell/reset tests passed including exact float32 boundary reproduction of old float64 threshold comparison. Physical full-state summary-vs-full comparison FAILED original rtol2e-4/atol2e-5 (one observed difference4.334e-5). A/A diagnosis pending; summary remains opt-in, not adopted. Reset batch guard candidate tests4 passed, GPU A/B in progress; no sparse-reset claim.
- Current benchmark TensorBoard http://localhost:6015/ verified reward via HTTP, repeated step1 are restored trials, not continuous training. Other jobs untouched.


### JIT-PERFORMANCE-20261009 完成实现与短基准，继续独立集成

- user_requirement：暂停仅针对原200轮，明确继续优化任务。原series0008保持stopped，不自动恢复。
- repo_artifact：c70af4f已推送origin/agent/generative-bridge-v1-1；新增缓存硬链接别名失效回归。307项CPU回归通过，随后CLI两项通过（含新prepare回归）。保留四份原有用户文档修改。
- reproduced：完整G worker100更新端到端62.93864→43.53951秒（约31%减少），25600样本来源逐条一致；RNG/normalizer/count精确一致，params/EMA/Adam通过预声明校准FP32容差。原紧容差失败和旧代码自身重复差异保留，不隐去。
- negative evidence：常驻B1 A→B→A成功标签不一致，不采用；不同实际物理步数下的60.13→31.98秒不能称正确加速。B4/B8未执行。整轮加速与G恢复能力改善未验证。
- next：准备c70af4f冻结快照的独立1轮集成performance_integration_0002_20261009，仍用原教师物理路径；旧attempt0001资源阻塞未启动。本条不宣称新试点已启动。报告JIT/docs/generative_bridge/PERF_AUDIT.md、benchmark.csv和correctness_report.json已推送。


### JIT B1原实现重复稳定性补查（2026-10-09）

- user_requirement：继续优化；用户指出差异可能来自原实现，并倾向采用常驻方案。保留为实验候选，不把原实现当绝对真值、不伪造已通过物理验收。
- execution：performance_repeatability_20261009已通过原20GB资源门控启动，监督2168163；原始fresh-process与常驻B1各6次固定TRAIN请求A/B/A/A/B/A，最多81600物理步，无训练反馈，无新增200轮。watcher2171093 heartbeat已确认。
- boundary：原series0008维持暂停；独立1轮集成0002已prepare但尚未启动，先完成此次有界重复性诊断以避免GPU竞争。


### JIT B1重复性补查结果与单轮试用实现（2026-10-09）

- reproduced：六次A/B/A/A/B/A实际GPU请求，两臂各自标签稳定，跨臂六次成功ID均一致（A=[1,7,13]、B=[]）；初始观测和动作精确相同，但原实现首步qvel重复差异可达0.006，常驻0.00572。之前偶发标签差异仍保留，根因UNKNOWN，不将旧实现视作绝对真值。
- performance：fresh-process112.92269秒、12954 charged steps；常驻38.87275秒、7582步。实际物理工作不同，不称等工作量2.9倍加速。
- user_requirement/decision：用户明确倾向采用，推进一次有界实验试用；不是伪造正式验收。新增输出路径/轮号/commit/<=150万物理步绑定的trial receipt，formal_acceptance=false；plan preflight要求恰好1轮且无自动延长。常规正式验收门保留。
- repo_artifact：1a4d3ef，核心相关249回归通过；随后trial预检及profile9测试通过，独立审查无单轮阻塞。准备performance_b1_trial_0001_20261009从R30完整包独立执行R31，保持G2000/PPO128000和完整搜索重放。原200仍暂停。


### JIT 一次性200轮接续事件

- 独立1轮加速集成已启动：performance_b1_trial_0001_20261009，完整R30起点，单轮授权常驻B1，17-world完整搜索重放，PPO128000/G2000，最多150万物理步；不计入原200轮，不自动采用或扩展。监督PID2192267，TB6035新学生奖励待产生后由文件事件核验。原series0008保持暂停。


### JIT B1单轮试用启动核验（2026-10-09）

- artifact_checked：performance_b1_trial_0001_20261009已启动，status running R31，0/1完整轮；监督2192267，watcher2192375 heartbeat正常无delivery_errors。原series0008保持暂停；未启动其他集成attempt。
- visibility：TensorBoard http://localhost:6035 HTTP200已核验；当前学生PPO尚未产出奖励，scalar接口404，不能声称奖励已加载。文件事件观察器2191998会在产生后核验并保存monitoring/reward_http_verified.json，不轮询训练状态。
- code：1a4d3ef已推送远端；trial绑定单轮/输出/代码，正式物理验收仍false。全流程实际完成、整轮速度和G能力结论尚未验证。入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/performance_b1_trial_0001_20261009/INDEX.md。

#### STTW-PERF-20261009 continuation and PPO A/B
- user_requirement: small floating discrepancy alone must not block optimization; old implementation is a comparator, not assumed ground truth. Preserve discrepancy evidence and prioritize method/physics/safety/done/reset contracts. No finding that old controller is incorrect.
- Local commitb473b0d adds opt-in batch reset guard. Hot512x128 summary+guard samples27.455516/27.382838/27.380626s vs summary median27.648829s, no robust gain claim. Its optional whole-rollout profiler was interrupted by conversation steering after hot samples; trace/final receipt incomplete, not called complete.
- Local commit8f84c33 PPO synchronization. Identical frozen rollout805154bc1434d97057737519e5cefbd9715383db5143acca8bd34db72be3a2fa: baseline median0.100336s, candidate0.069801s (30.4% PPO-only reduction, not overall training). Normal, temporal-weight-nonzero, NaN and hard-KL tests yielded bitwise-equal gradients/policy/Adam/RNG and identical metrics on this fixture. Focused30 tests and current combined116 regression tests passed.
- Full six-case evaluation A/B is running on original update100 checkpoints, benchmark root performance worktree/runs/performance/baseline_evaluation_100, PID2180010. Two warmups/three hot measurements and separate diagnostics, no formal training. Matched first two cases: upper0/upper1 checked5ms fields identical to original; B0 tiny differences (XY<=2.3e-5m), failure decisions identical. Full panel not yet complete.
- Isolated benchmark TensorBoard6015 restarted after interruption, HTTP reward verified; original experiment monitors untouched. Next: finish full evaluation/audits; batch evaluation A/B; cache/size sweeps and end-to-end report.


### JIT 一次性200轮接续事件

- repo_artifact/artifact_checked：独立1轮加速试点TensorBoard6035的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/performance_b1_trial_0001_20261009/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。


### JIT B1单轮试用完成与状态包核验（2026-10-09）

- reproduced：performance_b1_trial_0001_20261009完成1/1独立R31；PPO128000、G2000，wall862.43598秒，charged physics345618。历史原R31 wall3997.17227秒、physics332621；观测整轮墙钟减少78.42%，轨迹/数据/工作量不同，非严格等工作量速度证明。teacher worker274.3秒，G76.41秒。
- actual physics：32起点均有记录，31搜索batch、11独立重放；10 verified_solution、1 replay_rejected、20 searched_no_solution、1 not_scheduled。保持失败拒绝，无阈值放宽。
- artifact_checked：resolve_completed_boundary完整验证已发布R31状态包；TensorBoard6035当前奖励40点实际HTTP加载，末点累计learner transition3712000。integration_verified.json和integration_report.json保存。
- decision：保留优化代码及试用结果；不声称正式物理等价、重复可控加速或G能力提升。原200任务仍paused，不自动续训，不采用此支线checkpoint覆盖原R32。
- evidence：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/performance_b1_trial_0001_20261009/INDEX.md；远端报告提交d83c2a6。下一步在明确续训指令后锁定起点及新实现，不跨支线混合状态。


### JIT 授权B1续训阶段事件

- B1续训停止，无自动重试：RuntimeError("resource gate blocked: {'ready': False, 'reasons': ['shared GPU requires explicit authorization and memory margin'], 'phase': 'unknown'}")；/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0009_b1_resume_172/status.json


### JIT 用户授权恢复原200轮（2026-10-09）

- user_requirement：用户同意从原R32完整包恢复，先2轮核验，稳定后完成原200轮剩余部分。原已完成28轮，最多新增172轮；不将独立试点R31作为父模型。
- execution：冻结实现1a4d3ef；逐轮单轮范围凭证串行R33–R204，先R33/R34完整状态、PPO128000/G2000、teacher receipt、learner累计步数验收，再170轮。任何工程失败停止，无自动重试；不伪造物理等价。
- prepared/starting：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0009_b1_resume_172_retry01，监督2339166，TB6036计划从累计3840000步连续显示。首尝试资源授权token格式错误，训练前停止，修正新目录保留原失败。当前初始化/完整包核验中，尚不声称PPO已执行。


### JIT B1原200续训启动已核验（2026-10-09）

- artifact_checked：series_0009_b1_resume_172_retry01父监督2339166，R33子监督2342997已启动，顶层running、0/172新增轮、原累计28/200；首轮启动尚在初始化，不称PPO已完成。
- visibility：TensorBoard http://localhost:6036 HTTP200，watcher2340218心跳正常、无delivery_errors；新奖励尚未产生，文件事件观察器会在产生后核验并记录monitoring/reward_http_verified.json。
- continuation：首2轮完整校验后自动接余170轮，总计不超过原200；从原R32完整包恢复，未混用独立试点R31。入口：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0009_b1_resume_172_retry01/INDEX.md。


### JIT 授权B1续训阶段事件

- repo_artifact/artifact_checked：原200轮B1续训TensorBoard6036的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0009_b1_resume_172_retry01/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。


### JIT 授权B1续训阶段事件

- B1续训停止，无自动重试：AssertionError()；/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0009_b1_resume_172_retry01/status.json


### JIT R33续训监督误报修复（2026-10-09）

- root cause：series0009 retry01的R33已completed，PPO128000/G2000，wall741.26575秒；teacher实际19 roots匹配panel19。外层脚本错误assert roots==32，导致父状态failed且未计入完成；不是训练失败。旧父failed与日志保留，不伪造completed。
- fix：新series0010 launcher按actual panels长度及root_id集合核对teacher receipt/results，真实19起点通过，删一条结果明确拒绝。父起点为原链segment0033完整R33；累计29/200，剩171轮，不重跑R33。R34完成后与已验证R33构成两轮门槛，之后170轮。
- pending：完整包校验执行中，准备新续训/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0010_b1_resume_171；模型/训练核心实现1a4d3ef不改。


### JIT R34修复后恢复核验（2026-10-09）

- artifact_checked：R33完整包resolve_completed_boundary核验通过，learner lifetime3968000；series0010已启动R34，顶层running、原累计29/200、剩171轮。主监督2464159，watcher2465747心跳无错误，TensorBoard6036 HTTP200。新R34奖励尚待产生，事件观察器保留。
- repair boundary：仅修复launcher硬编码roots==32；实际panel长度与结果root_id全覆盖检查，真实19通过，缺一结果拒绝。旧失败父记录保持failed，R33 completed工件未改，未重跑已完成训练。已退出旧运行的三项monitor服务均按PID/命令路径核对，未动其他项目。
- evidence：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0010_b1_resume_171/INDEX.md；R34成功后通过原两轮门槛并进入余170轮，错误停止、无自动重试。


#### STTW-PERF-20261009 delivery and reduced testing (2026-10-09)

- user_requirement: 用户要求“不用测试这么多，请赶紧往后推进”。已核验并停止本任务剩余benchmark队列2452500及N256子任务2476597；原训练、其它项目进程未操作。receipt: performance-4090d/runs/performance/reduced_testing_receipt.json。其余规模扫描、更多重复及GPU temporal-mask试验不继续，未测项保留。
- reproduced: 完整update100六场景×三方法评估，2预热/3测量，实际每次39600个5ms active ticks；总墙钟median752.835059→239.373127s，约3.15x；物理median729.868574→222.482510s。18份奖励审计全部通过；报告含审计/绘图/文件输出时间。原baseline可选第六重复中断的错误状态及有效五次记录均保留。
- adoption: V5.1默认使用批量评估，batched=False保留顺序回退；非V5.1不变。维度修复、PPO同步、共享受信本地编译缓存启用；统计summary/reset守卫保持opt-in，收益不稳且严格连续状态对比失败。无物理/奖励/控制/预算修改，无正式训练；不推进JAX chunk或局部路径搜索。
- limitation: strict numeric equivalence失败；fast_turn alpha0最大XY差0.0211668m，10s内已出现，安全/终止/时钟一致。旧A/A亦有较小差异，不能据此证明原实现错误或新实现完全等价。用户允许小误差不阻塞推进；完整原始对比与反例保留。
- cache: 首次准备49.566802s、复用17.998257s；热环境median4.204583/4.181844s，不称物理内核加速。PPO同rollout median0.100336→0.069801s，原完整Adam/RNG/NaN/hard-KL证据保留。
- evidence: /home/qy/STTW_CONTROL/runs/worktrees/performance-4090d/docs/PERFORMANCE.md，内含六场景XY与fast_turn16s链接；raw runs/performance/batched_evaluation_100、summary_verified_smoke及各timing/trace。最后仅补双端点8×16工程检查adopted_end_to_end_smoke；TensorBoard6015当前两端点train/mean_step_reward HTTP已核验。无端到端A/B提速结论。
- delivery: 隔离performance/r196-v51-4090d；未push（AGENTS.override禁止自动push），未提交模型/缓存/trace。下一步仅回收工程检查状态、提交交付报告；不新增测试队列或正式训练。

- STTW-PERF final receipt: adopted_end_to_end_smoke completed5/5；每次两端点8×16、1024 active ticks，热样本9.786585/10.422120/10.246229s，median10.246229s（不含完整评估；无匹配旧端到端A/B，提速未测）。工程检查已结束，無待运行benchmark；最终本地提交7ab2cd8，未push。后续按用户明确实验指令使用已交付入口，不自动扩展测试或正式训练。


### JIT 授权B1续训阶段事件

- repo_artifact/artifact_checked：原200轮B1续训TensorBoard6036的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0010_b1_resume_171/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。


### JIT 授权B1续训阶段事件

- R33/R34两轮完整包、训练量、教师收据均通过。按用户已授权方案进入剩170轮；不增加原200轮总量。/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0010_b1_resume_171/two_round_gate.json

### STTW-V52-20261009 — minimal reward, physical best, lower tracking gate

- user_requirement: 执行STTW_V52_Codex.md及alpha0/alpha1 JSON；26c3fed真实源码先合入完成的4090D优化，保留R196 lower_alpha1、lower_reference_centered、ECBC/ESO、物理/时序/观测/动作/slew和两个独立端点。原授权修改后启动200新增更新、按证据100扩展最多400；后续在得知审计门槛后明确答复“按附件暂停训练”，覆盖自动启动。
- repo_artifact: 新隔离worktree /home/qy/STTW_CONTROL/runs/worktrees/v52-best-tracking，branch experiment/r196-v52-best-tracking；26c3fed相对77f715f仅证据/文档变化，先merge性能7ab2cd8→405cda1。奖励/配置d6b4660；best/计数/暂停交付4ed253d。主工作区用户修改、原父run、DVGC/JIT未修改；未push。
- artifact_checked: 父run /home/qy/STTW_CONTROL/runs/worktrees/r196-preference-v5-1-strong-primary/runs/preference_v51_strong_primary_100_20261009，两个对应last_completed为update100。现有update100_timeseries.npz18轨迹逐步e_task=(governed-raw)+(actual-governed)通过；零时间平移，新增仿真0。全程、[1,6)、[2,4)、governed平稳/过渡/恢复的bias/RMSE/P95/max/所有越限区间保留。wheel proxy差不是已验证真实滑移率。
- observed limitation: fast_turn B0下发[2.3,0]平稳后转角|e_lower|>.04rad持续[5.445,6.205)0.76s、[6.315,8.745)2.43s；alpha0下发约[2.374,.026–.028]时[9.020,10.240)1.22s。三段最终执行器/残差裁剪率0，姿态/航向仍在恢复。原日志无完整ESO/历史快照，具体内部原因未验证；不声称整个底层不可行，不换底层。门槛LOWER_TRACKING_LIMITED，用户明确暂停。
- parent physical selection: alpha0元组[0,1,0,3,9.03197057]，alpha1[0,1,0,5,11.17538901]；依次物理失败/工作界失败/主目标失败/10s联合保持失败/Q。两端qualified=false，主目标有效窗口通过并不等于整体达标。V5.2 best/last均不存在，父last不冒充新best。
- implementation: 仅指定peak_roll共同working项、alpha1单侧参考降速成本、普通/恢复速度deadband.01；分项cap合计5080及失败目标，无全局clip。准备Actor+std父100热启动、Critic/Adam新建、4value-only、每10保存、100/150/200及扩展每50数值验证；正式best安全优先、SHA加载、独立last和完整回合收敛扩展逻辑。工程运行未测，不声称集成完成或训练改善。
- verification: 7项必要CPU检查通过，独立公式/峰值/cap/failure、配置继承、误差区间、best安全顺序/partial拒绝/SHA、B0旧轨迹新奖励重评分；只读审查修复B0绘图重评分提前退出的问题。用户暂停前未启动工程8×16×2或正式训练；新PPO0、value-only0。
- evidence: /home/qy/STTW_CONTROL/runs/worktrees/v52-best-tracking/docs/V52_VALIDATION.md（首先六场景XY，best/last、达标、三层指令、全部误差入口）；runs/v52_parent_audit/lower_tracking_gate.json；planned run runs/preference_v52_tracking_best_20261009/status.json state=paused。无训练进程或新TensorBoard服务。
- next: 维持暂停；恢复授权后仅针对已定位恢复段解释/一次短复现，禁止批量搜索；若用户决定解除门槛，先补一次工程小批再按明确预算训练。

### STTW-V52-20261009 remote delivery

- user_requirement: 用户明确要求将当前V5.2审计结果上传远端；仅覆盖此次no-push限制，不解除训练暂停。
- artifact_checked: origin/experiment/r196-v52-best-tracking 已push并以git ls-remote核验为4486e0710ae0aa71b96e5f03ed1036320fb20bbe，与本地HEAD一致，工作区干净。
- evidence: docs/V52_VALIDATION.md全部相对链接核验；上传完整误差窗口/越限JSON、CSV、两端物理选择指标、暂停status和来源hash。复用已提交六场景XY与原update100 NPZ，无模型/缓存/重复大轨迹上传。
- report: https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/4486e0710ae0aa71b96e5f03ed1036320fb20bbe/docs/V52_VALIDATION.md 。训练仍paused，新物理和训练更新均0。


### JIT / Isaac 归档交付 SIM2SIM-DELIVERY-20261009

- user_requirement：上传此前双仿真器模型、详细参数、PPO实验数据及精简结果，仅数据与图、不含视频。本次不改变当前JIT路线或其他活动任务。
- repo_artifact / artifact_checked：上传MJX MJCF+mesh、PhysX原生USD+冻结控制代码、4988928/9904128源checkpoint、kd5两轮最终Actor/learner、完整配置、各407批标量、41067/70194条原始回合记录、双引擎XY/速度及训练奖励/loss/KL PNG。两轮各10002432步完成；warmstart单自然初态1.10s顶点/1.78s侧倾，fresh未起跳/3.18s停滞，不支持稳定落地或多seed成功率。
- boundary：旧PhysX1ms/MJX5ms与后续匹配5ms/20ms明确分开，XML默认1ms和实际运行覆盖值区分。knee外观修饰/末姿态回正未纳入结果；无视频、无重复PDF。本次无训练、无物理轨迹重跑。
- delivery：https://github.com/QaQaaa-zzz/DVGC/tree/09f1d1deb4aa3993341e163cc30ad30f8c505c5b/docs/evidence/sim2sim_ppo_20261009；branch reports/sim2sim-ppo-20261009；commit 09f1d1deb4aa3993341e163cc30ad30f8c505c5b。2026-10-09T19:08:45+08:00核实750个远端文件blob全部与本地一致，远端HEAD一致。回执：/home/qy/ISAAC——SIM/results/sim2sim_remote_delivery_20261009/upload_receipt.json。
- next：上传归档完成；后续只建议固定小面板隔离采样动作/训练布局差异，未经新指令不增加训练。

### STTW-V52-20261009 user resumes launch

- user_requirement: 用户明确“开启训练”，撤销此前暂停。保留LOWER_TRACKING_LIMITED已知证据，不声称问题已修复。执行此前未启动的一次8×16共2更新工程检查，通过后各4value-only，再每端新增200 PPO、按原附件改善条件100扩展、最多400新增；无额外墙钟截止。
- execution planned: /home/qy/STTW_CONTROL/runs/worktrees/v52-best-tracking，source4486e07；工程runs/v52_engineering_20261009；正式新目录runs/preference_v52_tracking_best_start_20261009（旧paused状态原样保留）。原父V5.1@100分别Actor/log_std热启动，Critic/Adam重置，原prepared bank复用。工程和正式独立计数；固定协议100/150/200及扩展每50数值验证，最终best完整图。
- resources: 单4090D，JAX预分配关闭；已有其他GPU进程不终止、不修改DVGC/JIT。完成后核验当前reward TensorBoard HTTP及PID/cwd/配置身份。

- STTW-V52 launch receipt: 工程runs/v52_engineering_20261009已完成两端各1更新，共256policy transitions/1024control ticks，退出0，两端stop=null。正式413f58c已启动PID2556760，cwd已核验为v52-best-tracking；run runs/preference_v52_tracking_best_start_20261009，初始200/端、最大400/端、4value-only/端。TensorBoard6016 HTTP200，当前正式reward尚待产生；工程reward与正式分开，不冒称已加载正式奖励。代码仅修复启动身份与进度字段，原优化/物理不变。旧paused记录保留。

- STTW-V52 verified running: PID2556760运行中；alpha0 value-only1/4完成、alpha1尚未开始warmup，PPO两端0/200。alpha0首批Actor bitwise unchanged、Actor梯度0、Critic更新finite；父两端checkpoint SHA分别c5679db90f2b6a90ea6e97ba6c4df7ca3986c01a9c4d3b0cea4da967469b7161 / 41448515dff7b0d8d44129a2ec7aaa39118db0044667246f7395901b38c1b946。prepared/lower身份匹配父run。TensorBoard http://localhost:6016 正式formal/alpha0 warmup/mean_step_reward实际HTTP加载，凭证run/reward_http_verified.json。启动的是Critic适应阶段，不称已完成PPO或解决底层问题。当前代码413f58c本地提交，未另push；旧4486e07审计远端保留。


### STTW-LOWER-INTERFACE-20261009 — bounded diagnosis and candidate preparation

- user_requirement: execute Codex_Lower_Interface_Repair.md; protect live V5.2, reuse lower_tracking_gate, at most B0 fast_turn12s once plus alpha0 once if needed. No formal lower training/resource reservation, no hot swap, no scans.
- artifact_checked: live PID2556760 cwd v52-best-tracking; source training413f58c, clean worktreeHEAD55f7706; both59/200 additional updates at initial inspection, frozen R196 SHA a2e5bf112e0cb986b4ec207fa9f20dcf90f48af6d7855b0abda32bfa13206672. Existing full internal diagnostic fields absent.
- execution planned: isolated repair/r196-local-lower-interface at runs/worktrees/lower-interface-repair from55f7706. Diagnostic source patch/identity saved before launch; bank index3, env77001/episode0, fixed fast_turn, slew .5/.3, float32/MJX, up to2400ticks per replay; no learner updates. Acceptance is causal chain measurement plus explicit replay agreement/limitations, not improved performance. Stop at12s or truefailure.
- candidate planned: reuse210D lower_command interface and trainer; freshweights only, late fixed physical validation, separate train_candidate/last/best, finite component-bound reward. Candidate alias reserved STTW_LOCAL_CMD_V1 but not registered/adopted. Training not_run.


## JIT / Isaac：JUMP-LANDING-V2-P0P1-20261009

source_type=user_requirement + repo_artifact；verification=reproduced（以下限定开发对照）；recorded_at=2026-10-09。仅更新本任务，不刷新其他活动实验状态。

- 用户任务书 `/home/qy/下载/CODEX_SIM2SIM_JUMP_LANDING_V2_20261009.md`：先P0/P1，再门控E1→E2→E3；历史证据不可覆盖，不直接1000万步。隔离工作树 `/home/qy/ISAAC——SIM/worktrees/jump-landing-v2`，分支 `experiment/sim2sim-jump-landing-v2`，本地提交 `df6952a`；无push。主报告 `sim2sim_v2/REPORT.md`，原始数据保留该目录runs/。
- P0配置溯源/冲突拒绝已验证，legacy不改变历史物理。源9904128有sim2sim DR/低速失败包装，历史PhysX不完整一致；并未发现历史warm奖励声明与有效奖励不符。两端本次5ms物理/20ms控制，PhysX rear kd5。
- exact9904128名义物理MJX自然初态：takeoff0.66/apex0.98/touchdown1.34s，最长严格保持0.42s，0.5/1s均失败，8s timeout而非侧倾失败。直接读取Warp solver contacts；失败host转换及SIGSEGV日志保留，后者精确预算UNKNOWN。
- 更正“这轮续训没有稳定落地能力”的笼统结论：warm中间02457600与04915200各一个自然开发初态通过严格1s，连续保持5.80/6.26s；平均vx0.629/0.317m/s，非2m/s跟踪或泛化证明。final确定性仍1.78s侧倾失败；随机动作三种子timeout但慢速且严格保持不足0.5s。
- P1 gate HOLD：同动作同origin384→1首次接触80ms出现小物理差并扩散；完整episode后普通reset显式初态相同但轨迹不等价，source80ms/warm820ms状态首次分歧。接触缓存/求解顺序机制尚未因果隔离。动作导出最大误差6.184e-6通过。
- 真实384×64两份rollout离线KL分解发现fresh的normalizer变化和SGD都可致巨大偏移；新门控真实batch硬拒绝后参数/完整Adam/normalizer逐位回滚。26项测试通过。E1配置/launcher准备完成但正式训练未运行，E2/E3未运行，所有持久接受更新0。
- 实际预算：PhysX53363控制/213452物理；MJX成功400控制/1600物理，首次失败4物理，SIGSEGV至少4最多1600物理。全部总物理下界215060，上界216656；不是零采样。无新1000万训练，未改历史docs/evidence。
- execution_status=completed（本轮有界诊断）；evaluation_status=partial（P1 gate未通过）；adoption_status=pending。中间候选非全局best，未替换源初始化。
- 唯一下一步：冻结learner隔离PhysX布局/reset接触后差异，补实际solver属性读回，判断P1 gate；通过后再执行491520步E1，E2/E3顺序推进。摩擦/映射/输入扩展保持独立未运行轴。

- STTW-LOWER-INTERFACE progress: candidate preparation committed63c9d68 in isolated repair/r196-local-lower-interface; 6 focused new +5 directly related legacy CPU tests pass. No formal lower training/installation/push. Late fixed physical best100/150/200, finite component bound4140.18,210D fresh policy, screened command slews, original PPO rollback+V5.2 rejection handling implemented but GPU candidate validation not_run.
- artifact_checked original saved control chain: ECBC offline output max error B0 1.915e-6 and alpha0 1.639e-6rad/s. In B0 second interval and alpha0 interval, roll contribution opposes and exceeds wheel-angle contribution; NN residual is correcting the angle every tick, final/actual rate closely match. Old path/return inputs and path-heading discontinuities observed/reconstructed, independent causal contribution UNKNOWN. Sole B0 diagnostic PID2806973 currently 1000/2400ticks; full internal/force result PENDING, no alpha0 replay. docs/LOWER_INTERFACE_REPAIR.md draft plus source/plots in docs/evidence/lower_interface_20261009.
- live V5.2 read-only progress during this task: {"0": 85, "1": 84}/200 per endpoint, state=running, PID2556760; no active-run edits or restart.


### STTW lower-interface repair：B0诊断完成及用户暂停轮询（2026-10-09）
- source_type: repo_artifact / user_requirement；verification: artifact_checked。隔离工作树 `runs/worktrees/lower-interface-repair` 的唯一 B0 fast_turn 12s 重放完成：2400控制tick、60000物理子步，无失败；alpha0未新增物理重放。原始证据 `runs/lower_interface_20261009/B0/`，报告 `docs/LOWER_INTERFACE_REPAIR.md`。
- 原生转向执行器力矩触界全程36/60000子步；首个失配区间22/3800，第二个区间0/12150；不能声称完全无饱和，也不能将短暂触界解释为全部持续角度滞后的原因。原生记录确认5.125s及5.930s路径选段跳变。重放与原轨迹有微小数值差异，不宣称bitwise相同；alpha0力矩状态未测。
- 候选代码本地提交 `63c9d68`、`6e47341`，正式候选训练未启动、未安装到R196或活跃上层；诊断及报告尚有未提交修改，未push。
- V5.2最后一次只读检查为新增alpha0/1各100/200、stage=evaluate100（该历史快照不冒充持续实时状态）；本轮未修改或停止该运行。
- 用户要求不再轮询、结束后由用户通知。已成功调用GNOME终端打开独立V5.2 stdout日志窗口，关闭窗口不停止训练；助手停止轮询，等待用户下一条指令。


## JIT / Isaac：DUAL-LANDING-NO-SPEED-20261009

source_type=user_requirement + repo_artifact；verification=reproduced；recorded_at=2026-10-09。用户最新目标：只需同一个模型两边都稳定落地，不要求保持速度；此条更新本任务优先级，历史P0/P1门槛与结果不删除。

- 当前工程候选：PhysX历史warmstart `transition_01966080`（新增1,966,080步，父源9904128）。文件 `/home/qy/ISAAC——SIM/runs/experiments/physx_jump_ori_from9904128_kd5_5ms_10m_20260928/training/checkpoints/transition_01966080/actor.npz`，两边SHA256均 `d2648c4f33b3e191293d2a4281d9a95f475f2a2be09c14a0ad8f605235c2d1a0`。策略/normalizer冻结，无后轮输出替换。
- 两边名义自然初态root[1.5,0,.15]与四元数一致，sim5ms/control20ms。MJX显式关闭训练DR及低速失败包装作为新验收的评估override，源代码/历史训练不改。PhysX既有kd5。最低前进速度取消；本候选保留原侧滑约束仍通过。
- MJX takeoff.74/apex1.0/touchdown1.30s，2.68s达1s稳定保持，最长1.10s；PhysX takeoff.98/apex1.24/touchdown1.54s，2.80s达1s稳定保持，最长5.68s。两边回放到8s，无侧倾终止，随后均驶过平台后缘回到地面。成功来自真实接触/姿态/角速度时序，非timeout或reward。
- 仅各一个固定开发自然初态，非独立holdout/多初态成功率；从历史checkpoint筛查所得候选，不称全局best。27项测试通过。主报告与XY/速度/接触图、CSV、轨迹、SHA核对位于 `/home/qy/ISAAC——SIM/worktrees/jump-landing-v2/sim2sim_v2/runs/no_speed_dual_landing/REPORT.md`。本地提交 `0b2c8cb`，未push。
- 此轮实际MJX2571控制/10284物理，PhysX配对410控制/1640物理，P1固定动作history诊断96控制/384物理；合计3077控制/12308物理，接受训练更新0。原声明n2/n384history诊断未运行，优先完成用户双引擎目标。E1/E2/E3新训练均未运行。
- P1剩余问题没有冒称修复：n1固定动作hard重建32ticks逐位重复，soft reset首次接触80ms分歧，0.64s内root最大差.0123m。求解器属性读回16/4,TGS,enhanceddeterminism=true,stabilization=false；内部缓存未隔离。
- execution_status=completed（本轮配对验证）；evaluation_status=complete（单开发初态范围）；adoption_status=pending（工程展示候选，非论文泛化采用）。当前不启动新PPO；若进一步需要鲁棒性证据，再声明小扰动自然初态配对面板。


### DUAL-LANDING-01966080 视频与模型保存（2026-10-09）

source_type=user_requirement + repo_artifact；verification=artifact_checked（原轨迹原生渲染，非新动力学试验）。用户要求保存视频和模型，已保存至 `/home/qy/ISAAC——SIM/worktrees/jump-landing-v2/sim2sim_v2/runs/deliveries/dual_landing_01966080_20261009/INDEX.md`。

- 完整checkpoint五文件（Actor含normalizer、learner state、配置、身份、export检查）、两端物理模型资产、原始配对轨迹与接触数据、XY/速度/姿态图、CSV及68文件哈希清单均保存。ActorSHA保持 `d2648c4f33b3e191293d2a4281d9a95f475f2a2be09c14a0ad8f605235c2d1a0`。
- videos/mjx_realtime.mp4、physx_realtime.mp4与paired_realtime.mp4均400帧、50fps、8秒，经ffprobe验证；前者原MJX-Warp状态用MuJoCo EGL渲染，后者原PhysX状态用Isaac RTX渲染，无姿态修饰。渲染physics steps0，PhysXroot误差<=4.77e-7m、关节误差0。模型未训练/更新。
- 固定单自然初态的工程成功范围不变，非多初态鲁棒性。目录含INDEX、manifest、render verification和日志；历史证据未覆盖。
- 实现本地提交0740965，无push。execution_status=completed；evaluation_status=complete（保存/渲染核对）；adoption_status=工程候选保存，论文泛化未验证。


### JIT 授权B1续训阶段事件

- B1续训停止，无自动重试：RuntimeError('round 62 exited 1; no automatic retry')；/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0010_b1_resume_171/status.json


### STTW V5.2 training completed：2026-10-10只读核验
- source_type=repo_artifact；verification=artifact_checked。运行 `runs/worktrees/v52-best-tracking/runs/preference_v52_tracking_best_start_20261009` 的status为complete/best_evaluated；PID2556760已不存在。每端新增300 PPO更新、4 value-only rollout，父模型100更新单独计数，lineage400；本次未新增训练或仿真。
- best与last不同：alpha0 best新增200（SHA6afd105630cd7c7a02715f0c234eb39071c2c8800790a900dc02b248c23a59ae）；alpha1 best新增250（SHAfae95ab4634b021ad36be4e692493784fbbb123adcdfb20c2dd288553f801222）。两端last均300。实际checkpoint SHA复核通过，final_best_loaded记录匹配，evaluationbest完整图与数值产物存在。
- 已评估候选best均qualified=false。alpha0物理失败0/6、工作侧倾门槛失败0/6、主目标失败1/6、10秒联合保持失败6/6；alpha1依次0/6、1/6、0/6、1/6。fast_turn的alpha0/1峰值侧倾0.300326/0.341544rad；alpha0工作门槛通过包含既定数值容差，不能称严格不超过0.30。两端fast_turn10秒保持失败、16秒保持通过，延伸不能覆盖主窗口失败。
- 两端decision_0300均no_supported_extension、converged=false；记录的training_gain为-0.01545/-0.09591，validation_gain为+0.19898/+0.37741，不能称所有指标持续共同改善或已收敛。本次未续到400。
- 下一步：解释最终best的逐场景主动让步/底层跟踪分解及未达标原因，继续保持冻结R196；不因本次状态查询启动候选训练。最终XY与每步奖励入口：该运行evaluationbest/INDEX.md。本次仅结果读取和共享台账同步，无源码修改、commit或push。


### JIT R62教师重放复合冲突诊断（2026-10-10）

- user_requirement：再次报错，查看原因；本次只读取诊断，未改协议或启动重试。
- artifact_checked：series0010累计完成28个新增轮R34–R61，原任务57/200。最后完整R61；R62在teacher阶段失败，学生/G更新未开始，本轮supervised_updates0。
- direct cause：teacher0007选择candidate2，搜索label1，重放label0；source candidate0同时从0变1，candidate7也0变1。17lane完整重放completed，所有search/replay labels均为0/1。production.py的quarantine_source_conflict仅允许selected replay label1；当前selected0导致不进入隔离，后续source_control_repeat_conflict分支直接raise unknown or failed teacher/control repeat。因此日志将完整物理失败重放归到incomplete分支。
- uncertainty：真实物理标签差异根因仍UNKNOWN，不能仅凭此认定常驻scratch污染或原仿真错误；没有OOM/显存门控失败证据。此前32root硬编码问题未复发。
- evidence：series_0010_b1_resume_171/segment_0062/round_0062/teacher_worker/teachers/0007_layout_repeat.json，以及execution/teacher_worker/teacher_worker.log。
- next：补充source翻转且selected失败的组合分支回归，明确隔离/拒绝语义；不得将candidate2改标成功或跳过独立重放。修复后续接仅以完整R61包为父，R62失败费用保留。


### STTW V5.2最终结果远端交付准备（2026-10-10）
- 用户明确授权当前结果打包并push；无新增训练授权。V5.2分支experiment/r196-v52-best-tracking本地提交2eeef42，报告docs/V52_VALIDATION.md更新为完成后的真实结果，保留启动前历史。docs/evidence/v52_final_20261010含35PNG、18原始5ms NPZ、两端best/last与晚期评价/决策、回合窗口及训练标量摘要，共约24MB；不提交权重、缓存、大内部trace或重复中间轨迹。checkpoint SHA与最终reload身份核对通过，JSON解析及git diff --check通过。push正在执行，成功与否以随后receipt为准。
- 底层修复复核：隔离分支repair/r196-local-lower-interface候选代码提交63c9d68/6e47341；已有必要CPU检查记录11 passed。唯一B0 12s诊断完成；ECBC姿态/轮角贡献竞争、R196残差抵消和旧path/return接口任务差异有证据，独立因果份额未隔离。新210维局部候选未训练、未做GPU工程批、未注册或部署；诊断报告和相关诊断源码尚未提交。不能声称底层跟踪修复已验证成功。本次只交付V5.2已完成结果，不将候选偷偷安装或启动200更新。


STTW V5.2交付receipt（2026-10-10，repo_artifact/artifact_checked）：push成功；git ls-remote核验origin/experiment/r196-v52-best-tracking = 2eeef426367e8922cf4fd334ce4989fc6fcc0515，本地工作树clean。远端报告 https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/experiment/r196-v52-best-tracking/docs/V52_VALIDATION.md 。本次没有推送底层候选分支，没有新增训练。


### JIT R62复合重放冲突修复与十轮通知（2026-10-10）

- user_requirement：修复后继续；正常弹窗改为每10轮一次。
- implementation：5b3f376允许source翻转且selected replay label0的完整有限结果隔离；保留0、不作新收益/成功demo，未知/不完整仍停止。相关252回归通过；通知7测试通过（十轮去重、错误/最终完成保留、子轮复用父watcher）。
- notifications：新计划notification_owner复用父监视器，取消子轮启动/完成popup。父按原campaign累计60/70/…每10轮一次，错误立即、最终完成另行。
- execution preparing：series0011_b1_resume143，监督1011565；从失败segment0062解析完整R61和已消耗成本，累计57/200剩143，不篡改父failed。冻结新快照code_replay_conflict_fix，新的奖励横轴从7552000步继承。当前尚在初始化，未宣称R62训练完成。
- evidence：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/INDEX.md。


### STTW底层诊断远端交付完成（2026-10-10）
- source_type=user_requirement + repo_artifact；verification=artifact_checked。用户明确要求底层诊断一并推送；独立分支origin/repair/r196-local-lower-interface已推送，git ls-remote核验HEAD=4e956f010c69f2d559cc8c839b0dac74c291e9f2；本地工作树clean。
- 提交包括docs/LOWER_INTERFACE_REPAIR.md、5张XY/控制链/内部状态PNG、紧凑数值证据、可选诊断采集与重建工具、必要测试；此前候选实现63c9d68/6e47341随分支保留。11项必要CPU测试本次重跑通过（2.95s），证据JSON解析和git diff --check通过。原始大型内部trace、状态pickle、模型不提交；报告明确远端包不含全部独立重放依赖。
- 候选训练/GPU工程批/旧失配完整状态对照仍not_run，未部署、未改R196。没有新增仿真或训练，没有更改其他项目；V5.2最终结果仍在独立experiment/r196-v52-best-tracking分支2eeef42。
- 远端入口：https://github.com/QaQaaa-zzz/STTW_CONTROL/blob/repair/r196-local-lower-interface/docs/LOWER_INTERFACE_REPAIR.md 。下一步如需候选训练，应另行确认正式资源与预算，本次push不构成训练授权。


### JIT 修复与十轮通知已生效（2026-10-10）

- artifact_checked：series0011顶层running R62，恢复完整R61，原57/200，剩143；计划notification_owner指向总任务watcher，ACTIVE_RUN progress_interval10/progress_initial57。watcher1014335心跳正常无delivery_errors，TensorBoard6036 HTTP200；新奖励尚待产生后自动核验。代码5b3f376已push。无每轮启动/结束popup；60/70/…累计轮次通知，错误和最终完成保留。


### JIT 授权B1续训阶段事件

- repo_artifact/artifact_checked：原200轮B1续训TensorBoard6036的首轮episode/sum_reward已实际加载；凭证：/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/monitoring/reward_http_verified.json。这是训练可见性核验，不是能力改善结论。

### STTW-LOWER-NOALPHA-20261010 — authorized lower-first training

- user_requirement: Execute Codex_Lower_First_NoAlpha.md with lower_local_tracking_no_alpha_v2.json. Lower source4e956f0; frozen upper2eeef42, best0 added200/best1 added250 protected. No upper training/reward changes/merge/distillation/push.
- owner: current Codex session; isolated experiment/lower-first-noalpha at /home/qy/STTW_CONTROL/runs/worktrees/lower-first-noalpha. Reuse lower_command and lower_command_training; fresh210/211 ELU256→128, ECBC/ESO/plant/action limits unchanged.
- implementation: five actual gaps addressed; targeted5 CPU tests pass after expected failing reproductions. Full rollout-boundary state/RNG/Actor/Critic/Adam serialization and explicit --resume-checkpoint/--target-updates entry. Same immutable published5ms stream for local/R196 fixed7 cases.
- budget: one engineering8env×16steps×2 updates, then separate fresh512×256×200=26,214,400 control transitions; extensions only100 blocks with evidence, max400=52,428,800. Fixed validation100/150/200, extension250/300/350/400; no20/25. No wall cutoff. Single formal lower; unrelated jobs protected.
- run directories: runs/noalpha_v2_engineering_20261010 and runs/noalpha_v2_formal_20261010 under new worktree. Execution planned; evaluation not_started; adoption pending. Local protocol qualification required before one original6.315s full-state mismatch recovery; both gates required before frozen upper combinations. Configuration and published stream hashes, prepared bank/model identities recorded by run manifest.

- STTW-LOWER-NOALPHA launch verified: engineering8×16×2 completed, finite gradients and actor delta 0.06953679025173187; formal PID1129270 cwd /home/qy/STTW_CONTROL/runs/worktrees/lower-first-noalpha, fresh initialization. Current 1/200 updates, 131072/26,214,400 transitions, state=running. TensorBoard PID1129271 http://localhost:6006 formal/. train/mean_step_reward actual scalar HTTP loaded, receipt /home/qy/STTW_CONTROL/runs/worktrees/lower-first-noalpha/runs/noalpha_v2_formal_20261010/reward_http_verified.json. First fixed validation100 pending.

- STTW-LOWER-NOALPHA targeted correction: review identified accepted counter absent for non-temporal lower PPO and stale-resume overwrite gap. Formal PID1129270 requested own SIGTERM boundary stop at update6/786432transitions; full state saved. Commit9a2c66e,7 new focused tests pass, no repeated engineering batch. Exact resumed PID1146166 from boundary6 in same run with no sample repetition. Accepted count reconstructed6 from original accepted_epochs; correction receipt preserved; original scratch manifest/checkpoints unchanged. Resume preflight now rejects stale/non-owned boundaries and out-of-cap/backward targets before writes. No upper/JIT edits, no push.

- STTW-LOWER-NOALPHA user steering: 用户要求停止Codex轮询，训练结束后用户再叫回收结果。后台训练PID1146166继续；当前训练日志快照18/200，2359296/26214400。桌面CMD终端tail实时输出已打开（gnome-terminal退出0），/home/qy/STTW_CONTROL/runs/worktrees/lower-first-noalpha/runs/noalpha_v2_formal_20261010/view_progress.sh；一次训练曲线快照training_progress.png，TensorBoard6006 formal/.。固定验证100/150/200保留；200后停止，用户叫回前不自行扩展、旧失配回放或组合实验。后续按原授权和证据门槛判断。此为运行交接，不是训练/控制验收完成。


### JIT-AUDIT-20261010-HISTORY — 实验、结果与过程图片对应总索引

- user_requirement：汇总之前做过哪些实验、结果和过程图，写入唯一台账，证据匹配。本轮仅JIT；STTW内容保持原样。使用报告技能的证据分级方法；按用户要求直接维护既有Markdown台账，不创建另一份权威报告或网站。
- scope/count：本对话主线及当前桥接相关 **15组研究问题/实验族**（含纯评价与工程试点），不是15个独立训练种子或15次GPU运行。恢复目录、重试、检查点不重复计数；S04随机臂只排队，不能算已做。8月所有Phase D/Tube、9月所有奖励小消融及全部旁线未穷尽，history_coverage仍PARTIAL。
- verification：repo_artifact / artifact_checked（具体粒度见每行）。本次未训练、未重放物理、未修改JIT代码；历史聚合率明确来自既有核验报告，未假称本轮重新计算全部回合。旧输出与源码快照内附带的历史图片不混入新实验过程图。
- 当前状态更正：9月旧探索任务仍96轮；10月桥接是后继不同协议。3.2、H006的recovery_0002“未进入学生”仅为历史快照；recovery_0004已生成A/B/C训练评价，但流水线最终失败、三候选均未正式采用。当前连续学习的实验采用不覆盖该历史拒绝结果。

| 稳定索引 | 实验/问题 | 工况与执行范围 | 结果与边界 | 对应证据及本轮核验级别 |
|---|---|---|---|---|
| JIT-S01 | 固定π0：学习探索 vs 均匀随机 | 两臂各3072候选；不补训π0 | 成功2126/3072 vs 2078/3072（69.21% vs 67.64%）；成功到达格子1386 vs 1385。覆盖优势未建立；实际交互337188 vs347736，不是严格等实际成本。 | [机制报告](../../DVGC/JIT/runs/experiments/exploration_paper_20260915/analysis/mechanism_review/INDEX.md)；报告及派生汇总核对；未重跑 |
| JIT-S02 | repair10起点：学习探索100轮 | 102400候选；已完成，最终0098 | 3653次局部失败转成功，其中1096在采用轮次。最终配对复测623/1024，起点643/1024；第0步25.78%→16.80%。未证明整体提高。 | [最终复测](../../DVGC/JIT/runs/monitoring/lineage100_diagnosis_20260919/REPORT.md) / [轮次汇总](../../DVGC/JIT/runs/monitoring/results_audit_20260928/summary.json)；报告及派生汇总核对；未重跑 |
| JIT-S03 | repair10起点：随机探索100轮 | 训练完成，最终0099；补评4起扰点×256回合 | **最终成功566/1024=55.27%；训练前643/1024=62.79%，学习探索最终623/1024=60.84%。随机训练后分别低7.52、5.57个百分点，本面板未改善且低于学习探索。** | [本次配对结果](../../DVGC/JIT/runs/monitoring/random100_final_comparison_20261010/INDEX.md) / [原始汇总及配对检查](../../DVGC/JIT/runs/monitoring/random100_final_comparison_20261010/summary.json)；随机0099新增实测，另两模型同种子/同批量历史轨迹复用；非跨训练种子结论 |
| JIT-S04 | 原始Phase U起点：学习/随机100轮对照 | transition4988928；学习96/100，随机未启动 | 当前源0093；98304候选、10574局部转化，7407在被拒绝轮次。尚无两组最终配对结果。 | [学习当前源](../../DVGC/JIT/runs/experiments/phase_u_discovery100_20260920/nan_recovery_20260926/lineage/current_source.json) / [随机恢复计划](../../DVGC/JIT/runs/experiments/phase_u_random100_20260920/queue_recovery_20260926/INDEX.md)；原始状态/目录核对；数值来自已核对96轮汇总 |
| JIT-S05 | Fresh RSI约100万步 | 1368快照，20%固定起点/80%RSI，从零训练 | 998400步；快检0/100，对照78/100；84回合在扰动前结束。不能算已学会完整起跳。 | [运行与核验入口](../../DVGC/JIT/runs/experiments/fresh_rsi_comparison_20260918/INDEX.md) / [汇总依据](../../DVGC/JIT/runs/monitoring/results_audit_20260928/REPORT.md)；报告及派生汇总核对；未重跑 |
| JIT-S06 | Fresh RSI 800万步 | 同类快照重置、从零训练 | 800万步完成；固定初态较晚扰动接近100%，随机初速度面板平均57.48%，早期仍弱。和1M快检种子不同。 | [训练入口](../../DVGC/JIT/runs/experiments/fresh_rsi_phase_u_reward_8m_20260918/INDEX.md) / [大样本评价](../../DVGC/JIT/runs/experiments/five_policy_random_initial_pulse_20260921/attempt_0002/analysis/INDEX.md)；报告及派生汇总核对；未重跑 |
| JIT-S07 | 三来源各追加约1500万步Phase U PPO | 三臂各14991360步；七策略×4起扰点×1000=28000评价回合 | RSI第10/15步100%/100%→45.1%/26.9%；repair70第0步12.5%→52.3%，第10步88.2%→10.9%。阶段取舍和回退，非普遍提高。 | [完整恢复评价](../../DVGC/JIT/runs/experiments/seven_policy_phase_u_15m_pulse_20260918/recovery/attempt_0001/reports/report_0000/INDEX.md)；报告及派生汇总核对；未重跑 |
| JIT-S08 | Phase U＋RSI融合学生 | 双教师同状态监督，单学生；320万PPO步 | 开发best在累计256万步；新种子快检463/500；继续到320万步明显退化。随机初速度250k协议中68.55%。不同协议不混比。 | [快检](../../DVGC/JIT/runs/experiments/phase_u_rsi_fusion_20260920/quick_check_100/INDEX.md) / [开发原始数据](../../DVGC/JIT/runs/experiments/phase_u_rsi_fusion_20260920/development_results.json)；报告及派生汇总核对；未重跑 |
| JIT-S09 | 固定初态四策略随机脉冲 | 4策略×5起扰点×10000=200000回合；评价不是新训练 | 五点等权平均：Phase U75.45%、RSI73.07%、repair10 68.80%、repair70 65.83%；各模型离地时刻不同，不是相位完全对齐。 | [评价索引](../../DVGC/JIT/runs/experiments/four_policy_five_onsets_10k_20260920/INDEX.md) / [相位/包线](../../DVGC/JIT/runs/experiments/four_policy_five_onsets_10k_20260920/analysis/envelope_timing/INDEX.md)；报告及派生汇总核对；未重跑 |
| JIT-S10 | 随机初速度五策略随机脉冲 | 5策略×5起扰点×10000=250000回合；评价 | 平均Phase U87.62%、融合68.55%、RSI57.48%、repair10 41.57%、repair70 29.73%。不含最新0098/0099/0093三个最终模型。 | [评价及图](../../DVGC/JIT/runs/experiments/five_policy_random_initial_pulse_20260921/attempt_0002/analysis/INDEX.md)；报告及派生汇总核对；未重跑 |
| JIT-S11 | 新Phase U speed2从零1000万步 | 完成10002432训练步；最后checkpoint八个种子评价 | 8/8达到顶点，8/8存活到400步且均timeout；无物理失败。不能把Phase U顶点/存活指标改称落地稳定恢复成功率。jump_ori是环境奖励别名，不选Actor。 | [训练状态](../../DVGC/JIT/runs/phase_u/phase_u_v4_speed2_denseckpt_deferred_10002432_seed820701_20260928/status.json) / [8回合评价](../../DVGC/JIT/runs/phase_u/phase_u_v4_speed2_denseckpt_deferred_10002432_seed820701_20260928/evaluations/transition_10002432/summary.json)；原始JSON核对；别名manifest已读 |
| JIT-S12 | 桥接v1.1：pilot＋追加一轮 | 两个学生各128000步，均从0093独立初始化；G继承 | 老师19/19、21/21；学生吸收7/19、10/21；两个学生均拒绝。core旧成功丢9、8；protected保持54/57、50/57。不能写一个学生累计学会17个。 | [结构化历史报告](../../DVGC/runs/worktrees/generative-bridge-results/JIT/docs/generative_bridge/CORE_RESULTS.md) / [pilot采用记录](../../DVGC/JIT/runs/experiments/generative_bridge_v1_1_20260928/recovery_0002/actor_acceptance.json)；采用原始JSON与报告文本核对；老师逐根未重放 |
| JIT-S13 | 桥接v1.2 A/B/C学生消融 | A=PPO+保持，B=教案+PPO+保持，C再加BC预热；每臂128000步 | 三个学生均有评价且未采用：core丢20/30/9，protected丢16/27/21；25个老师验证解上学生成功21/25、25/25、22/25。62根矩阵含33个unknown/incomplete，不能把25作为全部根。流水线最终failed/FileNotFoundError；四组合重放冲突隔离，不称全流程正常完成。 | [采用原始记录](../../DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/campaign/acceptance_report.json) / [老师学生矩阵](../../DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/campaign/result_matrix.json) / [流水线状态](../../DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/campaign/status.json)；原始JSON核对 |
| JIT-S14 | 邻域探索＋生成桥接＋连续学生 | 四起扰0/5/10/15，每轮128候选；PPO128k/G2000/E更新；2轮试点后授权200轮 | 本次快照原任务63/200、R67完整、R68运行。R67固定9条件DEV：P0 186/288→学生201/288，新增68/丢53；相对初始C 228/288→201/288，新增40/丢67。adopted=true只是允许连续训练，formal_adopted=false。 | [实时任务](../../DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/status.json) / [R67配对评价](../../DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/segment_0067/round_0067/generalization_comparison.json) / [采用语义](../../DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/segment_0067/round_0067/acceptance.json)；原始JSON及监督进程核对；运行中快照非最终结果 |
| JIT-S15 | 常驻B1加速独立试点 | 1轮R31；相同声明PPO128k/G2000/32roots/17候选 | 完成，862.44秒 vs历史3997.17秒（观察墙钟少78.4%）；345618 vs332621计费交互，31搜索/11重放/10验证解/1拒绝。非等工作量物理配对，不能证明正式等价或学习提升。 | [完整性能收据](../../DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/performance_b1_trial_0001_20261009/integration_report.json) / [试点索引](../../DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/performance_b1_trial_0001_20261009/INDEX.md)；原始JSON核对 |

**工况对应规则。** S09/S10均为控制步0.02s、3步四通道归一化动作脉冲、起扰0/5/10/15/25、最长400步，稳定恢复窗口0.5s。S10另加初始速度/角速度噪声；S09没有。S11是Phase U顶点与存活指标，不与稳定落地成功混用。S14采集起扰0/5/10/15，但DEV为幅度0.25/0.4/0.6×起扰0/10/20×32回合=288，seed8100801；这是反复使用的开发面板，不是封存TEST。S15只是工程性能试点。

**R67细分DEV证据（每格32回合）。** 起扰顺序0/10/20：幅度0.25下P0成功32/16/24，学生30/19/20；0.4下P0 28/12/24，学生21/21/26；0.6下P0 25/5/20，学生15/23/26。早期t0均退化、部分较晚时刻改善，不能用总成功净增15掩盖53个旧成功丢失。相对初始C总成功下降27。因无等总成本随机对照与最终TEST，不构成探索器或G独立必要性证明。

**过程图片索引（每图绑定实验及用途）。** 已核对以下路径存在；只链接既有图，不生成来源不明的宣传图。原始数值和NPZ仍由各实验索引管理。

| 实验 | 过程图/索引 | 解释限制 |
|---|---|---|
| S01 | [学习/随机成功率与覆盖](../../DVGC/JIT/runs/experiments/exploration_paper_20260915/analysis/mechanism_review/comparison.png) | 固定π0机制TRAIN；不是最终策略提升 |
| S02 | [离地点对齐跳跃轨迹](../../DVGC/JIT/runs/experiments/neighborhood_reward005_safe256_20260916/idle_resume_100/analysis/liftoff_aligned_20260919/liftoff_aligned_full.png) | 旧repair10起点100轮；不是原始Phase U新组 |
| S04 | [探索器奖励随轮次](../../DVGC/JIT/runs/monitoring/results_audit_20260928/phase_u_training/reward.png) | 仅完成96轮，1024候选奖励和，含5轮均值 |
| S04 | [探索器成功与采用](../../DVGC/JIT/runs/monitoring/results_audit_20260928/phase_u_training/success_and_discovery.png) | 局部补训成功含未采用模型，不是最终测试 |
| S04 | [探索器KL/熵/loss](../../DVGC/JIT/runs/monitoring/results_audit_20260928/phase_u_training/optimization.png) | 不是跳跃策略PPO loss |
| S07 | [七策略过程/对照图索引](../../DVGC/JIT/runs/experiments/seven_policy_phase_u_15m_pulse_20260918/recovery/attempt_0001/reports/reference_style_20260920/INDEX.md) | 三组约15M续训后的特定评价协议 |
| S08 | [融合开发曲线及历史训练诊断](../../DVGC/JIT/runs/monitoring/results_audit_20260928/training_evidence.png) | 跨方法预算不同；融合best/last开发曲线 |
| S09 | [固定初态相位与成功率](../../DVGC/JIT/runs/experiments/four_policy_five_onsets_10k_20260920/analysis/envelope_timing/success_and_phase.png) | 每点每模型10000；相同控制步不等于相同相位 |
| S09 | [五个起扰点包线](../../DVGC/JIT/runs/experiments/four_policy_five_onsets_10k_20260920/analysis/envelope_timing/envelope_all_onsets.png) | xz经验轨迹带，非安全认证 |
| S10 | [随机初速度五策略比较](../../DVGC/JIT/runs/experiments/five_policy_random_initial_pulse_20260921/attempt_0002/analysis/network_comparison.png) | 250000回合；不能套用固定初态解释 |
| S11 | [新10M训练曲线](../../DVGC/JIT/runs/phase_u/phase_u_v4_speed2_denseckpt_deferred_10002432_seed820701_20260928/training_curves.png) | 新Phase U训练；别与旧4988928起点混同 |
| S11 | [新10M最后模型代表性轨迹诊断](../../DVGC/JIT/runs/phase_u/phase_u_v4_speed2_denseckpt_deferred_10002432_seed820701_20260928/evaluations/transition_10002432/representative_diagnostic.png) | 代表性单轨迹，不代表八条都完全相同 |
| S12 | [桥接v1.1 pilot训练访问xz](../../DVGC/JIT/runs/experiments/generative_bridge_v1_1_20260928/recovery_0002/student/training/recovery_0002_student/train_panels/transition_128000/xz_visitation.png) | 访问分布不是采用/保持证明 |
| S13 | [桥接v1.2 C训练访问xz](../../DVGC/JIT/runs/experiments/generative_bridge_v1_2_20260928/recovery_0004/campaign/students/C/training/campaign_C/train_panels/transition_128000/xz_visitation.png) | 对应C；A/B同目录各有图，三个均未正式采用 |
| S14 | [连续学生R67训练访问xz](../../DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/segment_0067/round_0067/students/student/training/round_0067_student/train_panels/transition_128000/xz_visitation.png) | R67学生训练面板，不冒充固定9条件DEV或最终TEST |

**论文结论校准。** 支持：阶段能力互补；局部失败状态可被补训/桥接转化；存在显著老师—学生吸收与旧能力保持瓶颈；常驻B1有实际工程吞吐收益线索。不支持：学习探索已同成本优于随机、G已证明不可替代、累计包线等于最终单Actor能力、安全可达集或sim2sim/实车泛化。S01成功格子几乎相同、S02最终复测负结果、S12/S13未采用及S14相对初始C退化必须与正结果一起保留。

**后续最多三项。** ①在已授权运行完成/出错后回收完整边界、九条件DEV及新增/丢失，不以本次汇总增加预算或轮询服务；②如需证明探索必要性，补齐同源同预算随机对照及最终单Actor统一评价，目前S04未闭环；③解释老师有效教案到学生吸收/保持的具体瓶颈，包含重放冲突隔离，不以G的MSE下降代替能力证据。

- 2026-10-10 用户纠正：本节本地证据链接改为相对本台账文件的路径，避免IDE Markdown将 `/home/qy/...` 解析为工作区/预览根路径；原文件未移动。S03改为直接报告“训练完成，但最终配对效果未验证”，不再用训练过程转化次数替代效果。路径解析存在性已核对，用户IDE点击行为无法在终端代验。


### JIT-S03-EVAL-20261010 — 随机100轮最终模型配对补评

用户授权执行对比。冻结repair10初始、学习100轮最终0098、随机100轮最终0099；复用已存前两模型每起扰256回合轨迹，新增随机模型4×256回合，起扰0/5/10/15、三步四通道±0.25，最长400步、0.5秒稳定恢复。新增最多409600控制步，训练0。复用原冻结评估代码与256批量/种子，逐数组检查draw、请求与分母；不是独立训练种子或final TEST。GPU每阶段空闲显存≥20000MiB、不预分配、不停止其他任务。状态prepared；评价pending，结论等待原始输出。目录：[评估声明](../../DVGC/JIT/runs/monitoring/random100_final_comparison_20261010/declaration.json)。


### JIT-S03-EVAL-20261010 — 补评完成，替代此前结果缺失状态

- execution completed；evaluation complete；adoption not_applicable。本次只有推理评价，无训练，不改变现有模型。随机0099新增4×256=1024回合，计费409600/409600控制步；0010/学习0098复用历史2048回合。相同256批量、四起扰种子和冻结评估代码；XML/动作顺序/Actor观测契约核对，逐元素随机draw及共同存活请求、有限轨迹、全部失败分母检查通过。历史轨迹不是本日重跑，保留运行时数值漂移限制。
- 起扰0/5/10/15，每格256：训练前66/84/239/254；学习探索后43/90/241/249；随机探索后44/79/215/228。合计训练前62.79%、学习60.84%、随机55.27%。随机相对训练前少77次成功，相对学习少57次；对应配对回合bootstrap描述性95%差值区间[-10.55,-4.49]、[-8.40,-2.64]个百分点。
- 直接结论：本次固定初态四时刻等权面板，随机探索100轮最终模型没有提升；学习探索最终模型优于随机最终模型，但仍未超过训练前。各方法一个训练来源、实际训练成本不等、面板曾用于开发，不升级为算法普遍优越性或final TEST。
- 证据：[结果与图](../../DVGC/JIT/runs/monitoring/random100_final_comparison_20261010/INDEX.md)、[成功率图](../../DVGC/JIT/runs/monitoring/random100_final_comparison_20261010/success.png)、[跳跃xz轨迹图](../../DVGC/JIT/runs/monitoring/random100_final_comparison_20261010/trajectories.png)、[逐回合数据](../../DVGC/JIT/runs/monitoring/random100_final_comparison_20261010/episodes.csv)。PNG已视觉检查，原始NPZ保留；不生成重复PDF。此前“尚未找到最终对比结果”是补评前状态，现已由本条与S03替代。
- 后续：保留该负结果，与旧训练高成功率同时解释；如需要更强泛化结论，再单独声明新初态、独立种子和等实际成本试验，本次不自动扩展。

### STTW-LOWER-NOALPHA-20261010 — stage200 result and bounded extension

- user_requirement: 用户叫回查看训练结果、分析并继续原任务；沿用无轮询/桌面CMD偏好。repo_artifact/artifact_checked:200/200、26,214,400转移、200accepted，固定100/150/200及磁盘best重载评价完整。best200/last200（角色不同），train-reward候选164。
- fixed physical score tuples100=[0,0,6,3,6.24298],150=[0,0,6,3,4.45828],200=[0,0,5,2,2.84602]；new合格2/7，R1963/7，new未采用。small±窗口转角RMSE.006948/.007942rad vs R196.014099/.014545改善，但whole-case主跟踪未全过；negative steady持续7.835s>.02rad。旧2.43s状态未补评，不能称消除。
- audit:同一发布reference/rates双方逐值相同；best SHA5d2cf35739f81429b7f6cbff39066dc1ad6d72f939dd8524e955e0609ef37df4；200完整闭环checkpoint存在，Actor与resume_boundary一致。best重载轨迹有float32小差（maxXY.0001212m/maxdelta.00002908rad），四类计数一致；selection及重载数据都保留。
- decision: Q150→200改善36.16%、跟踪失败6→5/末段3→2、全部四训练族后25完整回合成本/速度/转角均改善，符合附件100延长条件。批准范围内执行exact200→300，新增13,107,200/累计39,321,600转移；保存每10/完整每50；数值验证250/300、沿用R196缓存；300重载best后停止，不自动扩到400。配置/reward/物理/权限/模型不变。
- evidence: [stage200分析](../runs/worktrees/lower-first-noalpha/runs/noalpha_v2_formal_20261010/analysis/RESULTS_0200.md)、[七场景XY与跟踪](../runs/worktrees/lower-first-noalpha/runs/noalpha_v2_formal_20261010/analysis/INDEX.md)。局部门槛关闭，旧失配恢复/新adapter注册安装/冻结upper组合均not_run；upperbest0added200/best1added250继续冻结，无上层训练/push。

- STTW-LOWER-NOALPHA continuation startup checked: PID1637265，exact update200→300，同run、reward/物理/模型不变；首201/202均接受，已核验累计202/300、26,476,544/39,321,600控制转移。TB6006 formal/. reward step202已通过HTTP加载，receipt run/reward_http_verified_0201.json；CMD200→300窗口启动退出0。阶段200精简图/数据与分析提交2b214f1，未push；采样统计随后附入同证据目录。固定steady约±.120922rad vs ordinary训练≤.10存在联合覆盖差距，保持原验收/冻结分布，未证明是失败因果。继续按用户偏好不轮询，300结束用户叫回再回收、判断≤400或门控下一步。

### JIT-S03-MULTIDIM-20261010 — 三模型初态随机化与共同探索器评价

用户已确认中等初态扰动及同一个冻结探索网络。repair10初始、learned0098、random0099，各1000回合（0/5/10/15步起扰各250），共享初态和随机种子；3步四通道±0.25。初态x±.10m/y±.05m、三姿态±3度、vx/vy±.20m/s、三角速度±.10rad/s；记录几何高度修正，不按成功筛选初态。同一个状态条件探索器不等于相同实际脉冲。冻结learned轮0099探索器及其原参考邻域，不将邻域标签冒充其他Actor能力。

无训练，正式最多1,500,000控制转移（3000计分+每模型250并行容量一致的无扰动副本，后者只有一个独特条件，不能扩大统计分母）；工程预检最多4800。GPU阶段启动门槛空闲显存20000MiB，关闭预分配，不停止现有训练。代码独立worktree `agent/multidim-envelope-20261010`，父c98701a；17项CPU测试通过，GPU预检pending。声明/初态/模型入口：[declaration](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/declaration.json)。执行prepared，评价pending，采用not_applicable；报告成功率、失败、XYZ/姿态/速度、无扰动相对偏移、共同尺度多维占据与敏感性，不将轨迹宽度或占据格数叫安全可达体积。

- JIT-S03-MULTIDIM启动核验：独立提交3488839，18项CPU合同测试通过；GPU预检4×400=1600步，实际qpos/qvel与声明库逐值吻合，有限、恰好3步扰动、无训练更新。三模型物理/观测/动作身份与环境配置核对通过；配置差异为训练来源/support/预算元信息，显式初态覆盖历史reset支持。正式串行评价PID1673161；[实时状态](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/status.json)，桌面watcher PID1673193心跳正常、delivery_errors为空。启动后2/15批完成，结果未定；无训练或现有任务重启。

- JIT-S03-MULTIDIM中间核验：全部3000计分回合完成并通过初态/脉冲/有限状态核验。初始291/1000、学习188/1000、随机220/1000；图和无扰动参考尚未完成。首个名义参考被legacy单回合专用接口在仿真前拒绝（0新增转移），[错误原件](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/nominal_precheck_failure.json)保留；[修正规格](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/nominal_amendment.json)仅移除专用标志，以原250容量/零幅度继续3个参考，预算不变、评分不重跑、物理/模型/代码3488839不变。恢复PID1717826，ACTIVE_RUN已更新。

### JIT-S03-MULTIDIM-20261010 — 完成：更强联合扰动下两组训练均退化

- repo_artifact/reproduced（本次声明面板）；execution completed，evaluation complete，adoption not_applicable。三模型各1000回合全部核验，另3×250同容量名义副本（每模型只有一个独特名义条件）；正式1,500,000控制步、工程1600、训练0。源码独立提交3488839；18项CPU测试，15批真实初始化/脉冲时窗/有限状态核验及原冻结输入哈希复核通过。名义预检错误已按上述修正规格恢复，原错误保留；未重跑评分回合、未改物理/奖励/模型，未push。
- **训练前repair10：291/1000=29.1%；学习100轮最终0098：188/1000=18.8%；随机100轮最终0099：220/1000=22.0%。本次两组训练都没有超过训练前，学习组比随机组低3.2个百分点。** 起扰0/5/10/15成功数分别初始71/95/63/62、学习64/46/38/40、随机58/61/48/53，每格250。学习相对初始新增104、丢失207，差−10.3pp；随机新增107、丢失178，差−7.1pp；按起扰组分层回合配对bootstrap95%分别[−13.7,−6.9]、[−10.4,−3.8]pp，学习相对随机[−6.0,−0.5]pp。区间条件于各方法单个训练来源，不能称跨训练种子显著优越/劣势。
- 成功轨迹峰值根部高度P50初始/学习/随机0.575/0.638/0.646m；落地点世界x的P5–P95分别3.872–4.298/3.936–4.235/3.835–4.154m。跳得高不等于更抗扰；侧倾超限517/726/683次为主要失败。成功判据是原冻结稳定落地前进0.5s，不要求回归原路径，不证明可指定目标命中。
- 多维覆盖已交付XYZ/XY/XZ/YZ、12维成功/全部存活分位带、名义相对偏差、离地对齐、初始x×起扰和实际施扰幅度。12维格数20960/13876/15913均>99%接近对应成功状态采样点数；**拒绝把该稀疏格数当可靠能力体积或提升率**。初始x±.1跨越原x2.5跳跃触发边界，四起扰是reset后步数而非严格同一物理相位。共同探索器依赖各自状态，平均有效脉冲RMS .138/.142/.143，不是相同实际动作序列。
- 名义250副本各自成功248/249/200，仅一个独特条件，不扩大统计样本。初态与首步动作一致但首步物理状态有微小差异并被放大；黑线为预先固定lane0，不能当唯一确定轨迹，相对偏移含数值敏感性。此协议同时改变初态和施扰规则，与旧固定初态均匀脉冲62.79/60.84/55.27%分别保留，不能混作单因素改善。
- 证据：[完整结果与11张图](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/INDEX.md)、[XY/XZ跳跃段](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/jump_detail.png)、[成功率](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/success.png)、[12维相对偏差](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/relative_bands.png)、[逐回合](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/episodes.csv)、[核验](../../DVGC/JIT/runs/monitoring/three_model_multidim1000_20261010/verification.json)。原NPZ/共享初态/冻结身份及分析代码均留存，关键图已视觉检查，本地链接存在性核对。
- 下一步最多两项，未执行：①同库仅初态扰动对照，分离起点敏感性；②冻结实际脉冲序列回放与侧倾失败诊断，分离探索器反馈和模型恢复能力。不自动追加实验/训练；本结果不支持扩大最终单策略抗扰能力的正面论文结论。

### JIT-VERIFIED-MODELS-20261010 — 高成功率旧图解释及模型身份核验

- 本次重新统计旧图对应10份逐回合CSV共450000行，计数与各条件summary及20260928派生汇总一致，未重跑仿真。每模型每协议5×10000；固定初态：原始PhaseU37727/50000=75.454%、RSI36536=73.072%、repair10 34400=68.800%、repair70 32915=65.830%；小幅速度噪声：PhaseU43808=87.616%、融合best34275=68.550%、RSI28739=57.478%、repair10 20784=41.568%、repair70 14864=29.728%。五模型checkpoint目录实际存在，精确来源及条件证据已写入[核验清单](../../DVGC/JIT/runs/monitoring/model_inventory_20261010/VERIFIED_MODELS.json)。融合累计256万对应student_b/transition1920000，不能按文件尾号误当累计192万。
- 旧图高分不对应刚才的三模型同工况：蓝线是原始PhaseU transition4988928；新测试训练前是repair10（旧图紫线），新1000回合未包含原始PhaseU/RSI/融合best。旧图初始位姿固定；右图仅vx/vy±.1、角速度±.05、三关节速度±.1；新测加位置/姿态、vx/vy±.2、角速度±.1且不加关节噪声。旧为独立均匀±.25三步动作，新为共同状态条件探索器；旧5起扰点包含25，新4点不含25。稳定落地0.5s口径相同，但多因素/种子/执行批量不同，不将成功率差叫单因素噪声效应或模型被覆盖。RSI旧固定第10/15/25步近100%，第0步仅21.12%；PhaseU旧固定第10步仅43.54%，不是全工况满分。
- 其他有评价的主要模型族：repair10学习100最终0098、随机100最终0099（固定四点复测60.84/55.27%，新联合扰动18.8/22.0%）；三组额外15M后代（repair10/RSI/repair70的固定4×1000回合2346/1197/1010成功，即58.65/29.925/25.25%）；新20260928 PhaseU10002432仅8回合顶点/存活验证、均timeout，不当稳定落地成功率；FreshRSI1M历史快速测试0/100，不能因训练完成就称合格。PhaseU发现奖励0093当前源仍96轮，缺最终同协议统一复测，不列为已证明更强。详见既有S04/S06/S07/S11证据。
- 当前桥接系列新增核验最近完整R71：固定9条件DEV学生228/288=79.17%，P0 186/288=64.58%，初始C 228/288=79.17%；对P0新增85/丢43，对初始C新增45/丢45。总数追平初始C但并非原有成功集合完全保留。formal_adopted=false，仅实验连续训练采用；不与旧50k或新1000面板横排名。证据：[R71原始配对](../../DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/segment_0071/round_0071/generalization_comparison.json)。状态快照原任务67/200、当前R72；旧R67/63轮快照保留为历史。本次未修改/启动/停止训练。
- 下一步建议而未执行：在同一新1000初态/探索器面板补入原始PhaseU4988928和融合best，才能回答库存最强模型在新工况是否仍最好。当前不能把旧87.62/68.55%套到新联合扰动工况。


### JIT-ALL-HARD1000：全模型较难初态统一评价（2026-10-10，准备与预检）
source_type=user_requirement+repo_artifact；verification=artifact_checked。用户授权旧评估更换较难初态、所有已讨论模型各1000次，并加入正在桥接训练的最新完整学生。按旧图协议假设保留均匀随机±0.25动作脉冲3步，起扰0/5/10/15/25各200，已提出非阻断澄清，尚无回复；正式未启动。共享上次1000组初态（x±.10m/y±.05m，姿态各±3°，vx/vy±.20m/s，局部角速度各±.10rad/s；无vz/关节额外扰动）。最长400步/.02s，保留stable_forward_recovery判据。
15模型：原PhaseU、repair10、repair70、FreshRSI8M、融合2.56M、学习100轮0098、随机100轮0099、三种+15M PhaseU、FreshRSI0.9984M、融合last3.20M、PhaseU discovery0093、新PhaseU10.002432M、桥接R71学生；R71在声明时是最新完整已验证round。单Actor评价，不启用生成器在线救援。
预算：15000计分回合+每模型一个无扰动条件的200并行副本，正式最多7200000控制转移；预检最多3200，无训练。独立worktree agent/multidim-envelope-20261010，父3488839；仅增加rerun_reference来源元数据兼容，物理/动作/奖励差异仍拒绝，10项接口测试通过。GPU批前空闲20000MiB门槛，关闭预分配；保留现有训练。
[声明及完整模型路径](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/declaration.json) · [模型身份](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/models.json) · [实时状态](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/status.json)。后续：预检通过后串行执行、生成成功率/失败/XY与多维成功经验包线、核对并回收结果。不宣称能力提升或安全可达体积。

- JIT-ALL-HARD1000协议确认与更正：用户回复明确选择**共同探索网络，0/5/10/15步各250**，覆盖准备阶段的随机脉冲假设；正式随机评价未启动。声明/spec已改为与上一批同一个round0099冻结探索器及参考邻域，15模型共享初态与采样种子，实际扰动因状态而异。原随机准备归档superseded_random_preparation；仅完成一次随机桥接预检1600转移，计工程开销、不计正式分母。正式15000计分+3750名义副本=最多7500000控制转移；总工程预检4800（随机1600+两个学习施扰预检各1600）。
- 评估代码04aaf6f；允许冻结学习施扰加载PhaseU，允许只读评价连续learner来源的桥接学生，默认训练加载仍拒绝该新初始化类型。16项CPU检查通过，两个新模型GPU各4×400通过初态、脉冲、有限性、零更新验证；首次旧接口拒绝在仿真前发生，保留错误原件。正在训练的PID1011565仍保留。桥接冻结R71（声明时最近完整轮），不会随训练推进更换测评权重。

- JIT-ALL-HARD1000启动核验：正式主进程1802708已运行，当前onset_05_phase_u、已完成1/75批。15模型冻结配置输入哈希、两相环境、XML、观测/动作字段全部一致；两关键模型GPU预检通过。自动分析用历史291/1000结果回算吻合，生成图已检查；该fixture不计新物理实验。watcher1802741心跳正常、delivery_errors为空，保留DBus。正式完成后自动生成[索引](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/INDEX.md)并持锁追加结果到本台账。当前结果尚未完成，不宣称模型提升。代码04aaf6f本地已提交，未push。

### STTW-LOWER-NOALPHA-20261010 — stage300结果回收

- user_requirement: 用户叫回分析新的训练；无Codex轮询。repo_artifact/artifact_checked：300完成，300接受更新/39,321,600控制转移，固定250/300与best磁盘重载完整。未新增训练或仿真。
- best300/last300不同角色；随机奖励候选281由282采样计分。250 tuple=[0,0,4,1,2.426835]未合格；300=[0,0,0,0,1.739090]、重载Q1.739101，7/7合格；R1963/7、Q3.506210。Q200→300改善38.89%。只有一次后期合格，不能称附件全部完成。
- steady_negative原[2.165,10)7.835秒>.02rad失配消除，300只入弯[1.04,1.515).475秒；小角±RMSE.003186/.004913rad，平台有效/方向通过。.9–.96秒转弯/回正瞬态仍在；XY仍有原始位置偏差。此结果不证明旧B0[6.315,8.745)消除或上层任务合格。
- audit:best SHA8aa8a90e76dce8c8ad34a8cbca3b4602015926f3d18e24af3000ac2bc26b3577；完整update300/Adam/RNG/512环境历史保存；metrics1–300无缺重复；reference逐值相同、R196与upper0/1哈希不变。重载maxXY.000130654m/maxdelta.00001666rad，计数判断一致，不称bitwise。原200数据保留，stage300独立analysis_0300。
- decision:原6.315秒完整闭环恢复比较待做，连续后期合格仍需核验；未自动延长400/注册/安装/upper组合/上层训练/push。旧恢复点失败时遵循定向恢复状态补训，不堆普通任务。
- evidence:[七场景实际XY/跟踪与完整结论](../runs/worktrees/lower-first-noalpha/docs/evidence/local_lower_noalpha_v2_0300/RESULTS_0300.md)；run/training保存selection与最终重载原NPZ，analysis_0300保存逐帧奖励重建。

- STTW stage300交付：本地逻辑提交7313084，30文件含七场景XY/逐时奖励分项/真实与轮速速度对照及训练loss/KL。图视觉检查、七场景command/rate/time/reference逐值核对、奖励重建数值/有限性、文档链接及diff检查通过；未重跑工程批或旧11项/全仓测试，未push。

### STTW-LOWER-NOALPHA — 原完整状态恢复比较启动声明

用户明确要求继续并开始，复用旧B0初始完整快照，单次重放1263控制步至6.315s，构造十帧真实old-controller局部历史，原R196310历史保留；相同plant/ESO/actuator状态各400步/2s固定[2.3,0]，总2063控制转移，训练0。仅力测量附加不改物理。新底层best300；验收1s内入.10m/s/.04rad连续.5s且不重新离开、不新增工作界越限。2项新增必要验收测试通过，原11/全仓不重跑。未安装/注册/上层训练。输出runs/worktrees/lower-first-noalpha/runs/parent_recovery_best300_20261010；execution prepared、evaluation pending；失败只进入该类状态定向修复。

### STTW-LOWER-NOALPHA — 原状态恢复通过与350检查点

repo_artifact/reproduced：单次1263步prefix与原B0关键轨迹逐值一致，相同全部physical树状态及10有效因果局部帧，400步各2s；new best300 .445秒入带并保持到终点，R1962s全转角超差。峰值侧倾new.072501/R196.080807，无新增工作界/物理失败；new后轮force触限4/10000子步、不隐去。总2063控制转移/训练0，未重复完整12s诊断。单状态局部恢复结论，不证明世界路径或V5.2组合。

决策：300局部首次通过、原状态通过，但连续后期合格缺第二次；最后至多100更新块先执行300→350（50/6,553,600转移、累计45,875,200），350是既定验证停止检查点。通过则结束底层训练；未过再依据同一证据规则决定余下50、绝不超过400。不重新初始化/修改分布reward/物理；保留best300/final重载数据。用户原任务授权及本次明确开始，无需追加确认；无持续轮询。upper仍冻结、无组合/上层训练/注册采用。

证据：[旧完整状态XY与恢复结果](../runs/worktrees/lower-first-noalpha/docs/evidence/local_lower_parent_recovery_20261010/INDEX.md)，9新增/直接相关检查通过，旧11/全仓直接复用。

- STTW恢复交付0ced7d7本地提交、未push；完整300→350已启动PID1860608，当前运行300/350，39,321,600/45,875,200转移，保存原final_best_validation_0300。桌面CMD窗口退出0，TB6006 HTTP200；等待301reward当前标量加载核验后停止主动轮询。命令/身份见run/resume_launch_0300.json，新日志console_0301_0350.log。下一次用户叫回先收350固定验证及重新加载best，再判断冻结组合门槛；不自动上层训练。

- STTW350启动核验完成：PID1860608，完整边界300恢复；已接受301/350，39,452,672/45,875,200控制转移，state running，TB6006 formal/. reward301 HTTP实际加载，receipt reward_http_verified_0301.json；CMD窗口已打开。现在停止Codex主动轮询，350结束用户叫回再审查第二次合格/最终best及冻结组合。补充post-step1秒边界验收测试3项通过，与原7必要v2检查构成10项；追加本地提交624aac1，未push。

- JIT-ALL-HARD1000自动回收（2026-10-10 14:16:36）：75/75批执行与初态/有限性/脉冲/端点核验完成，15000计分回合，零训练更新。Phase U（4,988,928步） 658/1000 (65.8%)；Generative bridge R71 student 309/1000 (30.9%)；lineage_repair_0010 292/1000 (29.2%)；lineage_repair_0098 184/1000 (18.4%)；Random exploration final 0099 222/1000 (22.2%)；全新RSI（8,000,000步） 134/1000 (13.4%)；lineage_repair_0070 219/1000 (21.9%)；融合学生（累计256万步） 205/1000 (20.5%)；lineage_repair_0010 + Phase U 188/1000 (18.8%)；全新RSI（8,000,000步） + Phase U 158/1000 (15.8%)；lineage_repair_0070 + Phase U 345/1000 (34.5%)；Fresh RSI 0.9984M 0/1000 (0.0%)；Fusion last 3.20M 26/1000 (2.6%)；Phase U discovery repair0093 299/1000 (29.9%)；Fresh Phase U 10.002432M 194/1000 (19.4%)。[结果及图](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/INDEX.md)。图已程序生成，人工视觉与结论复核待下一次分析；共同自适应探索器不代表相同实际扰动力序列，结果仅限声明初态/施扰分布。


### JIT-ALL-HARD1000 — 2026-10-10完整结果人工核验
source_type=repo_artifact；verification=artifact_checked（原15,000物理回合独立重算，未新增训练/仿真）。75/75批及15模型全套图已完成，成功数与summary逐项一致。排名：原PhaseU65.8%、repair70+PhaseU34.5%、桥接R71 30.9%、discovery0093 29.9%、repair10 29.2%、随机100轮22.2%、repair70 21.9%、融合2.56M20.5%、新PhaseU10M19.4%、repair10+PhaseU18.8%、学习100轮18.4%、RSI+PhaseU15.8%、RSI8M13.4%、融合last3.20M2.6%、RSI1M0%。
学习100轮相对共同起点repair10新增成功100、丢失208，净-10.8pp；随机净-7.0pp，学习比随机-3.8pp。桥接R71相对原PhaseU新增82、丢失431，净-34.9pp；对repair10仅+1.7pp，回合配对bootstrap95%[-2.2,+5.7]跨零。仅声明分布下结果，不支持整体抗扰提升或探索学习优于随机；局部获益不能代替旧能力保留。
主要失败为侧倾越界（桥接521、学习728、随机685、原PhaseU275），桥接另155禁止接触。学习组初始x<2.5m成功40/499=8.0%，x≥2.5m144/501=28.7%；原PhaseU69.3%/62.3%。窄初态/起跳条件适应为待检验解释，不冒充因果证明。共同探索器请求RMS原PhaseU.128、桥接.136、学习.143、随机.144，不能叫完全相同实际扰动；排名第二repair70+PhaseU名义副本0/250成功，数值分叉与单条件副本限制保留。
[完整分析](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/INDEX.md) · [XY/XZ同图](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/key_models_xy_xz.png) · [全部排名](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/audited_ranking.png) · [初态/时刻分层](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/key_models_strata.png) · [独立核算与配对区间](../../DVGC/JIT/runs/monitoring/all_models_hard_initial1000_20261010/result_audit.json)。60张模型PNG和15份坐标NPZ齐全；关键轨迹图已视觉检查，所有索引链接存在。
下一步建议（未启动）：保留原PhaseU为比较锚点并加入旧能力保留门槛；补初态-only及共同开环动作序列分离施扰影响；针对起跳区前/侧倾/落地失败定向修复并保留旧成功初态。评价已完成不代表主训练结束：本次读取桥接主流程仍在R74运行，测评仅冻结R71，不作中止/采用/续训动作。

### STTW-LOWER-NOALPHA — 350结果与最终剩余50声明

- user_requirement：用户叫回继续分析；repo_artifact/artifact_checked：350完成、350接受更新/45,875,200转移、final best重载结束。last350tuple=[0,0,1,1,1.734559]，6/7；best保持300，重载Q1.739119、7/7，随机奖励候选336由337采样计分。
- steady_positive速度平稳RMSE.053094>.05，持续[1.52,10)8.48s超差，转角.014875<.02；末段速度也失败。new小角±.008797/.002873rad，前者较300退步。未放宽阈值或用reward/零摔倒替代合格。best300原失配恢复.445秒入带的结论保留，但不是对last350的保证。
- 后25完整回合成本比前25升1.88%，分族信号混合；250/300/350失败4/0/1，非三次同类平台。按已声明最后至多100澄清块补剩余350→400的50，新增6,553,600、累计52,428,800；完整350learner/闭环状态精确恢复，无reward/采样/物理修改、无新工程批。400后停止；即使400通过也不能把300与400叫连续通过；不自动注册安装/上层组合或训练。
- audit：metrics1–350无重复/缺号、完整512历史/Adam/RNG、best300 SHA不变、双方reference/rates逐值同一、R196/upper哈希不变。没有新增仿真，原输出复用；best重载小数值差但判断不变。
- evidence：[350七场景XY与退步证据](../runs/worktrees/lower-first-noalpha/docs/evidence/local_lower_noalpha_v2_0350/RESULTS_0350.md)。原300、350selection与本次重载数据独立保留，无全仓测试/push。

- STTW350证据本地提交bce6201，未push；七场景图已视觉检查、逐值shared reference核对、完整learner/闭环状态与protected upper/R196哈希检查、diff/报告链接通过，未重跑全仓。最后350→400阶段启动PID1973907，完整350边界，同run/config/reward/物理，新增6,553,600至52,428,800硬上限。CMD窗口退出0，TB6006 HTTP200；等待首351标量核验后不持续轮询。原本次重载best保存final_best_validation_0350，350selection与完整lastcheckpoint保留。

- STTW最后阶段启动核验：PID1973907，351/400接受更新、46,006,272/52,428,800转移，state running；TB6006 formal/. train/mean_step_reward351或更新已通过HTTP实际加载，receipt reward_http_verified_0351.json。现在依用户偏好停止主动轮询，400结束用户叫回回收；到硬上限不新增训练/凑连续合格/上层训练。


### JIT-BRIDGE-PI0-RETEST — 桥接初始π0直接复测（2026-10-10）
用户明确要求桥接最初π0也按同协议1000回合并比较。已核对桥接audit/binding的transition_0及初始化receipt：Actor a06acbc8…、normalizer db84aad7…、critic 5b864131…均与PhaseU transition_4988928一致，重绑定零训练。此前658/1000属于同参数原PhaseU；本次仍按用户要求直接加载桥接保存checkpoint验证运行配置，不将此前分数冒充新实验。
预算：π0新增1000计分（四起扰0/5/10/15各250）+250名义副本=最多500000控制转移，无额外训练/预检；相同初态库、冻结探索器round0099、幅度±.25与持续3步、400步上限、stable_forward_recovery门槛。复用已完成R71 309/1000与原PhaseU658/1000作为配对比较；不移动R71比较目标。独立代码04aaf6f、串行GPU空闲20000MiB启动门槛、保留其它训练。
[实时状态](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/status.json) · [声明](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/declaration.json) · [身份核对](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/identity_comparison.json)。准备完成后启动；完成自动出图与更新本台账。


### JIT 授权B1续训阶段事件

- B1续训停止，无自动重试：RuntimeError('round 74 exited 1; no automatic retry')；/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/status.json

- JIT-BRIDGE-PI0-RETEST自动回收（2026-10-10 14:26:45）：5/5新增批执行与初态/有限性/脉冲/端点核验完成，新增1000计分回合、另复用原PhaseU与R71各1000回合，零训练更新。桥接初始π0（直接加载） 651/1000 (65.1%)；Phase U（4,988,928步） 658/1000 (65.8%)；Generative bridge R71 student 309/1000 (30.9%)。[结果及图](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/INDEX.md)。图已程序生成，人工视觉与结论复核待下一次分析；共同自适应探索器不代表相同实际扰动力序列，结果仅限声明初态/施扰分布。

- JIT-BRIDGE-PI0-RETEST人工核验完成：5/5新增批、1000计分+250名义副本、500000控制转移，零训练。π0直接路径651/1000=65.1%，R71复用309/1000=30.9%，差-34.2pp，分层配对bootstrap95%[-38.0,-30.3]；新成功84、丢失426。四起扰π0为196/148/127/180，R71为92/81/63/73（各分母250），均下降。原PhaseU旧结果658/1000，参数相同但33回合标签分叉；初态和首步动作逐值一致、首步qpos/qvel已有微小差，不能称逐轨迹等价或0.7pp训练变化。共同自适应施扰边界不变，结论为R71在声明工况未保留π0整体抗扰能力，不自动归因于生成器单一模块。
[完整对照](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/INDEX.md) · [XY与成功率同图](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/pi0_vs_r71.png) · [配对计数](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/comparison.json) · [数值重复差异](../../DVGC/JIT/runs/monitoring/bridge_pi0_hard_initial1000_20261010/repeat_numerical_diagnostics.json)。独立重算与自动summary吻合，图视觉检查、索引链接检查通过。未改/停主训练；后续仍建议π0旧成功保留检查、同外扰分解、定向修复。


### JIT 精简连续TensorBoard与训练诊断（2026-10-10）

- user_requirement：多次续接曲线间断、内容太多，要求合并精简并诊断。已沿实际continuation_parent链汇总R1–R73已完成轮次，排除独立试点及未采用失败重跑；R3优化器bootstrap边界、R33 B1切换明确。单run main八项，横轴round、每轮记录均值，不插值、不虚构R27发布恢复耗时。
- artifact_checked：http://localhost:6037 实际HTTP八tag/73奖励点；原日志保留。证据/home/qy/DVGC/JIT/runs/monitoring/continuous_key_metrics_20261010/INDEX.md，overview.png、metrics.csv/json和lineage.json。当前为停止时快照，不冒充自动跟随未来续训。
- findings：R24–33训练回报均值701.1→R64–73的886.6，但固定DEV84.13%→77.36%，R73为194/288=67.36%；九个DEV条件声明核对一致。提示近期恢复回退/波动，非过拟合或遗忘定论。G train MSE0.04575→0.04061不证明恢复增强；teacher比例任务组成变化不能直接归因G。R68后耗时约25–28分钟，根因未细分。
- failure：series0011在R74 student_student阶段停止，XLA GEMM autotuner的ptxas退出139；具体工具链原因UNKNOWN。本轮69/200完整至R73；没有新训练/仿真，也未修改算法或重启。下一步定位编译器故障并保留固定DEV对照。


### JIT-SUCCESS-ENVELOPE-12D — 成功范围量化（2026-10-10）
source_type=repo_artifact；verification=artifact_checked。用户要求进一步比较范围并把12维成功包线量化成数值；仅读取已有轨迹，无新增仿真/训练。采用事后描述性“12维等效宽度指数”：reset→离地→首次有效接触→真实终点三阶段各16点；每模型每起扰组无放回30条成功轨迹、共120条，重复200次；各维q95-q05宽度相对同次π0样本宽度取12维×48点几何平均，π0=100。相同初态/动作/物理与原成功定义保留，角度沿轨迹unwrap，小宽度共同下限防零。
事件对齐指数：π0 100；R71 109.6（抽样2.5–97.5百分位103.8–116.3）；repair10 138.8；学习100轮122.9；随机100轮124.3。成功率分别65.1/30.9/29.2/18.4/22.2%。桥接阶段指数起跳前102.7、飞行113.8、接触后112.2；x130.5、vx141.8、z124.7、roll117.4/pitch118.4，y97.4、vy91.0、yaw97.7、ωx89.2，非各方向单调扩展。
全程统一进度但不对齐事件时桥接121.9；对齐与阶段权重选择明显影响数值，不将全部差值归因于一个机制。小宽度阈值0.1×/10×时事件指数109.7/107.4，方向稳定。109.6表示典型相对宽度约1.096倍，不能解释为12维可控体积。
联合近邻补充：12维共同尺度[位置.05m/角度3°/线角速度.2]，事件对齐，支持120与不重叠查询80成功轨迹/模型，查询分布两模型各半，50次划分。RMS半径1时仅邻近桥接27.0%、仅邻近π0 31.8%；总覆盖桥接42.3%、π0 47.1%。半径.5/2时排名改变，联合范围无尺度无关优势或包含关系证据；这是样本近邻覆盖，不是给未测状态判定可控。
结论更精确：R71成功轨迹的典型范围略宽且分布不同，但可靠性显著下降；不支持全面包住π0或更强整体控制能力。12维为完整状态投影，遗漏关节/历史，不能据此认证恢复域。应从候选新增区的完整状态做同条件撤扰恢复验证，方能支撑真实控制域扩张。
[量化报告与全部证据](../../DVGC/JIT/runs/monitoring/success_envelope_quantification_20261010/INDEX.md) · [宽度图](../../DVGC/JIT/runs/monitoring/success_envelope_quantification_20261010/event_aligned/width_comparison.png) · [联合覆盖](../../DVGC/JIT/runs/monitoring/success_envelope_quantification_20261010/event_aligned/joint_coverage.png)。公式尺度性质、等量样本、覆盖分解和链接检查通过，关键图已视觉检查；误差带只表样本抽取敏感性，不表训练种子置信区间。旧两份结果索引已追加量化入口。


### JIT-KEY-RESULTS-BUNDLE-20261010 — 精简结果归档
- user_requirement：当前结果、台账与模型测试一并打包；不含逐回合数据，体积不超过几十 MB。
- repo_artifact / artifact_checked：JIT/docs/evidence/current_results_20261010 收录连续训练汇总、15模型各1000回合测试汇总与关键XY图、π0补充对照、成功范围量化、台账快照和R74编译错误凭证。仅保留汇总和关键图，不含逐回合CSV、轨迹数组或权重；原始实验不变。
- 边界：原任务69/200完整至R73、R74编译失败；本次不恢复训练，不新增物理过程或能力结论。离线归档将存于 /home/qy/DVGC/JIT/runs/exports/jit_key_results_20261010.tar.gz；远端目标为 QaQaaa-zzz/DVGC 的 agent/generative-bridge-v1-1 分支，实际推送结果以Git凭证为准。
