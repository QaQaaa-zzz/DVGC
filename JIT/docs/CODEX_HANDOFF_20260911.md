# JIT 接手入口（维护至2026-09-12）

**当前接手顺序：** [PROJECT](../../PROJECT.md) → [CURRENT_STATUS](CURRENT_STATUS.md) → [方法协议](ENVELOPE_ITERATION_PROTOCOL.md) → [路线](JIT_TRAINING_ROADMAP.md) → [论文与图件](paper/README.md)。

截至2026-09-12，两轮延迟探索pilot的128k+128k训练已经完成，新增延迟0→1见证为0。残差PPO、per-π到达奖励、pending训练与延迟评价已实现。当前任务是分析奖励饱和与公平实验缺口，不是等待训练或再次实现已有模块。最新补完128,407交互；两轮关联实际348,371，完整证据与限制见当前状态。

旧supervised warm-start和full-action探索是历史阶段，不代表当前frozen π＋4D residual方法。critic质量罚项/扩散/全空间收敛均未实现或未证明。报告分支6121da9早于本地最新结果，应直接核对本地原始工件。此次论文/文档更新没有运行新训练。

---

## 2026-09-11 原始交接快照（历史正文，非当前启动指令）

下列数字和当时的优先顺序保留供溯源；其中“尚未实现/下一步”以本页上方现行入口为准。

# DVGC / JIT 技术交接与实验进度报告

更新：2026-09-11。接手对象：Codex。本文区分生产实验、已实现代码和待实现研究方案。

## 1. 接手结论与第一项任务

项目已经完成一次多策略、两轮自动“探索→训练→再探索”的生产实验，策略库扩展至 π0～π6。不是还在等待 π4 训练，也不是所有论文实验已闭合。

接手后先核对最新远端报告，整理已有数据和修订下一版评价流程，不要直接重跑已完成的 `all_proposers_v1`，也不要直接启动 π7、扩散模型或无限迭代。

优先顺序：核对同时落地/失败的事件语义 → 从现有 CSV 整理几何与贡献结果 → 接续评价加速和墙钟计时 → 多 checkpoint 小预算评价 → 轻量残差探索网络对照 → 视证据决定扩散模型。

本次交接只更新文档与证据摘要，不修改运行源码、不启动训练。用户已授权本项目常规修改、提交、推送，以及服务器自动发布精简报告；无需重新请求同类推送授权。

## 2. 仓库、环境和证据入口

- 仓库：`QaQaaa-zzz/DVGC`。
- 工作分支：`agent/two-phase-soft-tube`；不要读取 main 后判断当前状态。
- 报告分支：`agent/jit-run-reports`。
- 本次文档修订前，已核实运行代码提交：`eb9584d2e01a1ea10dc7835a6fe044dc2599c594`。文档提交会更新分支 HEAD，不代表重新完成仿真。
- 服务器目录：`/home/qy/DVGC`。
- 生产 Python：`/home/qy/mujoco_playground/.venv/bin/python`；`PYTHONPATH=JIT/src`。
- GPU：RTX 4090 D，24 GiB；生产日志 Warp 1.11.0 / CUDA Toolkit 12.9 / Driver 13.0。
- GPU 子进程使用 `CUDA_VISIBLE_DEVICES=<gpu>`、`JAX_PLATFORMS=cuda,cpu`；CPU backend 必须可供 Brax callback 使用。CPU 分析入口使用 CPU，但不主动隐藏 GPU。
- 原始 checkpoint、快照及完整图片主要留在服务器，不能假设 GitHub 的精简报告可独立重放实验。

主要已读证据：

1. [最新两轮总报告](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/summary.json)。
2. [第二轮探索者贡献](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/round_001/figures/proposer_metrics.csv)。
3. [π5 训练日志](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/round_000/training_attempt_000/process.log)；[π6 训练日志](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/round_001/training_attempt_000/process.log)。末尾包含 formal_report。
4. [π2/π4 同安排探索对照](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/pi2_pi4_paired_v1-3ed0b5c90d/summary.json)。
5. [π6 自身候选上的全策略接续评价](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/round_001/discovery/figures/summary.json)。
6. [π5 采集轨迹记录](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/round_000/discovery/tasks/acquire/attempt_0000/result/summary.json)；[π6 采集轨迹记录](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/round_001/discovery/tasks/acquire/attempt_0000/result/summary.json)。

读取策略：先 summary，再 publication 文件索引，再挑选 CSV、失败记录及日志末尾。不要打印全部大 JSON 或要求用户反复上传 ZIP。GitHub 没有某个深层文件不等于实验失败，先检查打包规则。

## 3. 研究目标和不可混淆的定义

任务：固定单轨双轮 bicycle–pendulum 机器人，在声明的起跳条件下，通过互补冻结策略发现经验跳跃包线。目标投稿方向为 RAL，但当前实验规模不能直接保证投稿充分性或录用。

- 起点为完整近地状态，x=2.5 m；约 3.1 cm 初始轮胎间隙已被用户接受，不要求重新接地初始化。
- up/down 专家提供 bootstrap 训练支持；已有统一成功策略提供中心轨迹。
- 中心线使用真实 π0 帧作参考坐标，不是强制跟踪目标，也不插值证明可达。
- 动作顺序：steer、rear-wheel drive、hip、knee；允许四通道合法动作扰动。用户已撤回“只能 hip/knee”限制。
- 当前扰动是声明窗口内的有符号动作偏置，随后限幅；不是直接提升/降低 qpos、注入速度或外力。
- “向下容易获得”是探索直觉，不是任意降低状态都可达的证明。

记号可写为经验到达集 R_hat、成功接续见证集 V_hat，经验支持 T_hat = R_hat ∩ V_hat。但它们是有上下文和协议约束的观测集合，不是完整数学可达域或具有普适成功概率的集合。

| 对象 | 实际含义 |
| --- | --- |
| 前向到达 | 从声明完整起点经真实动力学到达的精确状态和控制/事件历史 |
| 成功见证 | 某个冻结 evaluator 从同一完整上下文接续并满足成功终点 |
| 经验 Tube | 已到达且具有成功接续见证的状态集合 |
| 训练支持 | 用于 reset/replay 的抽样集合；历史 Tube0 不全是成功见证 |
| 物理单元 | 在声明分辨率下对状态投影去重；不保证单元内所有点可行 |
| proposer | 从起点产生新状态的冻结策略及其声明扰动机制 |
| evaluator | 从候选快照继续运行的冻结策略 |

新状态的真实前向轨迹证明“基础策略＋扰动机制能到达”；从快照继续成功证明“能接住”。它不能单独证明新 Actor 从固定起点独立到达该状态。不同 prefix/suffix 策略允许构成离线见证，不宣称由一个 Actor 执行全程。

用户接受已有数值重放差异，明确不要追加重放验证。保留该限制，不标成精确重放通过，也不把它重新设为继续实验的门槛。

## 4. 训练与实验谱系

| 阶段 | 已有成果与限制 |
| --- | --- |
| up/down、handoff、Tube0、统一种子 | 有历史完成记录。Tube0 222 行含 42 个历史失败标签，属于加权训练支持。当前 π0 是后续 Round1 工件，不能混称最早统一训练 |
| π1、π2、π3 | 已有冻结工件。π3 历史混合成功终点 gate 不可作公平选策证据，但策略可在新统一协议下评价 |
| 四策略标签和稠密扫描 | 完成过四策略共面板评价、5 cm 真实帧采集、多 proposer 扫描、局部边界及下缘探索。历史计数基线不同，禁止直接相加 |
| π4 smoke | callback 修复后完成 25,600 PPO 步并冻结；753 到达状态、743 集合见证、734 root cells。最终绘图 KeyError 后通过零仿真分析恢复 |
| π2/π4 对照 | 复用 π4，仅补 π2。探索预算 69,078 下新单元 513 vs 685；包含训练成本后，较小共同预算下 π4 尚未抵消成本 |
| π5、π6 自动两轮 | 均完成 128,000 PPO 步并冻结；全策略探索和评价完成。停止状态 round_limit_reached |

π4 第一次 callback 工程失败的 27,200 是预算保守计费，不是已经完成的训练步数。失败目录和后续成功目录必须保留；π4 最终绘图失败也不意味着 PPO 失败，不能为恢复图片重训。

## 5. 最新 all_proposers_v1 结果

服务器根目录：`JIT/runs/campaign/all_proposers_v1`。

| 阶段 | 新候选数 | 新 root cells | 累计 root cells | 累计本轮新交互 |
| --- | ---: | ---: | ---: | ---: |
| 继承种子 | — | — | 1,689 | 0 |
| round_000：补 π0/π1/π3，再训练探索 π5 | 3,147 | 2,940 | 4,629 | 483,320 |
| round_001：π0～π5，再训练探索 π6 | 5,323 | 4,667 | 9,296 | 1,275,465 |

总新候选 8,470；总新增 root cells 7,607；两次成功 PPO 合计 256,000 步。未报告新工程重试。总覆盖口径仅是本 campaign 继承的 knee/lower/paired panels 加本轮数据，不是已经对全项目历史 Tube 做过全局去重。9,296 也不是几何体积或连续宽度。

第二轮各 proposer 相对轮首的新增量：π0 596、π1 732、π2 629、π3 860、π4 715、π5 406、π6 729。旧策略仍有大量收益，因此不能把 4,667 全算成训练 π6 的收益；探索安排/策略库也不完全相同，不能直接排名。

两轮末累计按 proposer 的 root cells：π0 1,083、π1 1,667、π2 1,396、π3 1,679、π4 1,449、π5 1,293、π6 729。各 proposer 数据量/轮数不同。报告中跨 proposer 高维单元完全不重叠的现象，尤其需要配合几何投影和分辨率敏感性解释，不能把所有独有单元视为宏观包线拓宽。

新策略的接续能力也不等于 proposer 贡献：

- π5 自身提出的 935 个候选中，π5 接续成功 850，π2 成功 882，集合成功 898；π5 独有见证 1。没有要求 π5 全面替代 π2。
- π6 自身提出的 738 个候选中，π6 成功 736，集合成功 738；π6 独有见证 0。说明本面板的新到达空间与新的独有接续能力是不同收益。
- 单次成功是见证，不是可靠成功率估计；同一轨迹上的状态高度相关。

## 6. 下缘结果和未解决事件问题

本轮排除同时带 physical_failure 标记的轨迹后，成功完整轨迹最低控制步峰值：π0 0.502206 m、π5 首批 0.505106 m/第二批 0.505080 m、π6 0.508714 m。旧 π4 配对实验最低值为 0.516082 m；本轮观测最低峰值下降约 1.39 cm，且最低来自 π0。

这不是每个 x 截面的下缘，更不是精确物理最低轨迹。逐截面最小值可能来自不同轨迹，不能连接为一条可执行轨迹；峰值采样发生在控制步，也不代表连续时间精确最大值。

发现 4 条轨迹同时 valid_landing=true 和 physical_failure=true：π5 首批 1、π5 第二批 2、π6 1。当前采集器先检查 down_events.valid_contact_seen，再检查 state.done，因此记录可能出现两种标记。下一版必须查明事件是否同一步、历史锁存或终点优先级差异；不要凭名字直接删除成功标签，也不要忽略它。生产训练适配器对 physical_failure 的处理与此需要核对。

这项是事件定义一致性审查，不是用户已经拒绝的浮点快照重放调查。现有报告保留，新增派生审计；若需重分类，说明影响范围和版本。

## 7. 耗时与预算

| 本轮工作 | 交互次数 |
| --- | ---: |
| PPO | 256,000 |
| 训练后小面板 | 142（63+79） |
| 前向采集 | 15,289 |
| 全策略接续评价 | 1,004,034 |
| 合计 | 1,275,465 |

继承记录开销 195,551；合计记录开销 1,471,016，但仍不含全部共享 bootstrap 和早期 seed 采集成本，不能叫全生命周期总成本。

共有 11 批新探索、352 条前向扰动轨迹、49,346 个策略—候选评价、413 个标签子进程；标签 batch_size=1，串行执行。不同进程需要启动、载入和编译准备。接续评价占交互约 78.7%，是主要工作量来源；没有完整墙钟账本，不能把交互比例直接当成时间比例。

训练日志内部 training/walltime 分别约 25.66、25.31 秒，不涵盖全部进程启动/编译/评价，不能说整次训练只花 25 秒。GPU正常工作，未见新失败重试。性能优化应优先针对接续评价和重复进程开销，不应仅削减 PPO 步数。

## 8. 现有代码框架及图片资产

| 文件（JIT/src/jit_dvgc 下） | 职责 |
| --- | --- |
| envelope_campaign.py | 有界迭代、成本、训练子进程、冻结、增量与停止 |
| campaign_bank.py | 初始 π2/π4 证据复用、旧策略轮流探索、逐轮绘图与 CSV |
| iterative_probe_training.py | 新 witnessed support 适配、reset、PPO 配置与 first landing |
| dense_tube.py / dense_tube_runtime.py | 采集—投影—分片标签—合并—图片 |
| acquisition/causal_jump.py | 固定起点真实动作扰动采集与轨迹 receipts |
| unified_continuation_shards.py / frontier_label_shard_runner.py | 独立进程接续与身份约束 |
| analysis/capability_tube.py | 物理投影和分辨率 |
| paired_discovery.py | π2/π4 回顾性同安排/成本对照 |
| result_bundle.py / result_publishing.py | 精简打包与自动发布报告分支 |

当前流程：先由各旧策略探索 → 聚合成功见证 → 构造 TRAIN support → 从上一冻结 Actor/normalizer 初始化、fresh critic/optimizer → PPO → 冻结新策略 → 新策略探索 → 下一轮。首次复用 π2/π4 后只补 π0/π1/π3；后续包含所有已有策略。每次每 proposer 32 组，固定均等分配，尚无按收益动态分配。

训练使用 20% 固定 x=2.5 起点、80% 已见证快照；phase/group 平衡，每阶段最多512个支持条目，当前多个来源有近期权重。完整上下文和 fresh continuation 时间语义保持显式，非任意物理状态 reset。

每轮 before_training/figures 与 figures 保存：每 proposer 的 PNG/PDF/SVG、all_proposers 图、pi_N_points.csv、all_points.csv、proposer_metrics.csv、x_slices_005m.csv、trajectory_receipts.json、figure_manifest.json。子 discovery 另有同候选集 evaluator 图和表。根目录有 campaign_progress 与 coverage_cost。

图是各 proposer 的累计真实到达投影，不是各 evaluator 在同一面板的胜率排名。空白 evaluator label 是未评价。保留历史 bank 版本，不能把 π6 未测的旧状态默认判失败或成功。

CSV 可用于重新绘图；图片和完整数据留服务器，精简发布通常不含深层图片。交接时已核对报告/表格，并未声称逐张检查了生产图片。需排版时读取服务器图或从 CSV 重画。保持 phase 分离的5 cm切片；物理去重仍是位置0.1m、速度0.1m/s、角度0.5deg、角速度2deg/s等既定定义，不把显示间距当物理网格。

## 9. 128k PPO 是否充分：尚未验证

128,000 是所有并行环境累计的交互步数；128环境下平均约每环境1,000步。它是warm-start小规模适应预算，不是已经测得的收敛阈值。

当前生产 checkpoint 只有0与128k，末尾有小TRAIN面板；没有32k/64k中间能力评价，也没有自适应延长。有限loss、完成训练和新覆盖都不能证明收敛。

下一版应在同次训练保存32k/64k/128k，对预声明固定TRAIN开发面板测成功接续，并给每个checkpoint相同小探索预算；所有评价成本入账。决定延长训练需要明确阈值、最大步数和跨seed验证，不临时无限追加。正常延长是否保留optimizer/critic状态，应与“从冻结Actor重新训练”区分并锁定。旧多轮不同支持的loss不能直接横向比较。

## 10. 学习型探索网络：用户提出，尚未实现

用户想要一个网络在不同基础π下学习扰动，使运动不明显失控、并到达该策略及整个集合此前未覆盖的区域。建议先做轻量条件残差策略：

`a_t = clip(pi_i(o_t) + delta_theta(o_t, pi_i(o_t), policy_condition, goal, history), action_limits)`

这是拟议结构，不是现有源码功能。先冻结基础策略库，训练探索网络；一个学习阶段内锁定版本；之后再用新增见证训练下一基础策略，交替更新，不同时任意改变全部策略导致无法归因。新policy条件如何泛化（policy ID、基础动作、embedding）需选择和验证，不能假定数字ID自动泛化。

建议目标：以整个集合的有效新增覆盖为主，单策略新颖性为辅；约束动作幅度、突变和明确失败风险。不奖励扰动幅度本身，不用强直立惩罚排除合理跳跃俯仰。仅奖励陌生状态会鼓励探索摔倒姿态。便宜的覆盖计数或预测新颖性可作引导，最终Tube入库仍需真实prefix/suffix见证。

工程选择需先做有界小试验：每步反馈残差或低频短段残差；扰动幅度和slew限制；多步延迟新增收益归因；阶段/目标条件；变化的覆盖基线如何按训练块冻结并版本化；网络训练本身和标签成本均计入。不要默默把探索器变成另一个拥有不受限权限的完整控制器。

保留固定扰动流程作基线，固定初始库、目标范围、分辨率和总预算，比较新增有效cells、几何范围、失败/无见证比例、到达与评价成本。新探索器增加的独立参数与推理时间也记录。残差探索是已有研究路线，不能将“动作相加”本身宣称原创。

## 11. 扩散模型：可选研究分支，不是下一轮默认任务

扩散模型可以条件生成一段扰动动作序列，表达多种时序模式；条件可包括完整状态、基础策略、目标区域和阶段。生成后必须在真实仿真执行再评价。直接生成状态轨迹不能证明可达。

它不会天然寻找物理边界：仅拟合已有成功轨迹可能重复旧行为；需要目标/新颖性引导、候选筛选或强化学习。现有状态很多但轨迹相关，不能把几千帧当成几千独立动作方案；完整动作序列数据是否满足训练需求应先审计原始工件。

建议比较次序：固定扰动 → 小型残差探索网络 → 有证据表明需要多模态时序生成时再做条件扩散扰动。比较总交互、墙钟、推理延迟和新增有效支持，不只展示生成轨迹好看。扩散模型训练、采样、调参和失败尝试成本不能免费。

参考原始论文（此前讨论中已查阅，仅作方法依据，不证明本任务效果）：[Residual Reinforcement Learning](https://arxiv.org/abs/1812.03201)、[Diffusion Policy](https://arxiv.org/abs/2303.04137)、[Random Network Distillation](https://arxiv.org/abs/1810.12894)。Diffusion Policy 的原论文任务与本项目不同；RND是可选新颖性代理，不是物理覆盖度量。

## 12. 接手开发顺序和验收条件

### P0：既有结果解释与事件一致性

- 先保留4条冲突receipts，核对事件顺序和success/failure优先级、采集器/训练/接续的一致性；给出受影响统计，原始报告不覆盖。
- 复用现有CSV，统一累计基线、分辨率敏感性、各phase截面和完整低峰值轨迹。图有空缺就保留，不填hull证明边界。
- 加入明确的start/end/elapsed计时：初始化、编译/首次step、采集、标签、训练、绘图、上传。不能以Brax内部walltime代替总耗时。

### P1：评价加速（未实现）

探索入库只需存在成功见证，可以按预声明顺序逐个evaluator尝试，一旦成功便停止该候选剩余评价。失败需要完成声明bank才记为no-witness；缺失/超时/工程失败保持未知。未运行label为untested，不能填0。

必须新建版本协议，保留完整评价旧路径；自哈希、catalog、policy、端点、种子、上下文身份、缓存、cost一并更新。不能简单改一个循环break却继续声称全策略矩阵完整。

固定小比较面板仍跑所有evaluator以衡量独有能力；训练支持的存在性见证不要求所有策略标签。验收：相同冻结bank/样本/确定键下，存在性结论与完整评价一致、未知不变负例、错误和重试计费一致，测实际墙钟收益。再评估扩大单进程分片或复用worker；先测显存，不盲目并行导致历史OOM复发。

### P2：训练长度与控制实验

实现多checkpoint保存、小预算TRAIN评价、明确最大步数和延长规则；保持训练与评价成本。用小生产验收后再扩大，不为补checkpoint重跑已完成128k结果。

### P3：轻量残差探索

先锁冻结基础库和固定扰动对照，再实现有界残差、有效新颖性收益、完整前缀记录。CPU测试只覆盖逻辑，需生产小试验和等预算比较。结果支持后再接入自动campaign。

### P4：扩散可选与论文实验

只有轻量网络基线建立且轨迹数据/多模态需求明确后，评估扩散探索。不把论文创新强行改为“用了扩散模型”。

## 13. 距离论文还缺什么

| 证据 | 当前状态 | 要补的实验 |
| --- | --- | --- |
| 工程闭环 | 两轮已运行 | 不需要为证明能跑再盲目重跑 |
| 多策略贡献 | 有开发数据 | 同一协议、同预算、区分proposer/evaluator与几何意义 |
| 训练的净收益 | π4小对照仅探索侧有优势，计训练未抵消成本 | 固定bank不再训练 vs 迭代训练，严格总成本对照 |
| 重复性 | 主要单条训练谱系 | 独立训练/搜索seed，按轨迹/祖先组统计，不拿帧做独立重复 |
| 下缘/几何扩展 | 有约0.5022m成功峰值及切片数据 | 完整轨迹、局部边界、分辨率敏感性；不宣称全局极限 |
| 学习型探索 | 未实现 | 固定扰动与残差探索对照；扩散可选 |
| 全成本 | 有局部记录及失败预算 | 补共享bootstrap、旧seed、开发评价和调参总账 |
| 最终评价 | 未使用final TEST | 方法锁定后独立评估；开发ACCEPTANCE不冒充未见测试 |
| bootstrap贡献 | 有历史流程 | 若主张必要性/效率，补up/down等对应消融 |

不能诚实给一个“还差百分之几”或保证再跑两轮就够RAL。现在是已得到有效开发证据、准备效率优化与正式对照阶段。精确物理边界证明不是本方法已具备或必须伪造的产物。

## 14. 数据角色、运行与交接纪律

TRAIN用于支持、探索和开发选择；CALIBRATION用于可选预测器校准；ACCEPTANCE一旦被用于调参就是开发证据；final TEST/JCE/JEL保持未打开。历史bootstrap叫test的split不因此变成未见最终测试。沿祖先/轨迹分组隔离，不仅按行随机分割。

原始run、checkpoint、attempt、summary保持不变；派生分析写新目录。代码/协议变化后不要绕过输入锁强行resume。不要git reset/clean/stash/rebase/force-push。git常规推送已授权；通过已连接GitHub接口可提交。开发修复与生产验收分开陈述。

旧生产命令记录见 [两轮运行说明](JIT_ALL_PROPOSER_CAMPAIGN_20260910.md)，它已运行完成，当前不是待执行指令。接手先实现和测试下一版再给用户新命令。29项相关CPU测试是此前代码验证记录，不是本次文档修改又运行29项测试。

建议接手阅读顺序：[AGENTS](../../AGENTS.md) → [PROJECT](../../PROJECT.md) → 本报告 → [CURRENT_STATUS](CURRENT_STATUS.md) → [路线](JIT_TRAINING_ROADMAP.md) → 远端原始报告 → 对应源码。老交接和指导文件已归档，时间较旧的“下一步”不是当前授权目标。
