# JIT E→G→π 论文证据审查（2026-10-11）

本审查只读原结果，新增物理交互和参数更新均为 0。机器目录为同目录 `evidence_catalog.json`：包含 10 个可写主张、分母、数据角色、模型身份、精确原始路径、限制、建议正文/附录位置及待补占位。**不包含新 B_self_visit_demo_001 的中期或最终结果**，由主代理在该阶段结束后单独纳入。

可支持的主线表述是：冻结探索器产生困难状态；有限桥接搜索提供可复核教案；学生在部分训练状态和开发状态上表现出吸收；保持性仍有失败。**现有数据不支持“G 优于 Noise”“可靠消除遗忘”“最终学生已完成 sim2sim”**。

## 1. 图中各模块与数据身份

| 图中对象 | 已核验身份/职责 | 论文不能混写 |
|---|---|---|
| π0 | 旧 Phase U `transition_4988928` 的零更新绑定 `bridge_baseline/checkpoints/transition_0`；Actor `a06acbc8…`，normalizer `db84aad7…` | `jump_ori` 只命名冻结环境，不自动选 Actor；不是源 `9904128` 或 PhysX `01966080` |
| E73 | 固定 R73 explorer，state `3f77e061…`；在 π0 上执行一次 3 步残差，onset 0/5/10/15，动作增量上限各通道 0.25 | D1 没有更新 E，也没有本轮 E-v-random 消融 |
| G73 | 固定 R73 EMA generator，state `84fb92c8…`，累计 162500 更新；本轮 G 更新 0 | identity 中引用 π0 不等于已证明所有状态都与其尾部兼容 |
| 老师 | G 或有色 Noise 给 H16 前缀，随后交回同一冻结 π0；固定候选两次重放 | 复合老师成功不是单策略学生成功 |
| R5 | 历史 PPO 学生 `round_0005_student/transition_128000`，Actor `db2f3deb…`，paired normalizer `c9b365c7…` | 与本轮 BC 学生分开；V 不是本轮 BC 的结果 |
| coverage BC1000 | fresh π0 Actor-only BC 阶段选中，Actor `f1d1d46…`；normalizer/critic 冻结；推理文件 `b9817141…` | 总训练到 2000，阶段选模是 1000；不能用 last 2000 替代 |

全部完整 checkpoint/哈希均在 catalog.models 及对应阶段节点中。π0 与 R5 的现存 payload、E/G state、选中 BC 推理文件 SHA 均已重核。

数据角色：8 份教案及 keep 属 **TRAIN**；7 根 **SOLVER_DEV** 不进梯度，但反复用于选 checkpoint；64 道 **DEV** 用于保持性警戒和比较；V 是新 **DEV_CONFIRM**。均不能称最终 TEST。D1、吸收与自访问状态是延续快照；V/64 道 DEV 是声明自然起点。前者“学生独立”仅表示从该快照开始不调用老师，不表示完整自然起点执行。

## 2. 正文可放：R5 全任务开发确认，但同时报告旧损失

原目录：`/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/V_002`。原始 `*_main*/candidates.json` 的 1600 个计分条目与 `plan.json` 逐项对齐重计，另 400 个 padding lane 没有当作评价样本。

| 条件 | π0 | 固定 R5 | 新增 N01 | 丢失 N10 |
|---|---:|---:|---:|---:|
| A 名义初态、无脉冲 | 27/32 | 32/32 | 5 | 0 |
| B 随机初态、无脉冲 | 189/256 | 231/256 | 55 | 13 |
| C 名义初态、有脉冲 | 187/256 | 245/256 | 68 | 10 |
| D 随机初态、有脉冲 | 167/256 | 210/256 | 69 | 26 |

BCD 共 768 配对项：π0 543、R5 686；新增 192、丢失 49、净增 143。A 是**同一名义物理初态的 32 次执行**，不加入“独立能力”总数。onset 0/5/10/15 每组各 64 条；条件/祖先共享和固定 Actor 决定了推断范围。现成配对聚类 CI 可放附录，但不可外推到不同训练种子。

实际费用 800000 charged = 136198 active + 663802 padding，177.754 秒，更新 0。R5 保持 `best_dev_candidate`；`published_policy=null`。正文写“有限开发确认下的配对改善及遗忘”，不写组件因果提升、已采用或最终泛化。

## 3. 正文关键：D1 只支持同根互补，不能写 G 优势

完成结果为 **D1_003**，继承 D1_002 已完成 12 根及失败预留；D1_001 未运行，D1_002 OOM 不是被删除的试验。原入口：`.../D1_003/report_003/`；精确原根路径在 catalog.D1.root_table。

固定 E73 收集 TRAIN 128 和 SOLVER_DEV 32，共 160 个有效扰动后状态。π0 首次失败分别 37/128 与 8/32；只对入选失败根做足复核，不能把未复核失败称稳定失败。最后入选 16 TRAIN 稳定失败根、7 SOLVER_DEV 稳定失败根；另一个 solver-dev 初始失败根标签 0/1/1 被隔离。这是定向困难状态筛选，不是无偏任务成功率。

每方法每根 **16 候选 + 1 source，共 17 世界**；固定首次赢家，两次完整批重放；H16 后均交回同一 π0。Noise 参数 rho=0.8、scale=0.2。138 个原始结果 JSON（2346 lane 标签）及 status 重核通过，所有方法的赢家固定且资格复核与保存结果一致。

| 角色 | G-only | Noise-only | both | neither | ambiguous | 总根数 |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN | 3 | 5 | 5 | 2 | 1 | 16 |
| SOLVER_DEV | 1 | 1 | 2 | 3 | 0 | 7 |

因此 TRAIN 无歧义配对子集上 G 为 8/15、Noise 为 10/15；保留全部声明分母时 G 8/16，联合配对表中 Noise 10/16、歧义 1/16。**方法自身资格**则 Noise 是 **11/16**：根80在 G 批 source=0/1/0，但 Noise 批 source=0/0/0、赢家=1/1/1。论文主表建议保留完整五格及该脚注，不能把 10、11 两种分子混用。SOLVER_DEV 两方法均 3/7。

G-only 3 TRAIN + 1 SOLVER_DEV 提供“有限候选集合中互补”的见证；不证明这些状态只有 G 能解，不证明扩散必要，更不支持成功率优势。Noise-only 根更多。 **G-only只指G满足“冻结首次赢家+两次复核”资格**：额外原始审查发现TRAIN根26的Noise在首次和repeat1均无阳性，但repeat2候选11成功（该候选三次0/0/1）。它不满足资格，因此原配对格无需改写；却不能说Noise从未找到任何解。该阳性已加入catalog.D1.search_no_winner_replay_positive，原记录不改。8 个合格 G TRAIN 教案覆盖 4 个 onset、8 个祖先，共 525 帧；这个选后集合不能当总体8/8求解率。

### 实际成本必须与矩阵同时出现

| 成本范围 | charged | active | padding |
|---|---:|---:|---:|
| G 完成搜索+两次复核 | 72862 | 55235 | 17627 |
| Noise 完成搜索+两次复核 | 78472 | 48340 | 30132 |
| 共享收集、π0筛选、R5对照 | 22775 | 14167 | 8608 |
| G OOM 失败预留 | 6800 | UNKNOWN | UNKNOWN |
| D1 全阶段 | 180909 | 已知117742 | 已知56367 |

两方法候选数、最大 horizon、批形状及复核次数相同，**实际成本没有相同**。完成批 G charged 少 5610，但 active 多 6895；较少 charged 与 padding 差异有关。计入归属 G 的失败预留，G 线 charged 为 **79662**，超过 Noise 的 78472。历史 E/G 训练费用和提案计算时间未计入该比较；不能声称同实际预算或端到端效率优势。失败 reservation 的 active/padding 未知，不能塞入 active 或擅自删除。

## 4. 正文可放：老师→学生有限吸收，保留性负结果

原始阶段：`.../B_002/report_001`、`.../B_keep_coverage_003/coverage_report_0003`、`.../B_weight_ratio_001/weight_ratio_report_0001`。本审查对三阶段全部已达 checkpoint 的原 64-lane DEV candidates 和每节点 8 TRAIN+7 SOLVER_DEV 原结果 JSON 重计；固定警戒复核的原 JSON 也核对。

| 阶段选中点 | TRAIN 首次 | TRAIN 三次稳定成功/失败/歧义 | SOLVER_DEV | DEV ABCD，各16 | 首次新增/丢失 | 固定复核新增/丢失 |
|---|---:|---:|---:|---|---|---|
| 原 B，BC1000 | 4/8 | 4/3/1 | 3/7 | 16/10/15/10 | 20/7 | 13/3 |
| keep覆盖，BC1000 | 5/8 | 4/3/1 | 3/7 | 16/14/15/10 | 22/5 | 17/4 |
| ratio，BC100（被拒绝） | 2/8 | 本点未作所选学生稳定复核 | 2/7 | 16/11/14/13 | 22/6 | 17/7 |

coverage 连续完成 BC2000，按7根 solver-dev score 选择1000。全部节点0/100/500/1000/2000的 TRAIN 是0/3/3/5/5，SOLVER_DEV是0/1/2/3/2。稳定成功根68/37/10/26，稳定失败18/30/55，63为1/1/0。7根 DEV中的3个G已解根均学生成功；另4根G-NA均学生失败，必须报3/7，不能只报3/3。

增加 TRAIN keep：原1951帧扩为6289帧；新80次声明采集中61成功、19失败保留记录；合并87祖先编号，名义组20次执行仍仅1个名义初态，不能说87个独立物理初态。固定8 G教案525帧。首次旧损失7→5，但固定重复3→4，**可靠保持性改善未建立**。根30由历史学生稳定成功变为本阶段稳定失败，是“历史已吸收新技能丢失”，不同于π0旧损失。稳定旧损失 B_04/C_10/D_03/D_09 必须保留。

ratio只改变keep/demo系数1/1→1.5/0.5。同BC100，TRAIN3/8→2/8、SOLVER_DEV1/7→2/7、ABCD16/12/13/10→16/11/14/13。B格 π0/学生首次12/11，固定复核13/12；每格16道，一道6.25个百分点，超过固定>5pp警戒，故停止100/最多2000。阶段候选回退0更新π0，不能把回退后的0旧损失说成训练改善。不要用coverageBC1000对ratioBC100直接归因权重。

| 阶段 | 实际监督更新 | charged | active | padding | wall秒 |
|---|---:|---:|---:|---:|---:|
| 原 B_002 | 2000 | 313226 | 125803 | 187423 | 2125.230 |
| B_keep_coverage_003 | 2000 | 388062 | 136500 | 251562 | 2297.753 |
| B_weight_ratio_001 | 100 | 101416 | 37003 | 64413 | 735.687 |

这些阶段 PPO/E/G 更新均0。监督更新与物理费用分别计账。梯度冲突、MSE、reward只能放诊断附录，不能作为容量瓶颈或成功保持的因果证明。

## 5. 正文可放短诊断：自访问24 slots

原目录：`.../B_self_visit_001/report`；compact全部8张XY：
`/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/docs/evidence/retention_next_20261010/self_visit/INDEX.md`。

冻结coverageBC1000，8 TRAIN根各取j4/8/12，剩余桥接H16-j，之后交回固定π0，保持原总H16。全部24槽完整：G增量已复核12、source可救6、有限搜索未见解4、source歧义1、赢家歧义1。费用97801 =71025 active+26776 padding；wall705.106秒；学生/E/G/PPO更新0。

从历史学生三次稳定失败的18/30/55中，合格新教案分支仅：18j4、30j4、55j8、55j12，即**4 slots、3祖先**。它们是老师见证，不是学生改善。24槽也不是24个独立初态。`searched_no_solution`仅表示当前有限候选未找到见证，不等于理论不可解。

历史执行源码锁为 **90a3ddcb6c509d5a1836ecb16b32df6b8cb02dd0**；运行完成后代码改动不回写原身份。旧 capture 行的 `coordinates/phase/cell` 继承父根，只是祖先元数据；真实query使用snapshot/trace。独立审计读过80张原NPZ trace并核验物理状态、固定赢家、source标签、预算与祖先关系。本次重新读取审计JSON及矩阵；未重跑仿真。

## 6. 附录：sim2sim是独立工程参考，不能接成当前学生成功

同一Actor文件 `.../physx_jump_ori_from9904128_kd5_5ms_10m_20260928/training/checkpoints/transition_01966080/actor.npz`，现存SHA `d2648c4f33b3e191293d2a4281d9a95f475f2a2be09c14a0ad8f605235c2d1a0`。源MuJoCo checkpoint为seed820707的transition9904128，随后PhysX PPO 1966080 transitions；不是零样本迁移，也不是retention π0/R5/BC。

原数据：`/home/qy/ISAAC——SIM/worktrees/jump-landing-v2/sim2sim_v2/runs/no_speed_dual_landing`。原 `mjx_warm01966080/landing.json` 和 `physx_warm01966080/landing_lane_0.json` 各400条控制输入，均strict_landing_1p0s=true、contact_invalid_ticks=0；一秒保持分别于2.68秒和2.80秒达成，最长保持1.10秒和5.68秒。同一固定开发初态每引擎1个，并经过历史checkpoint筛选。原评分本会话已离线全量精确复算，本次又重读原结果及Actor SHA，无新增物理。

与retention成功口径不同：sim2sim持续1秒接触/姿态稳定、**无最低前进速度**；retention持续0.5秒且vx≥0.5m/s。起点x1.5m对x2.5m，标定接触/执行器不同；普通reset接触历史和batch-layout敏感性尚未解决。不能把两引擎结果加到retention成功率，不能声称sim2real。建议独立附录/工程可行性段，不作为E→G→π核心性能结论。

## 7. 草稿占位建议

- 【待补：最终固定学生与π0在独立自然起点TEST的配对成功、新增、旧损失、失败阶段及置信范围】。
- 【待补：相同实际交互与完整历史训练费用的G-v-Noise；另做E-v-random发现效率消融】。现D1的Noise是桥接对照，不是E对照。
- 【待补：多训练种子；严格区分π0旧技能、历史学生新技能、当前教案的新增/遗忘/重学】。
- 【待补：同一retention最终学生和π0在共同验收口径下的配对跨引擎验证】。
- 【由主代理单独填入B_self_visit_demo_001最终完成数据，不把本审查的旧coverage数字写成新阶段结果】。

不建议把“扩散必需”“可靠不遗忘”“全流程自主持续提升”“学生跨引擎泛化”写进摘要结论。主文可以完整报告有限证据链及负结果；附录保留全部根/槽位、原始费用、身份哈希和评分口径。
