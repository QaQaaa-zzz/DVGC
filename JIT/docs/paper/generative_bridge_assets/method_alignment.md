# 当前 E→G→π 方法与用户框图的逐项对齐（供中文论文初稿整合）

核验时间：2026-10-11 17:20 +08:00。工作树 `/home/qy/DVGC/runs/worktrees/jit-retention-first`，HEAD `0f4ed45d506fcfa816eb3250eea57c19e33405ed`。只读源码、冻结配置、已保存运行记录及用户 PDF；没有新增仿真、训练或修改主稿。本文件为给主写作者的方法素材，不是新的研究事实台账。

框图来源：`/home/qy/下载/JIT-core-method-network-icons.pdf`，单页，已提取文字并实际查看页面。图的主线是“可学习探索 E→扩散恢复教师 G→跳跃策略 π”，外环是学生学习进展反馈；九月的经验包线、历史多策略轨迹并集只可作为相关探索阶段与背景，不能取代这张图所定义的当前方法。

## 1. 写作时必须保留的三个层次

1. **完整方法与实际接线**：E 收集扰动后状态，G 搜索并验证恢复轨迹，学生以 PPO+示范+保留项学习，再把学生同根结果反馈给 E，并把成功轨迹反馈给 G。该完整顺序已在连续桥接系列执行，有 R73 的逐阶段记录；不只是待实现框图。
2. **本次 retention 子环节**：冻结 E73/G73 与原 π0，单独检验教师示范能否被学生吸收，以及旧能力是否保留。本次 BC、keep 扩充、权重对照与 self-visit 均不是重新跑完整 E/G/PPO 外环。
3. **尚待证明的目标**：图中的“Skill expansion / Robust generalization”应写成方法目标或待检验性质。已有开发集增益与损失并存，不能据此宣称完整外环比同预算基线更优、没有遗忘、扩散优于噪声搜索，或最终部署已成功。尤其连续系列的 `adopted=true` 只表示允许把完成的学生用于下一轮，`performance_gate=false`、`formal_adopted=false`，不是性能验收通过。[S1, A1]

## 2. 符号、输入与输出

| 符号 | 论文含义 | 真实实现接口与限制 |
|---|---|---|
| \(s_t\) | 仿真闭环状态 | 物理位置/速度、真实观测 FIFO、上一动作、阶段与着陆事件、随机状态等；不能只用 \((q,\dot q)\) 代替完整快照。 |
| \(o_t\in\mathbb R^{76}\) | 跳跃 Actor 与 G 的条件观测 | 3 帧×25 项真实 FIFO，加 1 项 jump_signal。每帧含重力方向、角速度、加速度、转向/髋/膝位置与速度、前后轮速度、上次四维动作、前向速度、障碍相对位置、根部高度和历史有效标志。名为 estimated_forward_velocity 的量在当前仿真实现中取真实根部前向速度，不能直接称全可部署传感估计。[S2] |
| \(\pi_{\theta_k}\) | 第 k 轮冻结采集/接续策略 | Actor 输入76维，三层256隐藏单元，tanh-normal 分布，执行输出4维归一化动作。学生 PPO Critic 输入106维特权观测，三层256。Actor 与 Critic 不是一个输入接口。[S2,S3] |
| \(E_{\phi_k}\) | 探索扰动策略 | 当前 R73 已执行邻域版：原始输入386维＝106维特权闭环状态＋16×17邻域条目＋8统计量；条目由12维相对物理坐标、4证据标志与1有效标志组成。条目 MLP 与 masked-mean 得到64维摘要，拼接为178维，再进入三层128 ELU Actor/Critic 头。Actor 给出四维潜在高斯均值及状态相关标准差，经 tanh 和0.25缩放得到四维扰动；独立 Critic 输出标量价值。旧106维探索器、旧256×3 Swish 实现不能冒充当前实际配置。[S4,A2] |
| \(G_{\psi_k}\) 或 \(\epsilon_\psi\) | 条件动作序列扩散模型 | 条件是固定来源 normalizer 归一化的76维 \(o\)，另输入噪声步和16×4带噪动作序列；网络输出同形状噪声估计。条件一维 U-Net，时间轴下采样/上采样、残差块、FiLM、GroupNorm；通道64/128/256。动作序列最终是归一化绝对控制动作，不是加到 π 上的 residual。[S5] |
| \(\tau^T\) | 经完整任务验证的教师轨迹 | 前16步 G 开环动作＋后续冻结 π 的闭环动作，保留每步真实 pre-observation 与 executed action；其成功不能记作 π 独立成功。[S6,S7] |
| \(D_T,D_K,D_G\) | 学生示范、保留锚点、G 的成功轨迹库 | 三种来源/用途不同；历史示范是监督资料，不进入 PPO 的旧行为 rollout。TRAIN 可以训练与反馈；used DEV 选模/监测；最终 TEST 不回流。[S7,S8,S9] |

R73 邻域四个证据标志的准确含义是：当前源 Actor 的成功、失败，历史某 Actor 的成功，历史已验证教师的成功。未知与矛盾标签不应变成失败。邻域图在本轮采集前冻结，不能读取本轮尚未产生的学习结果。[S4]

四维动作顺序为 `[steer, rear_wheel_drive, hip, knee]`，各维先限制在 [-1,1]，随后映射为对应执行器的位置/速度目标；髋膝为关键帧中心的绝对位置语义。不要把四维动作统称四电机扭矩。控制周期20 ms，物理子步5 ms。[S2,S10]

## 3. 可直接写入论文的方法段落：可学习探索

在第 k 轮采集阶段，我们冻结当前跳跃策略 \(\pi_{\theta_k}\)，利用独立探索器产生短时、有界的动作扰动，使系统到达不同的任务状态。归一化执行动作为

\[
 z_t\sim\mathcal N(\mu_{\phi_k}(x_t),\operatorname{diag}\sigma_{\phi_k}^2(x_t)),\qquad
 \delta a_t=0.25\tanh z_t,\qquad
 a_t=\operatorname{clip}(\mu_{\theta_k}(o_t)+m_t\delta a_t,-1,1),
\]

其中 \(m_t\) 表示本回合唯一扰动窗口，\(x_t\) 由特权状态和已冻结的历史邻域证据构成。当前连续协议每轮128回合，窗口起点从控制步0、5、10、15中调度，每次持续3步；其余时刻由冻结跳跃策略控制。扰动在动作空间施加，不修改质量、摩擦或成功判据。记录请求扰动和经动作裁剪后的实际扰动，避免把请求幅值误作真正执行幅值。[S1,S4,S11,A1]

探索器的 PPO 更新使用本轮新采集的潜在高斯动作、行为 log-prob 和固定行为身份；不存在把旧根样本伪装成 on-policy 数据的过程。由于奖励在同根学生评价完成后才得到，当前实现将回合末奖励回传给各有效扰动动作，取 \(\gamma=\lambda=1\)，并使用 PPO 概率比裁剪、独立价值损失和潜在高斯熵。已执行的 RSL 路径采用4个声明 epoch、clip=0.2、target KL=0.01；未执行的后续动作、padding、未知标签不进入 E 的更新。当前新协议的 uniform_episode_fraction=0，无20%均匀随机探索支路。[S4,S11]

这里“新颖且可学习”是探索目标，不是当前状态满足的先验保证。可学习性由后续教师/学生物理评价提供可观察反馈，不由探索器预测值或累计奖励直接判定。

## 4. 可直接写入论文的方法段落：生成恢复与教师验证

对于当前策略接续失败且评价有效的扰动后状态，条件扩散模型生成有限组候选恢复动作序列。设 \(A_0\in[-1,1]^{16\times4}\) 是成功轨迹中的真实16步动作片段，\(\bar\alpha_n\) 为余弦噪声日程，则训练目标为

\[
 A_n=\sqrt{\bar\alpha_n}A_0+\sqrt{1-\bar\alpha_n}\epsilon,\quad
 \epsilon\sim\mathcal N(0,I),\qquad
 \mathcal L_G=\mathbb E\!\left[\|\epsilon_\psi(A_n,\bar o,n)-\epsilon\|_2^2/64\right].
\]

实现使用100级余弦日程、20步 DDIM 采样以及 EMA 参数。每个根生成16条候选，与1条源策略独立接续控制组成17个世界；当前协议不含额外15条 colored-noise 候选。G 只在根状态条件化一次，随后执行其动作序列前16步（0.32 s），再交还冻结 \(\pi_{\theta_k}\) 直至完整任务终点：

\[
 a_{t+\ell}^{T}=\begin{cases} A^{(c)}_{\ell},&0\leq\ell<16,\\
 \mu_{\theta_k}(o_{t+\ell}),&\ell\geq16.
 \end{cases}
\]

桥接交接点不清空 FIFO、事件或控制状态；这不是逐步重规划的 diffusion MPC，也不是把教师前缀动作加到源策略输出上。[S5,S6,S11]

候选选择先以完整恢复成功作硬掩码，仅在成功候选集合中按全轨迹动作变化平方和最小、任务回报最大、候选编号最小的字典序选择：

\[
 c^*=\operatorname*{lex\,argmin}_{c:\,y_c=1}
 \left(\sum_\ell\|a^c_\ell-a^c_{\ell-1}\|_2^2,-R_c,c\right).
\]

第一项从真实根上一动作开始累计，不只度量16步前缀。完整成功集合为空时记为“有限搜索未找到解”，不证明该状态不可恢复；工程错误、未知或不完整结果另记。选定候选在相同完整17-world布局和相同候选序号下复核；主线采用搜索后一次固定赢家复核，本次 retention/self-visit 独立诊断采用搜索后两次固定赢家复核，不得将两者混写为相同重复协议。若同布局源策略复核成功或源标签冲突，则不能把该根计作教师新增恢复能力。[S6,A3]

成功采用 stable_forward_recovery：先有真实腾空与有效车轮接触，再满足声明的连续25控制步（0.5 s）恢复条件，且没有物理失败或超时。冻结 descent 配置包含前进速度至少0.5 m/s、侧倾不超过10°、俯仰不超过15°、两轮距地上界0.02 m及穿透限制。只达到第一次着陆、短暂存活或高回报均不能代替完整成功。[S12]

**时钟语义应单独写明**：R73 主线从保存的物理/历史状态进入 fresh-continuation，重新开始 episode/phase 行政计时，并清零恢复计数；它不是沿原回合时钟的无缝拼接。本次 retention/self-visit 则显式采用 preserve_snapshot_episode_context，并把真实 MuJoCo 时间作为独立 snapshot_time 恢复。两条路径都不能在 G→π 交接点再次重置。[S11,S13]

## 5. 可直接写入论文的方法段落：学生学习与保留

部署候选是独立跳跃 Actor，教师只用于训练阶段的数据采集和监督。我们从已验证教师完整成功轨迹提取真实的逐步 \((o_t,a_t^T)\)，包括生成前缀和源策略尾段，利用行为克隆使学生在自身观测接口下复现动作。同时从原始策略真实成功 TRAIN 轨迹采样保留锚点，抑制学生在这些状态上的动作偏移：

\[
 \mathcal L_{\rm demo}=\mathbb E_{(o,a^T)\sim D_T}
 \left[\tfrac14\|\mu_\theta(N_\theta(o))-a^T\|_2^2\right],\qquad
 \mathcal L_{\rm keep}=\mathbb E_{o\sim D_K}
 \left[\tfrac14\|\mu_\theta(N_\theta(o))-\operatorname{sg}[\mu_{\theta_0}(N_0(o))]\|_2^2\right],
\]

\[
 \mathcal L_{\rm student}=\mathcal L_{\rm PPO}
 +\lambda_D(u)\mathcal L_{\rm demo}+\lambda_K\mathcal L_{\rm keep}.
\]

\(\operatorname{sg}\) 为停止梯度，\(N_0\) 是固定 π0 的统计量。公式中 \(\mu\) 表示策略分布的确定性归一化动作（tanh-normal mode），不是未变换高斯均值。为避免重复书写归一化，论文也可把 \(\mu_\theta\circ N_\theta\) 合写为 \(\bar\pi_\theta\)。示范动作是实际执行的目标，不对多条不同成功模式的教师动作直接求平均。[S3,S7,S8]

连续主线每轮在新的 TRAIN 支持上收集128000条 PPO 转移；\(\lambda_D\) 由0.2线性降至0.05，\(\lambda_K=0.2\)，两类辅助样本 batch 各256。R73 实际 student PPO：学习率3e-5、clip=0.1、\(\gamma=0.995\)、GAE \(\lambda=0.97\)、全梯度上界0.75、128环境、unroll25。其原始实现以本轮 normalizer.count 增量作为示范权重退火进度，normalizer 在 PPO 阶段更新；不能把后续 retention 的“全部 normalizer 固定”追溯写成所有历史主线都成立。PPO 的 Critic、Actor、Adam状态、RNG与统计量按完整 learner 继续；每轮仿真从声明训练支持重新开始，不恢复上一轮末的仿真接触状态。[S8,S14,A4]

学生训练重置包含20%完整起跳起点与80%快照支持；pending_fraction=0.5控制快照支持内部待学习部分。完整任务起点的训练扰动是前三步，幅值0/0.1/0.25的混合概率为0.2/0.4/0.4；不能把 E 的0/5/10/15采集窗口误写成学生 PPO 所有回合都使用相同四窗口。[S8,A4]

保持项约束的是有限锚点上的动作误差，不是任务成功不下降的理论保证。教师状态分布上的监督误差低，也不保证学生独立运行时仍访问相同状态；因此必须分别评价教师恢复、学生吸收和相对 π0 的保留/丢失。

## 6. Learning progress 与双向数据反馈：已经实现的准确含义

E 的反馈来自相同 TRAIN 根上“训练前源策略标签→训练后学生独立标签”，不是直接把 G 成功当作学习成功。对于证据有效且实际参与学习的根，当前质量奖励包含：前0后1且学生允许连续采用时 +2；前0后0时 −0.1；施扰期间直接物理失败时整条奖励 −2。若源策略已在复核中成功，则取消把该根当作新增0→1的 +2。未知、未尝试、来源冲突不填伪阴性；teacher_success_bonus=0。[S1,S15,A5]

额外新颖项按当前冻结源 Actor 的物理格访问账本计算：

\[
 r_i^{\rm nov}=
 \begin{cases}
 \dfrac{0.02}{|C_i|}\sum_{c\in C_i\setminus S_k}\dfrac1{n_c},&C_i\setminus S_k\ne\varnothing,\\
 -0.02,&C_i\subseteq S_k,
 \end{cases}
\]

其中 \(C_i\) 是本次实际扰动经过的物理格，\(S_k\) 是该冻结源 Actor 先前访问账本，\(n_c\) 为本批对新格的重复访问数；只有合法反馈掩码内的回合进入更新。因而 novelty 是有限分辨率、有限采样的探索统计量，不是严格可达集增长或容量证明。当前 E 奖励也没有对完整旧能力集合施加统一回归惩罚，不能把它称为“自动保证无遗忘的 learning progress”。[S15]

学生→G 的虚线同样已执行，但其作用是构造训练数据，不是反向传播贯穿三个网络：G 成功轨迹库按“历史50%、本轮已验证教师25%、本轮获连续采用学生的独立成功25%”请求采样，空来源重归一化；来源内按祖先、轨迹、真实16步窗口依次抽样。失败轨迹、无真实 pre-observation、非 TRAIN/generator_train 分支、跨 reset 或不足16步窗口均不能偷偷补齐。G 更新为最多2000步离线去噪优化；当前 R73 使用 last_valid 的完整状态与 EMA 推理，旧固定 G-DEV MSE只作监测。早期 fixed_dev_best 是另一历史规则，不应写入当前主线。[S5,S9,A6]

E、G、π 之间没有可微仿真链的联合端到端梯度，实际是“冻结收集→教师筛选→学生优化→同根评价→E更新/G数据更新”的交替程序。R73 `explorer_feedback.json` 与 `explorer_update/metrics.json` 可证明确有反馈和 E 优化；这些记录证明箭头执行，并不证明箭头带来净闭环收益。[S1,A5]

对应框图雪花符号，建议正文使用以下冻结时序表，避免把“采集冻结”误解为“全程不更新”：

| 阶段 | 固定对象 | 允许优化对象 |
|---|---|---|
| 扰动采集 | π_k 与其统计量；本批 E 行为身份；历史邻域图；G | 无在线参数更新；先保存本批新行为轨迹。 |
| 教师搜索/复核 | π_k、E、G 参数与实际候选动作；来源/角色/物理协议 | 无更新；只执行有限候选和固定复核。 |
| 原主线学生训练 | E、G、固定 π0 保留教师与离线示范 | 学生 Actor、Critic、PPO优化器和运行统计；监督项加入实际 PPO 总损失。 |
| 学生同根评价与反馈锁定 | 学生、源策略、教师标签 | 无更新；评价结果不通过仿真反向传播。 |
| E更新、随后G更新 | 已采集行为、同根标签、数据角色与来源身份 | E及其独立Critic的PPO更新；随后G的去噪更新/EMA，不同时改学生。 |
| 本次纯BC诊断 | π0保留教师、E/G、学生normalizer/Critic、keep数据 | 仅学生Actor与fresh Adam；没有PPO rollout。 |

## 7. 本次 BC+keep 与 self-visit 的论文定位

建议置于“教师技能转移与保留诊断”或“机制验证/消融”，不要作为完整外环主结果的替代：

- **纯 BC+keep**：\(\mathcal L_B=\lambda_D\mathcal L_{demo}+\lambda_K\mathcal L_{keep}\)，没有 PPO 转移或 E/G 更新。原 π0 初始化、fresh Adam/RNG、normalizer与Critic固定，Adam1e-5、clip1、每来源batch256。完整状态保存与恢复可读性检查不等于已实现训练 resume。各节点独立评价原8个TRAIN和全部7个SOLVER_DEV；4个没有G解的DEV仍评价学生，老师参考记NA。[S16]
- **keep覆盖对照**：原8根G教案525帧不变，keep从1951帧扩展为6289帧，只纳入真实源成功TRAIN。它检验保留锚点覆盖，不检验网络容量。先前结果仍有成对旧成功丢失，不能把数值MSE或负梯度余弦当成容量不足证明。
- **权重对照**：固定525 demo/6289 keep，只把 keep/demo 从1/1改成1.5/0.5。已因固定B格保留警戒在100步停止；回退π0产生0损失不叫训练改善。应只与控制组相同100步比较。
- **self-visit诊断**：固定 coverage BC1000 的学生、原π0及G73，从全部原8TRAIN根沿学生真实独立运行到 j=4、8、12 捕获完整 post-step 状态。G仍生成16步，但只执行剩余 \(16-j\) 步，再交回π0，不拼入此前学生前缀，不清空FIFO/事件/RNG/时钟。终止状态不作为快照。查询仍是16候选+source，固定赢家两次完整批重放。[S17]
- **已完成机制事实**：24槽齐全；历史三次稳定失败的祖先18/30/55上有18j4、30j4、55j8、55j12四个 source000/winner111 指导位置。该事实支持“这些学生访问状态存在经重复验证的恢复指导”；不证明学生已学会，也不证明G比同预算Noise更好。[A7]
- **self-visited示范补充**：保留原525帧，加四条固定第二复核教师完整成功 suffix 的228帧，总753；固定keep6289、同架构/损失/系数/seed、fresh π0，只改变示范支持。8原祖先等权，祖先内轨迹等权，真实桥接前缀/π0尾段各半。旧数据的动作/观测不变，但祖先内采样质量重新分配；525/753不是旧样本采样概率。本次实际旧/新质量为19/24与5/24。[S18,A8]

截至本文件核验时，self_visit_demo已完成1000监督更新、正在1000节点评价，**最终候选、稳定三次标签和保留结论待填**。首个完整100节点尚无吸收或可靠保留改善：控制/补充后TRAIN均3/8，SOLVER_DEV由1/7到0/7。整段任务不能写作已成功验证“补充自访问示范解决分布偏移”。最终数字请由主写作者读取 `B_completed.json` 与完整报告后再填。[A8]

## 8. 原框图箭头的证据状态与建议图注

| 框图箭头/文字 | 实现和执行状态 | 当前论文写法 |
|---|---|---|
| π+E→Simulator→Perturbed state | 已执行；π与E行为身份冻结采集 | “有界扰动采集”，保留实效扰动与完整状态。 |
| Perturbed state→G→Recovery control→π | 已执行；16候选、H16、完整成功后固定复核 | “有限候选恢复搜索”，不称全域可恢复保证。 |
| Teacher data→π，Distill | 已执行；原主线为PPO+示范+keep，本轮有纯BC诊断 | 正文补出PPO和keep，不能让读者以为只有纯蒸馏。 |
| 教师成功→Denoising update→G | 已执行；只有符合角色/完整性条件的成功轨迹 | “成功轨迹条件去噪训练”。 |
| π→Denoising update | 已执行；被允许连续采用学生的真实独立成功数据 | 不把teacher composite当Actor数据，不表示可微联合更新。 |
| Learning progress→PPO update→E | 已执行；同根前后标签、novelty与质量项 | 与教师能救严格分离；“反馈已接线，净收益未因果隔离”。 |
| π0→π1→…连续迭代 | 连续系列实际运行到R73，R74工程失败停止 | 不写已完成200轮；adopted不等于formal adopted。 |
| Skill expansion / Robust generalization | 目标、有局部DEV证据及明显负例 | 可保留目标标题，图注声明最终效果需配对保留/增益及独立测试支持。 |
| 图右侧 External disturbance | 仅凭核心框图/代码不足以证明外力域泛化 | 留“[待补：外力/负载等非动作扰动的固定物理协议、基线与完整结果]”，不要把动作残差自动当任意外扰。 |

建议中文图注：**可学习探索、生成式恢复指导与单策略学习的交替迭代框架。实线表示控制执行或样本传递，虚线表示网络更新或TRAIN反馈。采集与教师验证阶段冻结当前跳跃策略；生成教师仅在有限窗口接管，随后由固定策略完成任务。学生的同根学习结果用于探索反馈，符合角色和成功条件的轨迹用于生成器更新。技能扩展与鲁棒泛化为待实证检验的目标；示范成功与学生独立成功分别统计。**

## 9. 建议保留的论文缺口占位

- [待补：完整E→G→π外环相对同总物理预算、同初始化的普通继续PPO/固定随机探索对照；独立训练种子与区间。]
- [待补：G16与同预算噪声候选在相同失败根、相同17世界及固定复核下的公平对照；不能由G自身成功推出优于噪声。]
- [待补：Learning progress反馈与仅novelty/教师奖励的消融；当前已执行箭头尚无隔离其因果收益的证据。]
- [待补：最终self_visit_demo三次TRAIN、全部DEV、旧成功丢失与新增成功的同更新对照；失败/歧义完整保留。]
- [待补：最终锁定TEST、真实外扰泛化和实际部署；不得把used DEV或多策略并集改名为独立holdout/单策略能力。]

## 10. 源码与冻结产物索引（每个方法结论的出处）

以下S编号均指本次读取的实际源码；文件路径绝对化，函数名用于精确定位。A编号是实际运行配置/收据，只核读工件，不表示本次重跑了物理实验。

- **S1** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/closed_loop.py`：`feedback_rows`、`continuous_decision`、`ClosedLoopRound._run`（约253–385行），冻结→采集→教师→学生→评价→E/G更新顺序。
- **S2** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/observation.py`：`observable_frame`、`actor_observation`、`privileged_observation`；同目录 `constants.py`：76/106维、动作顺序、dt。
- **S3** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/ppo.py`：`make_network_factory`（约83行），Actor/Critic三层256及tanh_normal；`policy_retention.py::action_mse` 为四维/批次均值。
- **S4** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/rsl_pulse.py`：`torch_policy`、`initialize`、PPO更新段；`neighborhood_network.py::NeighborhoodNetwork`、`neighborhood.py::FrozenNeighborhood`与`generative_bridge/neighborhood_history.py::freeze_history`：386维、masked pooling、证据时序。
- **S5** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/network.py::ConditionalUNet`；同目录 `diffusion.py::cosine_schedule`、`ddim_sample`、`make_train_step`、`train_incremental`。
- **S6** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/teacher.py::select_teacher`；`production.py::teacher_search`、`replay_teacher`、`quarantine_source_conflict`；`teacher_worker.py` 为persistent原布局执行。
- **S7** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/data.py`：`validate_trace`、`build_action_windows`、`export_student_demonstrations`；`student_demo_bank.py::build_student_demo_bank`、`stratified_weights`（主线new/history 50:50与本次8祖先均衡不同）。
- **S8** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/student.py::make_joint_student_trainer`、`trainer_from_config`；`campaign.py::train_arm`；`student_clock.py::demo_weight`。
- **S9** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/feedback_data.py::admit_trace`、`build_corpus`、`sample_corpus`。
- **S10** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/action_mapping.py::map_action`。
- **S11** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/pulse_exploration_runtime.py::collect/evaluate/update`；`generative_bridge/rollout.py::prefix_action`；`explorer_admission.py::validate_admission/training_mask`；`continuation_profile.py::FIXED`。
- **S12** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/descent_semantics.py::advance_descent_events/classify_descent_terminal`；冻结配置 `/home/qy/DVGC/JIT/configs/descent_stable_forward_0p5s.json`。实际训练奖励另由`reward_mode=original_all_phases`约束，不应直接把descent配置中的所有奖励系数写成完整主线奖励。
- **S13** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/unified_continuation_labels.py::fresh_unified_continuation_state`；`pulse_exploration_runtime.py`约573–587行；`unified_envelope_snapshot.py::restore_unified_envelope_snapshot`。
- **S14** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/learner_continuation.py::LearnerHooks`。
- **S15** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/rewards.py::feedback`；`discovery_reward.py::conversion_feedback`。
- **S16** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/generative_bridge/warmup.py::warmup_actor`；`retention_b.py::B_worker/select_B/warning_cells`。
- **S17** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/retention_self_visit.py::worker/classify_query`；`pulse_exploration_runtime.py::snapshot_capture_fields/export_evaluation_snapshots`；`generative_bridge/rollout.py::bridge_execution_horizon`。
- **S18** `/home/qy/DVGC/runs/worktrees/jit-retention-first/JIT/src/jit_dvgc/retention_self_visit_learning.py::prepare_demo/_stable_failures/_winner_trace`；`retention_b.py::prepare_self_visit_demo/audit_self_visit_demo`。

为避免数十次重复超长路径，以下定义实际已核目录 R73：
`/home/qy/DVGC/JIT/runs/experiments/bridge_four_onsets_continuous_20261008/series_0011_b1_resume_143/segment_0073/round_0073/`。

- **A1** R73/`production.json`、`acceptance.json`：四onset、G last_valid、continuous_learning、无性能采用门。
- **A2** R73/`collection/network_inventory.json`：inputs386、hidden128×3、ELU、Actor62216参数、Critic61313参数；本稿不把图标节点个数当网络宽度。
- **A3** R73/`teacher_worker/results.json`：候选、source控制、赢家与verification的真实记录。
- **A4** R73/`students/student/config.json`：PPO参数、辅助损失、完整learner继承、reset和training_action_pulse。
- **A5** R73/`explorer_feedback.json`、`explorer_update/metrics.json`：novelty总1.52、quality总约1.2、teacher_success_bonus0、实际optimizer_updates4；这些数仅用于核对反馈确实执行，不代表收益优越。
- **A6** R73/`generator_incremental/attempt_0000/generator_selection.json` 与 `update_2000/manifest.json`：last_valid/update2000、累计state_updates162500、EMA，old_dev_metric=monitor_only；0.5/0.25/0.25真实来源混合。
- **A7** `/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_self_visit_001/plan.json`、`report/query_matrix.json`、`report/independent_audit.json`。
- **A8** `/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_self_visit_demo_001/plan.json`、`demo/manifest.json`、`demo/admission.json`、`status.json`、`B_checkpoint_results.json`；截至核验尚在1000节点评价，最终结论须等待完成报告。

记忆仅用于定位历史方法入口；本文方法与时序结论以上述本次实际源码、PDF和运行产物为依据。没有借旧九月经验包线稿重写本次主方法，也没有任何理论收敛/稳定性证明声明。
