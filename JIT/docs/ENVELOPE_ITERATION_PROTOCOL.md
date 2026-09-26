# JIT 经验跳跃包线与延迟探索协议

当前版本说明：2026-09-12。本文描述已经实现的有限外循环及后续比较必须保持的契约。历史冻结协议不被本页追溯修改。详见[当前状态](CURRENT_STATUS.md)与[论文公式](paper/JIT_PAPER_DRAFT.md)。

## 1. 对象与终点

固定 `assets/orange_bike_4kg_horizontal.xml`、2kg载荷、完整x=2.5m近地起点、0.005s物理积分/0.020s控制。初始约3.1cm轮胎间隙已获用户接受。终点为首次有效落地，不能换成长期稳定恢复。两轮离地清隙先超过0.05m，随后任一轮接地、穿透不超过0.01m、无body contact且有限，才构成落地事件；并发physical_failure不能成为无冲突成功。

完整候选记为χ，包含qpos/qvel/ctrl、控制FIFO和有效计数、last action、事件/phase、RNG、时间规则及来源。准确候选键绑定state/context/acquisition身份。单独位置速度相同不支持跨轨迹拼接；相同物理网格也不支持拼接。

## 2. 四类集合

- A：具有真实动力学前向来源的候选记录。
- W：A中至少获得一个同上下文、无冲突成功续接见证的记录。
- P：A中尚没有有效见证的pending记录；其最新评价可为完整bank no-witness或unknown。
- S：训练/reset支持；可以包括W以及按声明配额加入的P，不能把全部S宣称为包线。

库标签为1（存在成功）、0（全部声明评价者完成且均失败）、unknown（未完、错误、冲突等）。P中的0并非物理不可行；新的策略允许重新评价。保存逐评价者与每次追加结果，而非只保存OR。

## 3. 当前残差探索

基础π_i的Actor与normalizer在探索训练块内冻结。基础Actor读76D三帧观测，残差Actor和其独立Critic读106D privileged observation。残差Actor为256×3 Swish，输出4D tanh Gaussian，当前归一化残差界为各通道0.15。

$$a_t=\operatorname{clip}\left(\pi_i(o_t)+d\odot\tanh u_\theta(z_t),-1,1\right).$$

动作顺序为steer/rear-wheel drive/hip/knee。本轮髋膝为keyframe-centered absolute position目标；转向position、后轮velocity actuator。requested与effective residual都记录，幅度限制不构成稳定性保证。

每个新source π重新初始化残差Actor/Critic/optimizer；同轮batch之间继续更新。没有显式policy ID/goal输入。random对照逐步均匀采样同界残差，不训练。当前只给learned臂训练共享后继π，因此该pilot是条件探索比较。

## 4. 到达奖励与见证准入分离

`arrival_novelty_v1` 的ledger针对当前π自身基线及本探索块已奖励单元，与全历史bank并集分离。每个满足真实prefix/context检查的新root cell每batch总奖励1，同格多候选分摊，信用绑定实际候选动作tick。父轨迹后来失败不抹掉此前真实到达。

奖励不需要当前bank成功。所有需要展示为经验包线的候选仍需真实suffix见证。保留的`witnessed_novelty_v1`是另一奖励协议，不冒充当前方案。探索PPO没有基础task reward、失败罚项或critic质量惩罚；基础critic只记录telemetry。

物理网格为root的12维位置/速度/欧拉角/角速度加phase，分辨率分别0.1m/0.1m·s⁻¹/0.5°/2°·s⁻¹。5cm仅是纵向候选/绘图间隔。当前候选每episode最多4个，实际帧采样，不插值造状态；预算和上限可能影响观测新颖性。

## 5. 同上下文续接

对每个候选恢复声明的物理与控制历史，用冻结评价π单独续接，**不带探索残差**。当前`fresh_continuation_v1`重置声明的episode/phase行政计时及累计return，保留物理与控制历史；该时钟变换是协议的一部分，不能写成完整MuJoCo内部状态逐位一致。

first-success可在首个无冲突见证后停止其他评价者，未测保持未测。仅当所有声明评价者完成失败才为0；错误、缺片、冲突不得凑成负样本。全矩阵用于单策略比较，存在性短路用于见证发现，两种统计不同。

用户已接受历史数值replay差异，不额外引入精确replay准入门槛；同时不声称精确replay已通过。4条历史forward冲突保留，派生隔离规则不回写旧结果。

## 6. pending学习与延迟重评

新跳跃Actor由上一π的Actor+normalizer初始化，Critic和optimizer重新初始化；仍用任务phase reward，不用探索奖励。

当前reset为20%完整固定起点 +80%快照；快照up/down各半。某phase存在pending时，该phase中pending质量为25%，其余75% witnessed；没有pending则仍全witnessed。pending按trajectory均衡。本pilot pending训练支持仅upstream，理论混合总pending质量约10%，不是25%也不是20%所有reset。

训练新π后冻结身份，在旧pending完整上下文上追加新成员续接结果。新见证进入W，旧有效见证保留，仍无见证者保留P。此反馈更新支持选择和下一轮source π，不跨多个PPO训练反向传播，也不对旧探索轨迹重标奖励后作为on-policy样本重放。

固定小TRAIN面板独立于pending更新，用于诊断；4/4成功不证明完整能力或旧pending学会。当前两轮128k均完成，延迟0→1均0。

## 7. 经验包线与可作出的数学结论

先确认准确记录W，再投影去重得到已见证cell集合。历史有效见证不删除时，同一身份、分辨率和协议下的累计集合只增不减，这是集合并的性质。它不意味着单π能力单调、真实连续区域内部都可行、随机成功概率有保证，或有限循环会遍历完整物理空间。

报告新到达、已见证、新suffix成功、训练支持、宏观相位几何和成本分别对应什么对象。不得把12维零重叠cell直接解释为宏观区域完全不重叠；不得将不同protocol结果未经去重直接相加。

## 8. 数据、预算、运行与复用

仅TRAIN驱动探索/训练；CALIBRATION用于可选预测器；已使用ACCEPTANCE为开发证据；最终TEST/JCE/JEL保持封闭。frame/candidate共享祖先，独立重复按训练与探索谱系设计。

声明模型、配置、source π/bank、种子、扰动、候选上限、endpoint、完整context、预算/停止条件及比较臂后再运行。缓存复用核验准确工件与协议身份，旧completed工件读取不强迫其源码等于当前HEAD。不得删锁跨变更resume。

成本按前向（active与padding分开）、所有suffix、PPO、panel、失败/重试、继承基线和bootstrap分项记录；actual、reserved上限、wall time和GPU时长不同。存在性减少suffix有用步不自动等于整条管线的墙钟提升。大批量工程验证结果不自动替换旧锁定科学协议。

每阶段保存完整轨迹/动作/快照/标签、所有PNG/PDF/SVG与replot CSV、SHA清单、INDEX和成本。本次文档更新不执行新GPU工作，不覆盖已完成all-proposer或旧失败记录。
