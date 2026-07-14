# DVGC：Phase-Conditioned Empirical Recoverability Tubes for Single-Track Robot Jumping

## 论文核心版 v23（最终整合与复现定稿）

> 本文档是面向SCI小论文的方法与实验母版。
> v17继续作为工程实现规格；本版保留论文所需的完整定义、算法链条、实验问题和可复现接口，同时删除不必要的工程展开。
> 论文核心始终为：
> **Survival Bottleneck → Event-Anchored Backward Bootstrap → End-to-End Physical-Belief Recoverability Certification → Tube-guided RSI → natural-start完整跳跃验证。**

---

# 修订判定说明

本版没有直接照搬外部修改稿，而是作出以下技术判断：

1. “Viability显式加入阶段one-hot和连续进度”合理，予以接受；但必须保留原有PolicyState \(b_t\)，否则无法研究same physical / different belief。
2. “全部改为只进入下一阶段Tube”只部分合理。它适合作为Bootstrap冷启动目标，但不应取代最终Recovery定义，否则论文学习的是局部阶段可达性而不是经验可恢复性。
3. 因此采用双层目标：递归链式目标负责训练启动和阶段扩展，最终Recovery目标负责正式Tube认证、RSI核心集合和论文主结论。
4. Stable Baselines3和MuJoCo属于实现选择，不写入方法名称或理论贡献；若实际代码使用SB3，可在Implementation中报告。
5. “阶段估计滞后”是合理的部署风险，但默认单头Actor并不会进行四个策略之间的硬切换。应通过软阶段概率、历史观测、Filter滞回和训练期延迟随机化降低敏感性，并按控制周期而非任意毫秒值设计实验。
6. “策略更新后低频增量重标注”合理。重标注不应每个PPO迭代执行，也不应只使用固定周期；采用策略KL、固定评估集退化和最长标签年龄共同触发，并用固定branch预算优先刷新边界、高使用率和阶段连接状态。
7. 若未来采用Multi-head扩展，不接受在动作输出端进行朴素线性soft mixture。相邻阶段可能需要方向相反的物理控制量，直接平均会造成动作抵消。主方案继续采用单一共享Actor，并将连续阶段概率作为内部非线性特征。
8. Beta统计必须给出先验、后验和分位数定义。默认采用均匀弱先验\(\mathrm{Beta}(1,1)\)和\(5\%/95\%\)后验分位数，即90%等尾贝叶斯可信区间；不将其误称为频率学派置信区间，也不声称\(\mathrm{Beta}(1,1)\)是唯一“无信息”先验。
9. 重标注优先级同时使用Beta区间宽度和Viability Ensemble方差。前者反映有限branch样本带来的经验统计不确定性，后者反映函数逼近的模型不确定性；二者不可相互替代。
10. Tube质量不只报告AUC。必须在独立审计分支上定义保守Tube precision、recoverable recall、candidate-mass coverage和calibration，并报告Chain成功但Final Recovery失败、以及Final Recovery未经过认证入口的两类链式误差。
11. 对比稿在三处补丁的表达上更简洁，但仍存在soft mixture前后矛盾、将Beta(1,1)绝对称为“无信息先验”、以及仅保留Beta宽度而删除模型不确定性的问题。本版吸收其清晰表达，但保留统计和物理上的必要区分。
12. 低可恢复状态集合不再记为\(N\)，避免与branch次数\(N_z\)冲突；统一记为\(\mathcal T_{\mathrm{emp}}^-\)。物理Failure集合仍记为\(\mathcal F\)，二者不得混用。

---

# 0. 方法定位与范围

## 0.1 要解决的问题

单轨双轮机器人无法像四足或静态机构一样在障碍物前停稳、调整姿态后再起跳。机器人必须在持续前进、动态平衡和有限起跳窗口内完成压缩、离地、飞行、落地和恢复。直接使用PPO从自然起点训练时，早期策略通常在到达起跳窗口前已经失稳，因此很少访问飞行、落地和恢复状态，形成明显的 **Survival Bottleneck**。

本文不以精确参考轨迹模仿为核心，也不要求策略严格跟踪一条CoM轨迹。粗略CoM模型只负责给出物理合理的候选搜索范围；真正决定哪些状态值得用于课程训练的是当前策略在这些状态上的**最终可恢复概率**。

## 0.2 论文核心贡献

1. **Event-Anchored Landing-first Backward Bootstrap**
   针对单轨机器人难以从自然起点探索到成功跳跃的问题，从Landing开始依次扩展到Flight、Takeoff和Approach。Bootstrap阶段采用“进入下一已认证入口集合”的递归链式目标解决早期零正样本问题，但阶段完成和正式认证仍以最终Recovery为准。

2. **Phase-Explicit Physical-Belief Empirical Recoverability Tube**
   将真实物理状态、部署策略内部belief、真实阶段one-hot、阶段连续进度和任务条件共同纳入增广状态；通过冻结策略后的端到端分支rollout估计有限时域最终Recovery概率，构建阶段显式、策略条件化的经验可恢复Tube。

3. **Lag-Robust, Posterior-Calibrated Tube-guided RSI与Budgeted Relabeling**
   使用Beta后验可信界划分高可恢复、边界、低可恢复和证据不足状态，并用经验区间宽度与模型不确定性共同驱动增量重标注；通过连续阶段条件、阶段输出延迟随机化和延迟消融验证部署侧鲁棒性。

## 0.3 不作为本文核心贡献的内容

以下内容可以用于实现或扩展实验，但不应继续扩张为独立贡献：

- hierarchical RL或multi-policy架构；
- MPC/trajectory optimization teacher；
- residual policy；
- 跨domain worst-k、CVaR等风险统计；
- Isaac-specific Tube重新训练；
- 完整实机部署系统。

---

# 1. Introduction

## 1.1 背景

动态跳跃是单轨机器人实现越障和连续运动的重要能力，但其训练难度不仅来源于跳跃动力学非线性，还来源于训练早期的生存时间过短。机器人必须持续保持侧向与俯仰平衡、维持前向速度并在有限空间内完成起跳。如果策略在Approach阶段提前摔倒，就无法获得Takeoff、Flight和Landing阶段的有效经验。

参考轨迹或空间guideline能够提供运动意图，但固定参考无法回答一个更直接的问题：

> 对当前策略而言，哪些中间状态确实能够最终恢复到稳定骑行？

因此，本文将课程状态选择从“接近参考轨迹”转化为“具有经验可恢复性”。

## 1.2 研究问题

本文围绕以下问题展开：

- RQ1：Event-Anchored反向Bootstrap能否突破自然起点训练中的Survival Bottleneck？
- RQ2：端到端经验Recoverability是否比固定CoM corridor、阶段随机初始化或Chain-only集合更有效地指导RSI？
- RQ3：相同真实物理状态下，不同内部belief是否会导致显著不同的动作与恢复概率？
- RQ4：递归Bootstrap、最终Recovery认证、显式阶段条件和Beta统计设计分别贡献了什么？链式目标会产生多大局部成功—最终失败误差？
- RQ5：策略能否抵抗阶段估计滞后和传感器误差，并以可控的增量重标注成本维持Tube precision、coverage与calibration？

---

# 2. Related Work

建议分为四部分，避免将工程模块分别写成独立研究线：

1. **Reference-guided and imitation-based locomotion**
   讨论迭代参考、稀疏关键姿态、空间guideline和parkour-style reference learning。指出这些方法主要优化“参考表达或参考跟踪”，而本文优化“当前策略条件下的可恢复状态选择”。

2. **Curriculum learning and reset-state initialization**
   讨论阶段课程、backward curriculum、reverse curriculum和中间状态初始化。本文区别在于：reset状态不是仅按时间相位或几何距离选择，而是依据最终Recovery的经验概率选择。

3. **Viability, recovery and safe reinforcement learning**
   讨论可行域、可恢复集、failure recovery和value-based safety。本文不学习长期安全不变集，而是在有限时域、当前策略和跳跃阶段下估计经验可恢复性。

4. **Asymmetric and deployable reinforcement learning**
   训练Critic和Viability可读取特权信息，但Actor只能读取可部署传感器及其内部belief。Estimator和跨引擎接口作为实现支撑，而非额外核心算法。

---

# 3. Problem Formulation

## 3.1 机器人、任务与动作

机器人由浮动基座、前后轮、转向关节、hip和knee组成。策略动作维度为4：

$$
a_t=[a_t^{hip},a_t^{knee},a_t^{steer},a_t^{drive}]\in[-1,1]^4.
$$

动作分别映射为：

- hip位置目标；
- knee位置目标；
- steering位置目标；
- rear-wheel速度目标。

低层显式执行器根据动作目标、关节状态和电机动态计算力矩。Actor在50 Hz更新命令，执行器和物理引擎在更高频率更新。

任务为机器人从自然起点持续骑行，越过给定高度和宽度的台阶或平台，并在平台或目标地面上恢复稳定前进。论文主实验可使用一个标称障碍条件，泛化实验再改变障碍高度、宽度、初始速度和目标速度。

## 3.2 四个控制阶段与目标集合

控制阶段为：

$$
\phi_t\in
\{\mathrm{Approach},\mathrm{Takeoff},\mathrm{Flight},\mathrm{Landing}\}.
$$
Recovery不是第五个Estimator分类阶段，而是成功目标集合 \(\mathcal G\)。

| 阶段 | 起始事件 | 结束事件 |
|---|---|---|
| Approach | 自然起点或阶段reset | 进入起跳触发窗口 |
| Takeoff | 进入起跳窗口 | 前后轮均确认离地 |
| Flight | 双轮离地 | 任意轮首次有效着陆接触 |
| Landing | 有效着陆接触 | Recovery条件连续满足 |

有效接触满足：

- 接触几何属于前轮或后轮；
- 接触对象属于允许的地面或平台顶面；
- 接触法向方向满足阈值；
- 法向力超过最小阈值；
- 接触持续时间超过确认时间；
- 不存在车架、轮侧或障碍物侧面异常碰撞。

Recovery集合可定义为：

$$
\mathcal G=\left\{
\begin{aligned}
&|\mathrm{roll}|<r_{\max},\quad
|\mathrm{pitch}-\mathrm{pitch}_{terrain}|<p_{\max},\\
&|\dot r|<\dot r_{\max},\quad
|\dot p|<\dot p_{\max},\\
&v_x>v_{\min},\quad
|\mathrm{heading\ error}|<h_{\max},\\
&\text{至少一轮有效接触且无异常碰撞}
\end{aligned}
\right\},
$$
并要求连续保持 \(T_{rec}\) 秒。

## 3.3 Failure、terminated与truncated

明确失败集合 \(\mathcal F\) 包括：

- 车架、电机壳体或其他禁止部件触地；
- roll或pitch超过失稳阈值；
- 明显后退或朝错误方向运动；
- 障碍物侧碰后速度接近零且无法继续完成任务；
- 已越过允许起跳区域并在剩余时间内物理上不可恢复。

环境返回：

- `terminated=True`：达到Recovery或明确Failure；
- `truncated=True`：达到统一时间上限，但没有发生明确成功或失败。

普通MAX_STEPS属于truncated，不属于Failure。

## 3.4 可部署观测和Actor

部署Actor不能读取特权物理状态。定义控制边界上的部署观测：

$$
o_t^{dep}=
\left[
H_t^o,\,
H_t^a,\,
\hat p_t^{phase},\,
\hat p_t^{contact},\,
\rho_t,\,
\kappa_t,\,
\hat c_t,\,
u_t^{cmd}
\right],
$$
其中：

- \(H_t^o\)：控制频率传感器历史；
- \(H_t^a\)：动作历史；
- \(\hat p_t^{phase}\)：估计阶段概率；
- \(\hat p_t^{contact}\)：估计接触概率；
- \(\rho_t\)：阶段连续进度；
- \(\kappa_t\)：估计置信度；
- \(\hat c_t\)：障碍物估计及不确定性；
- \(u_t^{cmd}\)：目标速度等任务命令。

随机Actor为：

$$
a_t\sim\pi_\theta(\cdot\mid o_t^{dep})
$$
分支评估和部署评估使用确定性均值动作：

$$
a_t=\mu_\theta(o_t^{dep})
$$

## 3.5 Phase-Explicit Physical-Belief增广状态

Viability使用的增广状态定义为：

$$
z_t=\left(x_t^{priv},b_t,e_{\phi_t^{true}},\rho_t^{true},c_t^{true}\right),
$$

其中：

- \(x_t^{priv}\)：真实机器人动力学状态、接触、传感器内部状态和执行器内部状态；
- \(b_t\)：GRU hidden、Phase Filter状态、估计阶段/接触概率、Actor观测历史、动作历史及任务估计；
- \(e_{\phi_t^{true}}\)：由仿真事件真值确定的四阶段one-hot编码；
- \(\rho_t^{true}\)：由真实事件锚点计算的阶段连续进度；
- \(c_t^{true}\)：真实障碍物条件和任务命令。

阶段信息必须显式输入Viability Network，而不能只隐含在\(x_t^{priv}\)或\(b_t\)中。这样既保留Physical-Belief机制，又使网络能够区分相近物理状态在不同阶段下具有不同恢复含义的情况。

Actor仍只能读取可部署观测，不得读取真实阶段、真实进度或任务真值。真实阶段仅用于训练期Viability标注、Tube切片和Oracle消融；部署控制使用Estimator输出的阶段概率和过滤进度。

## 3.6 两类有限时域目标：递归Bootstrap与最终Recoverability

论文需要区分两个不同对象，不能混为一个标签。

### 3.6.1 递归阶段Bootstrap目标

阶段顺序为：

$$
\mathrm{Approach}\rightarrow\mathrm{Takeoff}\rightarrow\mathrm{Flight}\rightarrow\mathrm{Landing}\rightarrow\mathcal G.
$$

对阶段\(k\)定义下一阶段已认证入口集合\(\mathcal C_{k+1}\)。递归阶段成功事件为：

$$
\mathcal E_k^{chain}=\{\tau_{\mathcal C_{k+1}}<\tau_{\mathcal F},\ \tau_{\mathcal C_{k+1}}\le H_k\}.
$$

Landing阶段的下一目标直接取最终Recovery集合\(\mathcal G\)。该目标用于早期课程训练、候选扩展和阶段mastery判定。

### 3.6.2 最终端到端经验可恢复性

论文核心Tube仍定义为当前冻结策略从增广状态出发，在有限时域内到达最终Recovery且先于Failure的概率：

$$
\bar V_H^{\pi,\eta}(z)=\Pr_{\xi\sim p_\xi,\nu\sim p_\nu}(\tau_{\mathcal G}<\tau_{\mathcal F},\ \tau_{\mathcal G}\le H\mid z).
$$

递归目标回答“当前阶段能否可靠进入下一已认证入口”，最终Recoverability回答“从该状态能否完成全部剩余跳跃并恢复行驶”。前者是Bootstrap工具，后者才是Empirical Recoverability Tube的正式定义。

不同belief必须物化为不同增广状态\(z\)，不能把belief perturbation混入未来噪声。

---

# 4. Method

## 4.1 总体算法

整体流程分为Bootstrap层与Certification层：

1. 使用粗CoM模型和事件锚点生成物理合理候选；
2. 从Landing开始，以Landing到Recovery训练基本恢复能力并认证Landing入口集合；
3. 依次使用Flight到Landing入口、Takeoff到Flight入口、Approach到Takeoff入口的递归目标扩展策略能力；
4. 每完成一个阶段，冻结当前Actor、Estimator和Filter，对候选增广状态进行端到端Recovery分支rollout；
5. 通过Beta后验估计最终Recoverability，并形成经验高可恢复、边界、低可恢复和未知集合；
6. 训练Phase-Explicit Physical-Belief Viability Ensemble；
7. 使用端到端经验认证的\(T_{\mathrm{emp}}^+\)和边界集合B指导RSI；
8. 策略更新后，对高使用率、边界状态及阶段连接处重新标注；
9. 最终仅从自然起点评估完整跳跃成功率。

递归阶段集合用于启动和扩展课程，但正式RSI的高权重核心集合优先采用端到端Recovery已认证状态。

## 4.2 Coarse CoM Guidance

粗抛体关系只用于确定候选区域：

$$
z_{com}(x)=
z_0+x\tan\alpha-
\frac{g x^2}{2v_x^2\cos^2\alpha}.
$$
由标称速度、障碍物高度和净空裕量得到：

- 起跳位置范围；
- apex位置范围；
- 落地点范围；
- 最低净空范围。

CoM Guideline不进入最终奖励，不要求Actor跟踪固定曲线。

## 4.3 Event Anchors与候选状态

定义事件锚点：

- takeoff window；
- dual-wheel airborne；
- apex neighborhood；
- first valid contact；
- recovery window。

候选构造：

| 阶段 | 候选来源 | 关键变量 |
|---|---|---|
| Landing | post-impact稳定接触、pre-impact低高度下降、短仿真impact快照 | clearance、\(v_z\)、pitch、pitch rate、roll、wheel speed |
| Flight | 已有rollout飞行快照、结构化扰动、粗反向proposal | flight progress、CoM速度、姿态、关节构型 |
| Takeoff | Approach/Takeoff真实快照、压缩状态扰动、接触模式扰动 | 障碍距离、前向速度、hip/knee压缩、前后轮接触 |
| Approach | 自然起点rollout、Takeoff邻域前向回溯状态 | 距离、速度、平衡裕度、任务估计误差 |

Candidate Validator至少检查：

- 几何穿透；
- 关节限位和速度范围；
- 阶段与接触模式一致性；
- 轮地距离和接触对象；
- 短时松弛后是否立即爆炸或异常；
- 传感器、执行器和PolicyState是否完整可恢复。

候选采样范围应在配置表中给出，并通过消融或敏感性分析说明不是依赖单个手工状态。

## 4.4 Event-Anchored Landing-first Backward Bootstrap

所有阶段共用同一个Actor，不训练四个独立策略。默认使用单头phase-conditioned Actor；Multi-head仅作为出现明确阶段梯度冲突后的结构消融。

### Stage A：Landing → Recovery

Landing阶段先学习基本缓冲和恢复，再以最终Recovery成功率认证Landing入口集合\(\mathcal C_L\)。

### Stage B：Flight → Certified Landing Entry

Flight阶段的Bootstrap成功事件为在Failure前进入\(\mathcal C_L\)。进入后继续rollout到Recovery，并分别记录链式成功与最终Recovery结果。

### Stage C：Takeoff → Certified Flight Entry

认证Flight入口集合\(\mathcal C_F\)后，Takeoff阶段以进入\(\mathcal C_F\)作为Bootstrap目标。

### Stage D：Approach → Certified Takeoff Entry

认证Takeoff入口集合\(\mathcal C_T\)后，Approach阶段以进入\(\mathcal C_T\)作为早期课程目标，并逐渐提高自然起点比例。

每个固定评估集同时报告：

1. 阶段链式成功率；
2. 端到端Recovery成功率。

课程早期可依据链式成功率LCB推进，但阶段完成和论文结果必须同时满足端到端Recovery门槛。

## 4.5 Deployable Phase Estimation

### 4.5.1 Streaming Estimator

单帧可部署传感器输入Streaming GRU，输出：

- 前后轮接触概率；
- Approach、Takeoff、Flight、Landing四类阶段logits。

训练损失：

$$
\mathcal L_{\eta}
=
\mathcal L_{BCE}^{contact}
+\lambda_{\phi}\mathcal L_{CE}^{phase}.
$$
Estimator按episode划分训练和验证序列，使用burn-in、padding mask和概率校准。

### 4.5.2 Phase Filter

不可导FSM负责：

- 接触/离地时间确认；
- 阶段滞回；
- 短暂反弹处理；
- 连续阶段进度；
- 事件化置信度。

落地后的短暂bounce不能永久锁死Landing，可在Landing内部允许短暂airborne状态，或在接触未达到稳定确认前返回Flight。

### 4.5.3 Phase-Lag-Aware Policy Training

阶段估计滞后是部署侧风险，但默认Actor是一个共享的连续策略，不根据\(\arg\max \hat p_t^{phase}\)在四个独立控制器之间硬切换。因此，延迟不会必然导致“整段执行错误阶段动作”，但可能使动作调节晚于真实接触事件。

为隔离该风险，训练时对Estimator输出引入可控的stale-output过程：

$$
\tilde p_t^{phase}=\hat p_{t-d_t}^{phase},\qquad
\tilde p_t^{contact}=\hat p_{t-d_t}^{contact},\qquad
\tilde\rho_t=\rho_{t-d_t},
$$

其中\(d_t\)以episode级或短时间块级方式从训练分布\(\mathcal D_{lag}^{train}\)采样，并允许输出保持、偶发丢帧和置信度下降。Actor接收\((\tilde p_t^{phase},\tilde p_t^{contact},\tilde\rho_t)\)，但仍同时保留原始传感器历史、动作历史和任务观测，使阶段估计成为辅助条件而不是唯一控制开关。

由于Actor控制频率为50 Hz，一个控制周期为20 ms。主实验优先采用：

$$
d\in\{0,1,2,3\}\ \text{control ticks}
\equiv
\{0,20,40,60\}\ \mathrm{ms}.
$$

若仿真接口保留高频传感器时间戳，可额外测试10 ms和50 ms；但应说明命令只在下一控制边界更新，因此其有效控制延迟可能被20 ms周期量化。

为避免训练只适配固定延迟，建议设置：

- nominal训练：\(d=0\)；
- lag-randomized训练：\(d\sim\mathcal D_{lag}^{train}\)；
- out-of-distribution测试：使用超出训练范围的更大延迟；
- Oracle phase仅作为性能上界，不进入部署Actor。

默认方案不采用Multi-head动作切换，而是将连续阶段概率向量作为单一共享Actor的输入，使网络内部的非线性层学习平滑的阶段条件控制流形。

若后续进行Multi-head扩展，需要同时避免两类错误：

1. 用\(\arg\max \hat p_t^{phase}\)硬切换单一head，会把小幅估计抖动放大为离散动作突变；
2. 在动作输出端进行朴素线性soft mixture，也可能产生物理控制量抵消。

设相邻阶段head输出分别为\(a_t^{F}\)和\(a_t^{L}\)，简单混合为：

$$
a_t=
\hat p_t^{F}a_t^{F}
+
\hat p_t^{L}a_t^{L}.
$$

当Flight需要下压车头而Landing需要抬头抗冲击时，\(a_t^{F}\)与\(a_t^{L}\)可能方向相反。在\(\hat p_t^{F}\approx\hat p_t^{L}\approx0.5\)的模糊区间，二者可能相互抵消，产生接近零但物理上无效的控制量。

因此，主方案采用连续阶段概率作为共享Actor的隐式特征，由非线性网络学习平滑的非线性阶段条件转换。本文可将这一机制描述为soft manifold transition，但不把它扩张为独立算法贡献。若必须使用专家结构，更稳妥的扩展是特征层门控、共享基础动作加阶段残差，或带滞回的稀疏专家选择，而不是直接平均最终动作。

## 4.6 Augmented Snapshot与Belief Variants

一个增广Snapshot必须绑定：

- 完整物理状态；
- 传感器与执行器内部状态；
- Estimator hidden和Filter状态；
- Actor观测历史和动作历史；
- 任务估计及不确定性；
- 版本元数据。

相同物理状态可以对应多个不同belief：

$$
z_i=(x^{priv},b_i,c^{true}).
$$
Belief variants通过以下方式生成：

- IMU bias、scale和噪声；
- 编码器量化或丢包；
- 传感器延迟；
- 轮速滑移；
- 任务估计误差。

修改历史后必须从规范hidden和Filter起点完整重放Estimator，不能直接任意扰动GRU hidden。

动作历史变体只通过命令延迟、保持和电机滤波生成，避免构造不可达的策略内部状态。

## 4.7 Branch-based Empirical Labels与Beta后验

冻结当前Actor、Estimator和Phase Filter。对同一个增广状态\(z\)执行\(N_z\)次条件独立的分支rollout。每次trial重新采样episode级动力学条件和未来过程噪声：

$$
\xi_m\sim p_{\xi},\qquad
\nu_m\sim p_{\nu},
\qquad m=1,\ldots,N_z.
$$

在固定增广状态、冻结策略版本和固定随机化分布的条件下，各branch使用独立随机种子。若采用成组domain条件或公共随机数降低比较方差，必须按组保存元数据，并在统计分析中避免把相关trial误当作完全独立样本。

同一条branch同时记录链式标签和最终Recovery标签；两类标签可以相关，但分别建立Bernoulli统计。

### 4.7.1 递归阶段标签

对于非Landing阶段\(k\)：

$$
y_m^{chain}
=
\mathbf 1
\left(
\tau_{\mathcal C_{k+1}}<\tau_{\mathcal F},
\ \tau_{\mathcal C_{k+1}}\le H_k
\right).
$$

Landing阶段定义：

$$
y_m^{chain}=y_m^G.
$$

链式标签仅用于Bootstrap mastery、阶段入口集合和链式误差诊断。

### 4.7.2 最终Recovery标签

所有阶段统一记录：

$$
y_m^G
=
\mathbf 1
\left(
\tau_{\mathcal G}<\tau_{\mathcal F},
\ \tau_{\mathcal G}\le H
\right).
$$

正式Empirical Recoverability Tube仅由\(y^G\)对应的后验定义。若状态能够进入下一阶段入口但后续没有Recovery，则有\(y_m^{chain}=1,y_m^G=0\)，该状态不能进入端到端高可恢复Tube。

### 4.7.3 共轭Beta后验

对目标\(r\in\{chain,G\}\)，定义：

$$
S_r(z)=\sum_{m=1}^{N_z} y_m^r,
\qquad
F_r(z)=N_z-S_r(z).
$$

采用固定的均匀弱先验：

$$
p_r(z)\sim\mathrm{Beta}(\alpha_0,\beta_0),
\qquad
\alpha_0=\beta_0=1.
$$

观察到\(S_r\)次成功和\(F_r\)次失败后：

$$
p_r(z)\mid\mathcal D_z
\sim
\mathrm{Beta}
\left(
\alpha_0+S_r(z),
\beta_0+F_r(z)
\right).
$$

后验均值为：

$$
\bar p_r(z)
=
\mathbb E[p_r(z)\mid\mathcal D_z]
=
\frac{\alpha_0+S_r(z)}
{\alpha_0+\beta_0+N_z}.
$$

令\(Q_q[\mathrm{Beta}(a,b)]\)表示Beta分布的\(q\)分位数。默认采用90%等尾贝叶斯可信区间：

$$
L_r(z)
=
Q_{0.05}
\left[
\mathrm{Beta}
\left(
\alpha_0+S_r,
\beta_0+F_r
\right)
\right],
$$

$$
U_r(z)
=
Q_{0.95}
\left[
\mathrm{Beta}
\left(
\alpha_0+S_r,
\beta_0+F_r
\right)
\right].
$$

因此得到\(L_{chain},U_{chain},L_G,U_G\)。正文固定使用\(5\%/95\%\)分位数；附录报告可信水平敏感性，例如总尾概率\(\alpha_{cred}\in\{0.05,0.10,0.20\}\)，分别对应95%、90%和80%中央可信区间。Beta\((1,1)\)是本研究预先固定的均匀弱先验，不宣称其是唯一无信息先验；可在附录以Jeffreys先验Beta\((0.5,0.5)\)进行敏感性检查。

### 4.7.4 序贯分支预算与停止规则

为避免少量trial在先验影响下被过早分类，所有经验集合判定必须满足最小分支数\(N_{\min}\)。每次按小批量增加branch，直到满足以下任一条件：

1. 高可恢复停止：
   \[
   N_z\ge N_{\min}
   \quad\land\quad
   L_G(z)\ge\eta_h;
   \]

2. 低可恢复停止：
   \[
   N_z\ge N_{\min}
   \quad\land\quad
   U_G(z)<\eta_l;
   \]

3. 稳定边界停止：
   \[
   N_z\ge N_{\min},
   \quad
   \eta_l\le\bar p_G(z)\le\eta_h,
   \quad
   U_G(z)-L_G(z)\le\epsilon_B;
   \]

4. 达到最大预算\(N_z=N_{\max}\)。

达到\(N_z=N_{\max}\)后仍未满足前三项的状态进入证据不足集合U。\(N_{\min}\)、\(N_{\max}\)、\(\eta_l\)、\(\eta_h\)和\(\epsilon_B\)必须在正式多随机种子实验前固定，并在Implementation表中报告。

## 4.8 Phase-Explicit Physical-Belief Viability Ensemble

每个Viability Network包含：

- PhysicalStateEncoder；
- PolicyStateEncoder；
- 显式阶段one-hot与连续进度编码；
- TaskCondition输入；
- 共享融合层；
- final-Recovery head；
- 可选chain head。

输出定义为：

$$
\hat V_G(z)\approx
P
\left(
\tau_{\mathcal G}<\tau_{\mathcal F},
\ \tau_{\mathcal G}\le H
\mid z
\right),
$$

$$
\hat V_{chain}(z)\approx
P
\left(
\tau_{\mathcal C_{k+1}}<\tau_{\mathcal F},
\ \tau_{\mathcal C_{k+1}}\le H_k
\mid z
\right).
$$

论文主结论使用\(\hat V_G\)；\(\hat V_{chain}\)只服务Bootstrap和误差诊断。最简实现可省略chain head，直接使用链式Beta后验推进课程。

### 4.8.1 状态均衡的Bernoulli训练

对目标\(r\in\{G,chain\}\)，单个状态的branch平均损失为：

$$
\ell_r(z)
=
- \frac{1}{N_z}
\sum_{m=1}^{N_z}
\left[
y_m^r\log \hat V_r(z)
+
(1-y_m^r)\log(1-\hat V_r(z))
\right].
$$

总体损失为：

$$
\mathcal L_V
=
\frac{1}{|\mathcal D_z|}
\sum_{z\in\mathcal D_z}
\left[
\ell_G(z)
+
\lambda_{chain}\ell_{chain}(z)
\right].
$$

按状态而不是按全部branch直接平均，可以避免序贯评估中获得更多trial的困难状态在训练损失中被不成比例地放大。实现时使用BCEWithLogits，并采用state-balanced sampler。若省略chain head，则令\(\lambda_{chain}=0\)。

### 4.8.2 Ensemble与两类不确定性

设ensemble成员数为\(M_V\)。成员使用不同初始化、按snapshot group进行bootstrap重采样，并改变mini-batch顺序。训练、验证和测试按augmented snapshot或episode group切分，禁止同一物理快照的不同belief或branch跨集合泄漏。

记final-Recovery输出的ensemble均值和标准差为：

$$
\mu_V(z)
=
\frac{1}{M_V}\sum_{j=1}^{M_V}\hat V_G^{(j)}(z),
$$

$$
\sigma_V(z)
=
\sqrt{
\frac{1}{M_V-1}
\sum_{j=1}^{M_V}
\left(
\hat V_G^{(j)}(z)-\mu_V(z)
\right)^2
}.
$$

\(\sigma_V\)反映函数逼近和数据支持造成的模型不确定性；它不等价于Beta后验区间宽度。Beta宽度来自同一状态的有限branch试验，属于经验统计不确定性。

### 4.8.3 Phase-conditioned Support Score

为阻止网络在训练支持域外过度自信，按真实阶段计算support。对标准化后的融合特征\(h(z)\)，令\(d_k^\phi(z)\)为其到同阶段训练状态的第\(k\)近邻距离，定义：

$$
S_{\mathrm{sup}}(z)
=
\exp
\left[
- \left(
\frac{d_k^\phi(z)}{\tau_{\mathrm{sup}}}
\right)^2
\right].
$$

\(k\)、\(\tau_{\mathrm{sup}}\)及特征标准化规则在正式实验前固定。Support Score只作为经验确认前的网络预测门控，不替代branch rollout。

## 4.9 Phase-Conditioned Empirical Recoverability Tube

### 4.9.1 经验集合划分

按真实阶段定义高可恢复Tube切片：

$$
\mathcal T_{\phi}^{+,\mathrm{emp}}
=
\left\{
z:
\phi^{true}(z)=\phi,\
N_z\ge N_{\min},\
L_G(z)\ge\eta_h
\right\}.
$$

完整经验分类为：

$$
\begin{aligned}
\mathcal T_{\mathrm{emp}}^+
&=
\left\{
z:
N_z\ge N_{\min},
\ L_G(z)\ge\eta_h
\right\},\\
\mathcal T_{\mathrm{emp}}^-
&=
\left\{
z:
N_z\ge N_{\min},
\ U_G(z)<\eta_l
\right\},\\
B
&=
\left\{
z:
N_z\ge N_{\min},
\ \eta_l\le\bar p_G(z)\le\eta_h,
\ U_G(z)-L_G(z)\le\epsilon_B
\right\},\\
U
&=
\mathcal Z_{\mathrm{cand}}
\setminus
\left(
\mathcal T_{\mathrm{emp}}^+
\cup
\mathcal T_{\mathrm{emp}}^-
\cup
B
\right).
\end{aligned}
$$

其中：

- \(\mathcal T_{\mathrm{emp}}^+\)：端到端经验高可恢复状态；
- \(\mathcal T_{\mathrm{emp}}^-\)：经验低可恢复状态；
- \(B\)：已有足够证据确认其成功概率位于中间决策区间的状态；
- \(U\)：证据不足、区间过宽或尚未完成当前策略版本评估的状态。

物理Failure集合\(\mathcal F\)表示已经发生明确失败的系统状态；\(\mathcal T_{\mathrm{emp}}^-\)表示从当前状态出发，在当前策略下最终Recovery概率较低。二者不是同一概念。

边界B与未知U也不能混为一类。B表示“已经测清且确实困难”，U表示“尚未测清或缺乏支持”。

阶段递归入口集合另行定义为：

$$
\mathcal C_k
=
\left\{
z:
\phi^{true}(z)=k,\
N_{z,chain}\ge N_{\min}^{chain},\
L_{chain}(z)\ge\eta_h^{chain}
\right\}.
$$

\(\mathcal C_k\)只用于Bootstrap衔接，不等同于正式的\(\mathcal T_{\mathrm{emp}}^+\)。

模型预测分类为：

$$
\begin{aligned}
\mathcal T_{\mathrm{pred}}^+
&:
\mu_V-\beta\sigma_V\ge\eta_h
\land
\sigma_V<\sigma_{\max}
\land
S_{\mathrm{sup}}\ge S_{\min},\\
\mathcal T_{\mathrm{pred}}^-
&:
\mu_V+\beta\sigma_V<\eta_l
\land
\sigma_V<\sigma_{\max}
\land
S_{\mathrm{sup}}\ge S_{\min},\\
B_{\mathrm{pred}}
&:
\text{其余且在支持域内},\\
U_{\mathrm{pred}}
&:
\text{高模型不确定或低support}.
\end{aligned}
$$

用途严格区分：

- \(\mathcal T_{\mathrm{emp}}^+\)：正式PPO RSI来源；
- B：低权重课程、难例学习和重点重标注区域；
- \(\mathcal T_{\mathrm{pred}}^+\)、\(B_{\mathrm{pred}}\)和\(U_{\mathrm{pred}}\)：branch acquisition候选；
- \(\mathcal T_{\mathrm{emp}}^-\)：不进入正式RSI；
- 仅有网络预测而未经当前策略版本branch确认的状态，不能直接升级为正式高可恢复Tube。

### 4.9.2 独立审计下的Tube质量

Tube构建、Viability训练和Tube审计必须使用不同的branch随机种子。构建一个固定、按阶段分层的审计候选池\(\mathcal A\)，其状态来自预先声明的候选分布\(q^{audit}(z)\)。审计时冻结与Tube一致的policy、Estimator和环境配置版本，并使用独立且通常更大的branch预算。

定义状态级保守审计标签：

$$
g_i^{audit}
=
\mathbf 1
\left[
L_G^{audit}(z_i)\ge\eta_h
\right].
$$

令：

$$
s_i
=
\mathbf 1
\left[
z_i\in\mathcal T_{\mathrm{emp}}^+
\right].
$$

则保守Tube precision为：

$$
\mathrm{Precision}_{tube}
=
\frac{
\sum_{z_i\in\mathcal A}s_i g_i^{audit}
}{
\sum_{z_i\in\mathcal A}s_i
}.
$$

Recoverable recall为：

$$
\mathrm{Recall}_{recoverable}
=
\frac{
\sum_{z_i\in\mathcal A}s_i g_i^{audit}
}{
\sum_{z_i\in\mathcal A}g_i^{audit}
}.
$$

Candidate-mass coverage为：

$$
\mathrm{Coverage}_{mass}
=
\frac{1}{|\mathcal A|}
\sum_{z_i\in\mathcal A}s_i.
$$

如果审计池通过非均匀proposal生成，则三项指标使用已声明的importance weights估计。

另外报告Tube内聚合branch成功率：

$$
\mathrm{SuccessRate}_{tube}
=
\frac{
\sum_{i:s_i=1} S_G^{audit}(z_i)
}{
\sum_{i:s_i=1} N_{z_i}^{audit}
}.
$$

Precision衡量纳入Tube的状态有多少被独立审计重新确认；Recall衡量可恢复候选中有多少被Tube捕获；Mass coverage衡量Tube占候选分布的范围。三者必须联合报告，避免通过只保留极少状态人为获得高Precision。

Viability calibration在独立held-out branch上评估。除Brier score和ECE外，绘制预测概率—实际成功率可靠性图。经验Beta后验与神经网络预测分别校准，不能把Beta可信区间当成网络预测置信区间。

### 4.9.3 Chain-to-Final误差诊断

递归Bootstrap可能产生“局部阶段成功但最终恢复失败”。定义：

$$
\mathrm{FPR}_{chain\rightarrow G}
=
\Pr
\left(
y^{chain}=1,\ y^G=0
\right),
$$

称为false-progress rate。

同时定义：

$$
\mathrm{MSR}_{chain\rightarrow G}
=
\Pr
\left(
y^{chain}=0,\ y^G=1
\right),
$$

称为missed-success rate。该指标过高说明\(\mathcal C_{k+1}\)过窄、事件锚点不完整，或成功轨迹存在未被入口集合覆盖的替代路径。

阶段级链式—最终差距为：

$$
\Delta_{CF}^{(k)}
=
\mathbb E_{z\sim q_k^{cand}}
\left[
\bar p_{chain}(z)-\bar p_G(z)
\right].
$$

若认证入口覆盖绝大多数成功轨迹，则MSR应较低；较高FPR说明递归目标虽然解决冷启动，但不能替代端到端Recovery认证。本文不假设链式概率与最终概率存在严格乘法分解，而是通过上述指标进行经验诊断。

## 4.10 Tube-guided RSI

正式reset分布：

$$
p_{reset}
=
\lambda_{nat}p_{natural}
+\lambda_+p_{T_{\mathrm{emp}}^+}
+\lambda_Bp_B.
$$
建议：

- Landing早期：提高 \(T_{\mathrm{emp}}^+\) 比例；
- 阶段扩展时：保留上一阶段Tube并加入当前阶段边界；
- 完整任务后期：\(\lambda_{nat}\ge0.8\)；
- 最终评估：\(\lambda_{nat}=1\)。

Tube restore只在episode起点生效，物理状态与PolicyState必须同步恢复。

## 4.11 PPO、Critic与奖励

Actor使用可部署观测，Critic额外读取 \(x^{priv}\) 和 \(c^{true}\)。

PPO rollout存储：

- actor observation；
- action和log probability；
- privileged critic observation；
- reward；
- terminated/truncated；
- value。

GAE：

$$
m_t^{boot}=1-\mathbf 1(\mathrm{terminated}_t),
$$

$$
m_t^{trace}
=
1-\mathbf 1(\mathrm{terminated}_t\lor\mathrm{truncated}_t),
$$

$$
\delta_t=r_t+\gamma m_t^{boot}V_{t+1}-V_t,
$$

$$
A_t=\delta_t+\gamma\lambda m_t^{trace}A_{t+1}.
$$

奖励分为：

### Bootstrap局部奖励

用于Landing等早期阶段：

- 姿态恢复；
- 前向速度；
- 有效轮地接触；
- 冲击抑制；
- 动作平滑。

### 最终统一奖励

$$
r_t=
w_p r_{progress}
+w_s r_{safety}
+w_g r_{goal}
- w_a\|\Delta a_t\|^2.
$$

所有核心基线除“hand reward shaping”外使用同一最终奖励，避免把奖励差异误认为课程或Tube贡献。

## 4.12 外循环、策略漂移检测与预算化重标注

Recoverability是策略条件化的：

$$
\bar V_H^{\pi_k,\eta}(z).
$$

因此，Tube不能在策略持续更新时被永久视为有效。但频繁暂停PPO并重新执行全部branch rollout同样不可行。本方法采用checkpoint级的事件触发刷新，而不是每个minibatch、epoch或PPO迭代都重标注。

### 4.12.1 漂移探针集合

构建固定或缓慢更新的探针集合：

$$
\mathcal P=
\mathcal P_B
\cup
\mathcal P_{use}
\cup
\mathcal P_{conn}
\cup
\mathcal P_{nat},
$$

分别覆盖：

- 边界集合B；
- RSI中高使用率的\(T_{\mathrm{emp}}^+\)状态；
- Approach/Takeoff、Takeoff/Flight和Flight/Landing连接状态；
- 固定自然起点评估轨迹中的代表状态。

在每个策略checkpoint上计算：

$$
D_{KL}^{probe}(k,k_0)
=
\frac{1}{|\mathcal P|}
\sum_{z\in\mathcal P}
D_{KL}
\left[
\pi_{\theta_{k_0}}(\cdot|o_z^{dep})
\Vert
\pi_{\theta_k}(\cdot|o_z^{dep})
\right].
$$

同时记录固定评估集上的端到端Recovery变化：

$$
\Delta S_G=S_G(\pi_k)-S_G(\pi_{k_0}).
$$

### 4.12.2 重标注触发条件

当满足任一条件时触发增量刷新：

$$
D_{KL}^{probe}>\delta_{KL},
$$

或

$$
\Delta S_G<-\delta_{perf},
$$

或

$$
A_{label}\ge K_{max},
$$

其中\(A_{label}\)为当前标签自上次验证后的策略更新块数。第三项是最长刷新间隔，用于防止KL探针漏检局部决策变化。

\(\delta_{KL}\)、\(\delta_{perf}\)和\(K_{max}\)是实现超参数，应在标称任务上预先确定并保持固定。可以将“每20个PPO更新块”作为\(K_{max}\)的初始工程值，但不能把它写成普适常数；需报告敏感性或至少说明所用值。

### 4.12.3 固定预算的优先刷新

每次触发后不刷新全部Tube，而是在固定branch预算\(M_{ref}\)下按优先级选择状态。

定义当前标签版本下的Beta经验区间宽度：

$$
\Delta_{\mathrm{Beta}}(z)
=
U_G(z)-L_G(z).
$$

它主要反映有限branch样本产生的经验统计不确定性。Viability Ensemble标准差\(\sigma_V(z)\)则反映模型在函数逼近和数据支持上的不确定性。二者同时进入优先级：

$$
\begin{aligned}
q(z)
={}&
w_B\mathbf 1[z\in B]
+w_u\widetilde f_{use}(z)
+w_c\mathbf 1[z\in\mathcal P_{conn}]\\
&+
w_{\Delta}\widetilde\Delta_{\mathrm{Beta}}(z)
+w_{\sigma}\widetilde\sigma_V(z)
+w_a\widetilde A_z.
\end{aligned}
$$

其中带波浪号的连续量均在当前候选池内归一化到\([0,1]\)，并要求：

$$
w_j\ge0,
\qquad
\sum_j w_j=1.
$$

各项含义为：

- \(f_{use}(z)\)：近期RSI采样频率；
- \(\Delta_{\mathrm{Beta}}(z)\)：同一状态branch证据的后验区间宽度；
- \(\sigma_V(z)\)：Viability Ensemble模型不确定性；
- \(A_z\)：标签年龄；
- \(\mathcal P_{conn}\)：阶段连接状态；
- B：已确认处于概率决策边界的状态。

这里不能把“区间宽”简单解释为“必然处于生死边缘”。区间宽首先表示证据不足；真正的边界状态还必须满足\(\bar p_G\in[\eta_l,\eta_h]\)和宽度收敛条件。保留B指示项与\(\Delta_{\mathrm{Beta}}\)项，正是为了区分“已确认困难”与“尚未测清”。

选取top-\(M_{ref}\)状态执行新策略下的branch rollout，并为当前policy version重新建立Beta统计。旧策略后验只能作为排序线索，不能继续作为\(\pi_k\)下的概率证据。对于完全没有当前版本branch的状态，将其归入U；其当前版本Beta宽度按最大不确定性处理，或通过单独的“未标注”指示项进入刷新队列。

### 4.12.4 周期性全局审计

为了防止增量刷新长期遗漏，设置更低频的全局审计周期\(K_{audit}\)，对各阶段随机抽样状态重新branch评估。全局审计不要求重标注所有状态，而是估计Tube precision、coverage和校准是否系统性退化。

完整外循环为：

1. 固定当前Tube训练PPO若干更新块；
2. 在checkpoint上计算probe KL和固定评估集性能；
3. 未触发时继续PPO训练；
4. 触发时冻结\(\pi_k,\eta\)，在预算\(M_{ref}\)内刷新优先状态；
5. 更新Beta后验、Viability和Tube版本；
6. 达到\(K_{audit}\)时执行分层随机审计；
7. 最终从自然起点评估。

该机制控制的是Tube维护成本，不保证“长期安全”。本文仍只主张训练分布与有限时域内的经验可恢复性。

---

# 5. Implementation-Relevant Interfaces

## 5.1 传感器与动作

Proprioception至少包括：

- IMU orientation 6D；
- angular velocity；
- specific force；
- hip/knee/steering位置与速度；
- 前后轮速度；
- motor effort。

Task observation包括：

- obstacle relative distance、height/width estimate；
- 对应不确定性；
- target velocity。

高频传感器向50 Hz控制边界聚合方式必须在MuJoCo和Isaac中一致，例如：

- orientation、joint、wheel：last；
- IMU：last、min、max；
- motor effort：last、mean。

## 5.2 Contact Adapter统一语义

所有引擎统一输出：

```text
OracleContactState:
  front_valid_ground_contact
  rear_valid_ground_contact
  front_normal_force
  rear_normal_force
  front_normal_world
  rear_normal_world
  frame_collision
  wheel_side_collision
  obstacle_side_collision
```

Oracle phase machine只读取上述统一语义。

## 5.3 Snapshot和状态恢复

论文正文只需说明增广Snapshot完整保存物理状态和PolicyState。工程实现需通过round-trip测试验证：

- 恢复后传感器包一致；
- 执行器力矩一致；
- Oracle phase一致；
- Actor observation一致；
- 同动作和同随机种子下未来轨迹一致。

## 5.4 Estimator训练流程

推荐流程：

1. 使用正常驾驶和人工候选训练初始Estimator；
2. 在Backward Bootstrap中冻结Estimator并收集更多数据；
3. 使用全部数据最终校准并永久冻结；
4. 若Estimator更新导致Actor观测分布显著变化，进行短期Actor适配；
5. 从最终Actor与Estimator版本开始正式Viability标注。


## 5.5 Latency Buffer、Posterior Cache与Tube Versioning

工程实现需维护Estimator输出的时间戳或控制步索引，使phase/contact/progress延迟可以独立于原始传感器延迟进行注入。至少支持：

- 固定控制步延迟；
- episode级随机延迟；
- 短时间块输出保持；
- 独立的全传感器延迟实验。

每个Tube状态和标签记录：

```text
policy_version
estimator_version
tube_version
label_timestamp
branch_count_chain
success_count_chain
branch_count_final
success_count_final
posterior_quantiles
credible_level
audit_seed_namespace
support_score
reset_use_count
source_phase
connection_flag
```

这样可以计算标签年龄、RSI使用率、Beta区间宽度和优先刷新分数，并防止不同策略版本的Bernoulli统计被错误合并。构建Tube和独立审计必须使用不同的branch seed namespace。

---

# 6. Experiments

## 6.1 核心基线

| 方法 | 目的 |
|---|---|
| PPO | 自然起点直接训练 |
| PPO + hand reward shaping | 排除仅靠复杂奖励的可能 |
| PPO + CoM-guideline RSI | 直接比较最初的CoM中间状态初始化 |
| PPO + phase-randomized RSI | 比较仅按阶段随机初始化 |
| PPO + backward curriculum | 只有反向课程，没有Viability |
| PPO + empirical buffer RSI | 使用成功状态缓存，不学习Tube |
| DVGC-Physical | Viability只读取物理状态 |
| DVGC-PB-Clean | 使用真实生成的clean belief |
| DVGC-PB-Perturbed | 完整方法 |

不建议设置过多不相关基线。主表可保留PPO、CoM-RSI、Backward Curriculum、DVGC-PB-Perturbed，其他方法进入消融表。

## 6.2 研究问题与实验

### RQ1：是否突破Survival Bottleneck？

比较：

- 首次成功步数；
- 达到50%和80%自然起点成功率的总交互量；
- Approach阶段平均存活时间；
- 不同阶段访问比例。

### RQ2：端到端Tube是否优于固定初始化与仅递归阶段集合？

比较：

- natural-start完整成功率；
- \(T_{\mathrm{emp}}^+\)内的经验成功率；
- B集合成功率区间；
- RSI状态的实际恢复率；
- 不同reset策略下的学习曲线。

### RQ3：Physical-Belief是否必要？

比较DVGC-Physical、PB-Clean和PB-Perturbed。

增加直接机制实验：

1. 固定同一个 \(x^{priv}\)；
2. 构造多个不同 \(b_i\)；
3. 计算动作差异：

$$
d_a(i,j)=
\|\mu_\theta(o_i^{dep})-\mu_\theta(o_j^{dep})\|_2;
$$

4. 比较经验恢复率差异：

$$
d_V(i,j)=
|\hat V(x,b_i)-\hat V(x,b_j)|.
$$

如果物理状态相同但动作和恢复率显著变化，说明belief不是冗余输入。

### RQ4：递归Bootstrap、最终认证、阶段条件与统计设计各自贡献

结构和目标消融至少包含：

- DVGC-full：递归Bootstrap + 最终Recovery认证Tube；
- Chain-only：只按进入下一阶段集合构建RSI；
- End-to-end-only：从一开始仅用最终Recovery标签；
- no Viability：使用全部候选，不进行Recoverability筛选；
- no Boundary sampling；
- No explicit phase：Viability移除\(e_\phi,\rho\)；
- Physical-only：Viability移除\(b_t\)；
- Oracle phase / Estimated phase。

统计协议消融至少包含：

- credible level：80%、90%、95%中央Beta可信区间；
- prior sensitivity：Beta\((1,1)\)为主设置，Beta\((0.5,0.5)\)进入附录；
- no Beta-width priority：重标注优先级移除\(\Delta_{\mathrm{Beta}}\)；
- no ensemble priority：重标注优先级移除\(\sigma_V\)。

该组实验回答：

1. 递归链式目标是否主要解决冷启动；
2. 最终Recovery认证是否阻止局部可达状态被误纳入Tube；
3. 显式真实阶段和PolicyState是否提升Viability拟合与Tube校准；
4. Beta可信水平对Precision—Coverage权衡有何影响；
5. Beta经验不确定性与Ensemble模型不确定性是否提供互补信息。

另外按阶段报告：

- false-progress rate；
- missed-success rate；
- \(\Delta_{CF}^{(k)}\)。

这些指标用于揭示递归入口集合的误差传播，而不是将Chain-only的局部成功率等同于完整跳跃能力。

### RQ5：阶段滞后、泛化与重标注成本

#### A. Phase Estimation Lag

比较两种训练方式：

- no-lag training；
- lag-randomized training。

测试延迟优先按控制周期设置：

| 延迟档位 | 控制步 | 物理时间 |
|---|---:|---:|
| L0 | 0 | 0 ms |
| L1 | 1 | 20 ms |
| L2 | 2 | 40 ms |
| L3 | 3 | 60 ms |

若支持高频时间戳，可附加10 ms和50 ms，但需说明50 Hz控制边界造成的有效延迟量化。

分别测试：

1. 仅延迟\(\hat p^{phase},\hat p^{contact},\rho\)，隔离阶段估计滞后；
2. 延迟全部可部署传感器，评估端到端感知与控制延迟；
3. 输出保持或偶发丢帧，模拟Estimator更新不稳定。

指标包括：

- natural-start完整成功率；
- valid landing和recovery success；
- takeoff/first-contact事件后的动作响应延迟；
- 错误阶段概率持续时间；
- 峰值姿态误差和落地冲击；
- Oracle、无延迟Estimated和延迟Estimated之间的性能差。

不能只用“策略仍活下来”作为结论，应报告完整任务成功率和动作/事件对齐误差。

#### B. Relabeling Cost

比较：

- no refresh：策略更新后不重标注；
- always refresh：每个外循环刷新全部候选，作为高成本上界；
- periodic-only：固定周期刷新；
- trigger-budgeted：本文的KL/性能/标签年龄触发与固定预算刷新。

公平比较时保持PPO环境交互预算一致，并额外报告：

- 触发次数；
- 每次刷新状态数；
- 新增branch rollout步数；
- 总branch rollout步数；
- 独立审计的保守Tube precision、recoverable recall、candidate-mass coverage、aggregate success rate和ECE；
- Beta区间宽度与Ensemble方差的分布；
- natural-start成功率；
- 相同硬件上的wall-clock时间。

periodic-only中的周期和trigger-budgeted中的\(K_{max}\)可用相同初始值，例如20个PPO更新块，以区分“固定刷新”与“事件触发”的贡献。

#### C. 任务与部署泛化

主论文至少包含：

- 障碍高度；
- 障碍宽度；
- 初始或目标速度；
- IMU bias、编码器噪声和轮速滑移；
- 阶段输出延迟。

若篇幅允许，再加入MuJoCo→Isaac L0–L3。Isaac adaptation和worst-k/CVaR放附录或扩展实验。

## 6.3 评价指标

### 任务性能

- natural-start完整跳跃成功率；
- takeoff success；
- obstacle clearance；
- valid landing；
- recovery success；
- landing peak force或impact impulse；
- recovery time；
- peak torque与动作平滑度。

### 样本效率

- 首次成功所需环境步数；
- 达到50%和80%成功率的总交互量；
- 总交互量必须包含PPO、Bootstrap和branch rollout。

### Tube质量与统计复现性

必须在独立审计branch上报告：

- \(\mathrm{Precision}_{tube}\)：被纳入Tube且在独立审计中重新认证为高可恢复的状态比例；
- \(\mathrm{Recall}_{recoverable}\)：独立审计确认的高可恢复候选中被Tube捕获的比例；
- \(\mathrm{Coverage}_{mass}\)：Tube在固定候选分布中的质量占比；
- \(\mathrm{SuccessRate}_{tube}\)：Tube审计branch的聚合最终Recovery成功率；
- phase-wise precision、recall与mass coverage；
- B集合的后验均值、区间宽度和实际成功率分布；
- U集合比例及其随branch预算变化的收缩速度；
- AUC、Brier score、ECE和reliability diagram；
- \(T_{\mathrm{pred}}^+\)被经验确认的比例；
- support score与错误率关系；
- posterior width \(U_G-L_G\)随branch数变化的曲线；
- 80%、90%、95%可信区间下的Precision—Coverage曲线；
- false-progress rate、missed-success rate和\(\Delta_{CF}^{(k)}\)。

Tube construction branches、Viability训练分支和最终audit branches必须按seed与snapshot group隔离，避免同一随机试验同时用于集合构建和质量证明。

### Estimator与阶段滞后

- contact precision、recall、F1；
- phase macro-F1和混淆矩阵；
- takeoff、first-contact和landing事件延迟；
- phase probability age和stale-output持续时间；
- 真实事件后动作分布发生显著变化所需的控制步数；
- Oracle、无延迟Estimated和延迟Estimated性能差。

### 统计报告

- 主方法和核心基线至少使用5个独立训练随机种子；
- 计算成本极高的附加消融若只使用3个种子，必须明确标注并避免作强显著性结论；
- 聚合任务指标报告均值、标准差和跨训练种子的95%置信区间；
- 单状态Recoverability使用Beta后验可信区间，不能与聚合指标的频率学派置信区间混称；
- 自然起点评估episode数固定；
- 所有方法使用相同PPO预算、候选分布、任务随机化和评估协议；
- 显著性检验只作为补充，不替代效果量和置信区间。

## 6.4 计算成本报告

必须分别报告：

- PPO环境步；
- Bootstrap候选生成步；
- 初始Tube构建branch rollout步；
- 增量重标注branch rollout步；
- 重标注触发次数、平均刷新状态数和全局审计次数；
- Viability训练时间；
- 相同硬件上的总wall-clock时间；
- 可选Sim2Sim重标注成本。

除总训练成本外，还应报告“每提升一个百分点自然起点成功率所需的额外branch步数”，避免trigger-budgeted方案仅因刷新不足而看似便宜，也避免always-refresh方案在PPO步数相同的情况下隐藏大量额外计算。

---

# 7. Discussion and Limitations

应明确讨论：

- 递归入口集合可能产生误差传递，因此必须使用端到端Recovery重新认证；
- 链式成功率与最终Recovery成功率可能出现明显差距，需要分别报告；
- 分支评估带来的计算开销；
- 策略更新后Tube需要重标注，但漂移阈值和刷新预算会影响成本与标签新鲜度之间的权衡；
- KL探针只覆盖有限状态，可能漏检局部策略变化，因此仍需低频分层全局审计；
- 对精确Snapshot恢复的依赖；
- Estimator误差和阶段滞后可能改变课程状态价值；
- 若采用Multi-head扩展，输出端线性soft mixture可能把方向相反的阶段动作抵消，主方案因此使用共享非线性Actor；
- 50 Hz控制使毫秒级延迟实验存在量化效应，结果不能直接外推到任意硬件频率；
- Beta可信界依赖先验、可信水平和有限branch预算，只是当前统计协议下的后验不确定性，不是分布外安全保证；
- 高Precision、recoverable recall与candidate-mass coverage存在结构性权衡，必须联合报告；
- 经验Tube只在训练随机化分布和有限时域内成立；
- 跨引擎验证不能等同于实机成功；
- 风险敏感跨domain评分和在线Viability更新属于未来工作或扩展实验。

---

# 8. Conclusion

本文针对单轨机器人动态跳跃中的Survival Bottleneck，提出Event-Anchored Landing-first Backward Bootstrap和Phase-Explicit Physical-Belief Empirical Recoverability Tube。递归链式目标用于建立早期控制能力，最终Recovery标签与Beta后验可信界用于正式Tube认证。连续阶段概率通过共享非线性Actor实现平滑阶段条件控制，避免硬切换与输出动作线性平均的抵消风险。训练外循环联合使用Beta经验区间宽度、Viability Ensemble模型不确定性、策略漂移和标签年龄，在固定branch预算内刷新Tube。最终通过独立审计的Precision、Recoverable Recall、Candidate-mass Coverage、Calibration以及自然起点完整跳跃验证方法有效性。

---

# 附录 A：论文Algorithm 1伪代码

```text
Input:
  natural reset distribution p_nat
  candidate generators G_phase
  actor π, estimator η
  Beta prior α0 = β0 = 1
  posterior quantiles qL = 0.05, qU = 0.95
  branch limits N_min, N_max
  thresholds η_l, η_h, ε_B
  probe set P
  relabel thresholds δ_KL, δ_perf
  maximum label age K_max
  refresh budget M_ref
  audit interval K_audit

1: Pretrain and calibrate η
2: Train shared π with continuous phase probabilities
   and lag randomization over estimator outputs
3: C_next ← Recovery set G
4: for phase in [Landing, Flight, Takeoff, Approach] do
5:     Generate and physically validate phase candidates
6:     Train the same actor to reach C_next before Failure
7:     Evaluate chain success and end-to-end Recovery
8:     Freeze π, η and phase filter
9:     for each selected augmented snapshot z do
10:        Repeat independent branches in small batches
11:        Record y_chain and y_G on every branch
12:        Update Beta(α0+S_chain, β0+F_chain)
              and Beta(α0+S_G, β0+F_G)
13:        Compute posterior means and qL/qU quantiles
14:        Stop at high, low, stable-boundary decision,
              or when N_max is reached
15:     end for
16:     Construct C_phase from L_chain
17:     Construct T_emp+, T_emp−, B and U from L_G, U_G and N_min
18:     Train phase-explicit Physical-Belief Viability Ensemble
19:     Audit a disjoint state/seed subset
20:     Record Precision, Coverage, Calibration,
          false-progress and missed-success
21:     C_next ← C_phase
22: end for
23: Save reference policy π_ref and Tube version v
24: repeat
25:     Train PPO for one update block using p_nat, T_emp+ and B
26:     At checkpoint, compute D_KL^probe and ΔS_G
27:     if D_KL^probe > δ_KL
           or ΔS_G < -δ_perf
           or label_age >= K_max then
28:         Compute Δ_Beta = U_G - L_G
29:         Rank states using boundary, reset use, connection,
             normalized Δ_Beta, normalized σ_V and label age
30:         Select top-M_ref states under fixed branch budget
31:         Freeze current π and generate fresh current-version labels
32:         Rebuild current-policy Beta statistics and update Tube
33:         π_ref ← π; reset label age; increment Tube version
34:     end if
35:     if audit_age >= K_audit then
36:         Run stratified disjoint Tube audit
37:     end if
38: until training budget exhausted
39: Evaluate credible-level sensitivity, 0/20/40/60 ms phase lag,
      relabeling cost and natural-start complete jumps
40: Return final policy and versioned empirical Tube
```

---

# 附录 B：建议默认实验层级

## 主文必须完成

1. PPO、CoM-RSI、Backward Curriculum、DVGC；
2. Physical vs PB-Clean vs PB-Perturbed；
3. same physical / different belief；
4. natural-start success与总交互成本；
5. Oracle/Estimated phase；
6. 独立审计下的Tube Precision、Recoverable Recall、Mass Coverage和Calibration；
7. false-progress与missed-success；
8. 障碍高度和速度泛化。

## 附录或扩展

1. worst-k、Q0.1、CVaR；
2. MuJoCo→Isaac L0–L3；
3. Isaac-specific Tube；
4. neighborhood viability；
5. Beta\((0.5,0.5)\)先验敏感性；
6. 80%/90%/95%可信区间完整Precision—Coverage曲线；
7. 完整版本管理与工程单元测试。

---

# 附录 C：完整性检查

## 理论闭环

- [x] Actor只读取可部署观测；
- [x] Viability读取Physical-Belief增广状态，并显式输入真实阶段one-hot与连续进度；
- [x] Recovery是目标集合而非第五phase；
- [x] Bootstrap链式标签与最终Recovery标签明确分离；
- [x] 正式Empirical Recoverability Tube仍由最终Recovery标签定义；
- [x] finite horizon和随机性来源明确；
- [x] belief perturbation与未来噪声分开；
- [x] empirical与predicted Tube用途分开；
- [x] 策略更新后通过KL、性能退化或最长标签年龄触发重标注；
- [x] 阶段输出延迟与真实阶段特权输入的角色明确分离；
- [x] 默认Actor不使用基于argmax phase的硬策略切换；
- [x] 明确禁止Multi-head动作输出端的朴素线性soft mixture；
- [x] Beta先验、后验、后验均值及5%/95%分位数定义完整；
- [x] 90%等尾贝叶斯可信区间与频率学派置信区间概念分开；
- [x] Chain-to-Final的false-progress、missed-success和阶段差距已定义。

## 算法闭环

- [x] CoM只用于候选搜索；
- [x] 递归事件Bootstrap避免全零标签；
- [x] 端到端Recovery重新认证避免局部可达性冒充全局可恢复性；
- [x] Snapshot完整恢复；
- [x] branch rollout产生Bernoulli标签；
- [x] Beta后验、\(N_{\min}/N_{\max}\)和序贯停止；
- [x] Viability Ensemble和support gate；
- [x] Beta经验区间宽度与Ensemble模型方差同时用于重标注；
- [x] Tube-guided RSI；
- [x] natural-start比例后期提高；
- [x] 固定branch预算的优先增量刷新；
- [x] 低频分层全局审计防止KL探针漏检。

## 实验闭环

- [x] 直接CoM-RSI基线；
- [x] Backward-only基线；
- [x] Physical-Belief机制实验；
- [x]样本效率包含全部交互成本；
- [x] Estimator指标与0/20/40/60 ms阶段滞后实验；
- [x] no-refresh、always-refresh、periodic-only与trigger-budgeted成本消融；
- [x] 独立审计的Tube Precision、Recoverable Recall、Mass Coverage和Calibration；
- [x] 可信水平和先验敏感性；
- [x] Chain-to-Final误差传播诊断；
- [x] 任务泛化；
- [x] Sim2Sim降级为扩展验证。

## 可行性结论

该论文核心框架在仿真中是可实现的。首要工程风险不是网络结构，而是：

1. Snapshot能否严格恢复；
2. Landing阶段能否先学到非零Recovery；
3. branch成本是否可控；
4. 阶段估计滞后是否在离地与首次接触附近引发动作响应延迟；
5. 漂移触发阈值是否能在Tube失效前及时启动增量刷新；
6. Beta可信水平和branch预算是否形成可接受的Precision—Coverage权衡；
7. Estimator误差是否会显著改变Actor输入分布。

因此实施顺序必须从单domain、固定Estimator、Landing→Recovery和完整Snapshot round-trip开始，再逐步加入Physical-Belief、完整四阶段和跨引擎扩展。

---

# 附录 D：统计复现协议

正式实验前固定并公开以下配置：

| 项目 | 主设置 | 说明 |
|---|---|---|
| Beta先验 | \(\mathrm{Beta}(1,1)\) | 均匀弱先验 |
| 下分位数 | \(q_L=0.05\) | 90%中央可信区间下界 |
| 上分位数 | \(q_U=0.95\) | 90%中央可信区间上界 |
| 最小branch数 | \(N_{\min}\) | 防止少量trial过早分类，数值在pilot后固定 |
| 最大branch数 | \(N_{\max}\) | 控制单状态评估成本，数值在pilot后固定 |
| 高阈值 | \(\eta_h\) | Tube准入阈值 |
| 低阈值 | \(\eta_l\) | 低可恢复判定阈值 |
| 边界宽度 | \(\epsilon_B\) | 区分稳定边界B与证据不足U |
| 审计seed | 独立namespace | 不与Tube构建或Viability训练复用 |
| 策略版本 | 强制记录 | 不跨policy version合并Bernoulli证据 |
| 低可恢复集合 | \(\mathcal T_{\mathrm{emp}}^-\) | 不与branch次数或Failure集合混用 |
| 审计指标 | Precision / Recall / Mass Coverage | 使用固定独立审计候选池 |

实现要求：

1. 同一状态的chain和final标签来自同一branch，但分别计数；
2. Tube构建、Viability训练/验证和最终审计按snapshot group与随机种子隔离；
3. 每个状态保存成功数和失败数，而不仅保存后验均值；
4. 所有后验分位数使用同一个数值库和固定精度；
5. 可信水平、先验和阈值敏感性只改变统计判定，不改变PPO训练预算；
6. 主结果使用预先固定的90%可信区间，其他可信水平只作为敏感性分析；
7. 对策略更新后的旧后验，不做折扣后继续累加；重新建立当前策略版本统计；
8. 审计候选池同时包含Tube内与Tube外状态，才能计算recoverable recall；
9. 若使用importance sampling构造审计池，公开proposal和权重截断规则。

---

# 附录 E：核心符号表

| 符号 | 含义 |
|---|---|
| \(o_t^{dep}\) | Actor可部署观测 |
| \(z_t\) | Physical-Belief增广状态 |
| \(\mathcal G\) | 最终Recovery目标集合 |
| \(\mathcal F\) | 已发生明确物理失败的状态集合 |
| \(\mathcal C_k\) | 阶段\(k\)的递归认证入口集合 |
| \(y^{chain}\) | 进入下一认证入口的Bernoulli标签 |
| \(y^G\) | 最终Recovery Bernoulli标签 |
| \(N_z\) | 状态\(z\)的branch rollout次数 |
| \(L_G,U_G\) | 最终Recovery概率的Beta后验5%/95%分位数 |
| \(\mathcal T_{\mathrm{emp}}^+\) | 经验高可恢复Tube |
| \(\mathcal T_{\mathrm{emp}}^-\) | 经验低可恢复状态集合 |
| \(B\) | 已测清的概率边界集合 |
| \(U\) | 证据不足或当前版本未标注集合 |
| \(\mu_V,\sigma_V\) | Viability Ensemble均值与模型标准差 |
| \(\Delta_{\mathrm{Beta}}\) | Beta后验区间宽度\(U_G-L_G\) |
| \(S_{\mathrm{sup}}\) | Phase-conditioned support score |
| \(M_{ref}\) | 单次增量重标注branch预算 |

