# JIT 投稿方法总体框图：生成 Prompt

本图按完整方法设计绘制，不放实验结果、进度、性能数字或实现状态。参考用户提供的两张机器人论文框图的分区、留白、网络图标与箭头风格；机器人本体、动作和算法使用 JIT 的设计。英文图标注；中文图注见文末。图中 tube 是概念性的 x-z 投影，不是由本次生成器计算的实验包线。

## 可直接复用的英文 Prompt

```text
Use case: scientific-educational / infographic-diagram.
Create ONE polished, publication-oriented robotics research METHOD OVERVIEW figure. Landscape approximately 3:2, highest available resolution, white background, flat vector-like linework, razor-sharp legible English labels, generous margins. It should resemble a carefully composed IEEE robotics journal figure, not a poster or marketing infographic. The reader must understand BOTH the physical action perturbation and the closed training/data loop. The central concept is learning useful short perturbations to discover physically reached states, improving a jumping controller from these states while retaining old skills, and accumulating successful trajectory witnesses into an empirical jumping tube.

Title, in a single restrained line:
"Iterative Jumping-Tube Expansion through Learned Short-Pulse Exploration"

VISUAL STYLE
Use a white page with four softly tinted panels: pale blue for physical rollout, pale mint for continuation and labeling, pale lavender for learning, and pale warm ivory for tube construction. Panel headings dark navy, large and bold. Main text Helvetica/Arial-like. Mathematical symbols in a clean italic serif if possible. Thin rounded panel borders, minimal shadows, no glossy effects. Use blue for current jumping policy, amber for exploration and pending states, teal for successful witnesses, muted coral for failed tested states, violet for training feedback, light gray for unknown/unexplored space. Convey categories by labels and marker shapes as well as color. All arrows need clear arrowheads. Solid dark arrows carry actions, observations, or physical state data; violet dashed arrows carry training targets, parameter updates, or delayed outcome feedback. Dashed means learning feedback, NOT unimplemented work. Avoid crossing arrows, text overlaps, tiny footnotes, fake charts of measured improvements, watermarks, and decorative filler.

COMPOSITION
Use a well-aligned 2-by-2 panel layout. The left column occupies about 60% of width, the right 40%. Top panels (a) and (b) form the physical acquisition and verification pipeline. Bottom panels (c) and (d) form learning and geometric construction. Give the central data arrows wide gutters. Include small numbered modules and a compact arrow/color legend, without making every box a different color. Keep the tube illustration prominent: it must be large enough to reveal individual curved trajectories.

(a) TOP LEFT: "Phase-aware short-pulse exploration"
Show an observation box labeled "Observation + history" with compact lines "Joint states | base motion | previous actions". Split its flow into:
1. Blue neural-network block "Current jumping policy πᵢ", subtitle "76-D input · frozen during exploration", output arrow "Base action aᵢ".
2. Amber neural-network block "Residual explorer Eφ", subtitle "106-D: 76-D history + 30 privileged features", a small row of three layer icons labeled "256 → 256 → 256 / Swish", followed by "Gaussian → tanh → 4-D residual". Its output is δₜ, not a complete replacement action. A small annotation may say "4 means + 4 scales". The history has three frames; do not imply all 106 features are repeated three times. Show a small additional simulator-state input feeding only the privileged observation branch.
Route the residual through "Event gate + amplitude bound" to a summation circle, then a "Action clip" box. Show the equation clearly below this action path:
"aₜ = clip[aᵢ(oₜ) + mₜ ε ⊙ δₜ, −1, 1]"
Below it write "ε = 0.10 or 0.15 in normalized action space" and the four action channels in their exact order: "steer | rear drive | hip | knee". The amplitude is neither metres nor newton-metres, and is not a percentage of instantaneous policy output.
After the clipped action show "Actuator mapping → MuJoCo dynamics", with a small technically restrained illustration of the JIT robot: a single-track TWO-WHEELED self-balancing vehicle, two inline wheels, compact frame and an articulated hip/knee pendulum mechanism above it. No rider, no quadruped, no humanoid, no four-wheel car. The robot need not be photorealistic. The simulation returns a clearly routed "Next observation" arrow to the observation input. Mark "Control: 50 Hz".
At the bottom of panel (a), add a miniature ONE-jump arc with four event markers in chronological order, labeled "start", "liftoff", "apex", "descent". These are alternative pulse trigger locations for separate trials, NOT four pulses required in one rollout. Put the note "One selected event per rollout". Add a tiny pulse inset showing three successive narrow bars of different heights, bracketed "3 control steps = 0.06 s". Outside these three steps the residual is zero. The residual is sampled anew each pulse step; do not draw a constant three-step force.
The main output arrow from panel (a) to (b) is labeled "Real prefix + complete candidate snapshot s*".

(b) TOP RIGHT: "Same-state continuation and labeling"
Place a compact snapshot card: "s*: physical state + history + event / actuator state". Show "Pulse OFF" followed by "Continue with current πᵢ" and a decision diamond "Jump + stable landing?".
Next to the diamond, a small criterion box reads "Grounded + forward + stable posture for 0.5 s". This is continuous stability after landing, not first contact, and success then ends the rollout.
The "yes" branch goes to a teal cylinder/card "Successful witnesses Wᵢ" and then downward to panel (d). The "no" branch goes to amber "Eligible pending candidates Pᵢ", with a clearly routed arrow to panel (c), labeled "New training starts". Physically terminal prefixes go directly to the failed visited ledger, not into reset support. An inconclusive/error branch goes to a small gray card "Unknown: no negative label".
Include a second compact verification box in the lower part of (b): "Re-test the SAME s* with candidate πᵢ₊₁". It receives the newly trained candidate from (c). Its success arrow joins Successful witnesses; its failed-tested arrow goes to a muted coral card "No witness within training budget". Keep these failed tested states in a small "Visited ledger" card; they must NOT disappear and be repeatedly rewarded as new. Do not equate them with mathematically impossible states.
From the verified outcomes draw a violet dashed arrow back to the explorer update in (c), labeled "Delayed quality feedback". Do not introduce an old π₀…π₆ committee or an oracle choosing among historical policies; only current πᵢ and its newly trained candidate appear.

(c) BOTTOM LEFT: "Retention-aware policy improvement"
Use TWO visibly separate learning paths inside this panel so the reader never confuses their rewards or networks.
Main controller path: a blue/amber two-color data buffer labeled "Old successful starts + new pending starts". Add a small reset-mixture strip labeled "20% full starts + 80% snapshots" and "Phase-balanced snapshots: old / pending". If room permits, place "50 / 50 within each available phase" underneath, referring only to the snapshot portion when pending exists.
Connect to "PPO fine-tuning", subtitle "Warm-start actor from πᵢ" and "Original jump-task reward + KL constraint". Clarify with a small "Fresh rollouts" arrow from this buffer through a miniature simulator to PPO: stored states initialize new episodes; they are not old trajectories directly replayed as on-policy PPO samples. Output is "Candidate πᵢ₊₁". It branches upward to the same-state re-test in (b) and rightward to a diamond "New capability + old-skill retention + valid nominal jump?". An accepted branch produces a prominent blue block "Accepted next policy"; a rejected branch says "Keep πᵢ". A violet dashed outer arrow from Accepted next policy back to Current jumping policy in (a) is labeled "Next iteration". Do not imply automatic acceptance or guaranteed strict improvement at every iteration.
Explorer learning path: an amber/violet block "Explorer PPO update", with small separate neural icons "Residual actor Eφ" and "Own value critic Vψ". Both take the privileged history input. The original controller's critic is not a quality oracle. Show the objective line "rᴱ = 0.25 × new-cell novelty + outcome quality" and a short legend "+1: successful continuation; −1: still failed after attempted refinement". Unknown outcomes are excluded, not punished. A violet dashed arrow brings Delayed quality feedback from (b); another input says "Per-current-policy visited ledger" for novelty. An updated-parameters dashed arrow goes to Residual explorer in (a).
Add a short note inside the panel: "Collect → batch verify → learn → re-test". No gradient should pass through the simulator or between outer-loop policy generations. The dashed arrows represent label and parameter feedback, not end-to-end differentiation through physics.

(d) BOTTOM RIGHT: "Constructing the empirical jumping tube"
Make this panel visually rich but scientifically restrained. Draw a large x-z coordinate plane, horizontal axis "Forward position x", vertical axis "Base height z", no numeric ticks or empirical performance numbers. Draw a ground line, a fixed starting point, and a bundle of smooth SINGLE-jump curves that rise once and descend once, followed by a short nearly horizontal recovery tail ending at a small "Stable 0.5 s" marker. No second jump, rebound hump, landing collapse, or indefinite long tail. Use thin blue lines for the original narrow trajectory bundle and multiple teal lines around and beyond parts of it for additional accepted successful witnesses. A few amber perturbed paths branch near the selected pulse points. A few coral crosses outside the successful bundle indicate tested failures; surrounding white or very light gray areas indicate unexplored space.
The tube is the bundle of real successful trajectory witnesses in x-z projection. Give it a soft visual ribbon impression through many adjacent translucent lines; do NOT fill a convex hull or draw a certified solid safety boundary. Show clearly labeled "Initial tube T₀", "Added successful witnesses", "Tested, no witness", and "Unexplored". Place small sample dots along curves to convey visited cells. Indicate a short amber prefix segment meeting a teal continuation at a labeled point "s*"; the connection must be continuous, not a teleported join.
Above or beside the large plot, use a simple three-step iteration strip "π₀ → π₁ → π₂ → …" with three small trajectory-bundle icons. Caption it "Retain old skills; acquire new recoverable states". This illustrates the method's objective, not measured superiority or a theorem that each single-policy tube is a strict superset.
At the bottom, one compact conceptual statement:
"Tube archive = successful prefix + continuation witnesses"
and a separate small output card:
"Final controller π*: independent perturbed-jump evaluation"
These are TWO different outputs: the archive of successful trajectories and the final single policy. Do not imply the union of all historical witnesses is automatically controlled by the final actor. The final output card may list intended metrics "success rate | height / length | retained capability", without fabricated values.

FINAL QUALITY CHECK
Every panel must connect to the others as a legible closed method loop. Keep node text brief and equations accurate. The strongest visual emphasis should be the short residual action injection, the pending-to-refinement-to-retest loop, and the growing trajectory tube. The figure describes the complete designed method only: no experimental status labels, no implementation badges, no real/sim deployment claims, no diffusion module, no results tables, no invented success percentages, no claim of complete physical reachability. Reference style is the light-color, neural-network-and-robot architecture style of robotics journal method figures, but the composition and all technical content must be original and specific to this JIT method.
```

## 配套中文图注

**图：基于短时残差探索与策略迭代的经验跳跃 tube 构建方法。** 在每轮迭代中，当前跳跃策略固定，探索网络依据包含历史的观测，在指定事件附近输出四通道有界动作残差，并只持续三个控制步。真实前向仿真产生完整候选状态；撤去残差后，首先由当前策略续接验证。已成功的轨迹进入见证集合，未成功但结果明确的候选进入待训练集合。后继策略从当前策略初始化，利用旧成功状态与新候选状态混合重置并通过新仿真进行 PPO 补训，再对相同候选重评；经过旧能力保留与正常起跳检查的策略用于下一轮。新增格子与补训后的任务结果共同反馈给探索网络。图中 x-z 曲线束示意成功前缀及其续接轨迹构成的经验 tube，已访问但尚无成功见证的状态另行保留。成功要求落地后连续稳定前行0.5秒。见证集合与最终单个策略的能力分别评价。

## English manuscript caption

**Overview of iterative empirical jumping-tube construction.** (a) A history-conditioned residual explorer injects bounded perturbations into a frozen jumping policy for three control steps at a selected motion event. The simulator produces a real trajectory prefix and a candidate snapshot containing the complete physical and controller context. (b) With the pulse removed, the current policy is evaluated from the same snapshot. Successful continuations provide trajectory witnesses, while eligible unsuccessful candidates form a pending training set. (c) A successor controller is initialized from the current policy and refined using fresh PPO rollouts from mixed old and new reset states. Same-state re-evaluation provides delayed outcome feedback to the explorer; policy acceptance additionally checks new capability, nominal jumping and old-support retention. The explorer combines per-policy novelty with continuation quality, using its own value critic. (d) Successful prefix–continuation pairs form an empirical tube visualized as an x–z trajectory bundle. Visited states without a success witness remain separate from the successful archive. Task success requires continuous stable forward motion for 0.5 s after landing. The archive and the final single controller are distinct outputs. The trajectory bundles in this method diagram are schematic.

## 绘图语义核对

- 方法图不使用历史实验的数字、曲线或标签；概念曲线仅说明算法。
- `0.10/0.15`是归一化动作残差的逐通道上限；最终动作仍限幅，实际残差可能小于请求残差。
- 每条轨迹选择一个触发事件；三个脉冲步可以给出不同动作。
- 完整状态包括历史、事件和执行器上下文；不得只保留x/z。
- 新格子来自当前策略的探索记账；失败点保留在访问账本，不是成功tube。
- controller任务奖励与explorer奖励分开；质量来自真实续接/补训后的重评，不画成基础critic打分。
- 后继策略重评与保留检查均为必要数据路径；策略拒绝时保留旧策略。
- Tube见证累计与单策略能力不等价；示意图不声称连续空间覆盖或物理边界证明。

依据：`docs/EXPLORATION_PAPER_EXPERIMENTS.md`、`src/jit_dvgc/exploration_network.py`、`src/jit_dvgc/pulse_exploration_runtime.py`、`src/jit_dvgc/pulse_exploration.py`。按用户指示采用完整设计框架，不按实验完成程度分色。
