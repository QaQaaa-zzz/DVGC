# 落地后连续稳定前行恢复

2026-09-14 用户要求：落地后至少稳定向前运行 2 秒。新实验成功条件为 `stable_forward_recovery`，历史 `first_valid_landing` 配置与原始结果不改。

每控制步 0.02 秒；首次有效接触为计时零点，之后连续 100 步满足：车体前向速度至少 0.5 m/s，侧倾绝对值不超过 10°、俯仰不超过 15°，两轮地形间隙均不超过 0.02 m，轮子穿透不超过原来的 0.01 m，无禁止接触或非有限状态。任何条件中断，计时归零。成功还要求相对首次接触前进至少 1 m；现有摔倒、姿态硬限制与超时终止保留。接触几何是每控制步的判据，不宣称子步级连续稳定证明。

新配置取消首次接触强制成功；仅在连续稳定计数增长时给予恢复 tick 奖励，保留既有下降阶段姿态、角速度、动作平滑、前进与失败奖励。旧快照仅作为 reset guidance，不是新标准下的成功证据；恢复计时和成功标记清零，观测历史、物理状态与已有接触历史保留。

预声明训练：从 repair_0000 Actor 和 normalizer 初始化，新 Critic/optimizer；128 环境、每轮采集 25 步、训练 256000 transitions，64k/128k/256k 保存与评价。20% 固定跳跃起点，80% 历史快照（上下游各半）。没有本轮新 pending 的额外固定配额。采用旧累计 witnessed support 作为广覆盖起点，其中 witnessed 仅指历史首次落地语义。每个 checkpoint 固定 16 状态开发面板；这不能代替完整自然起点与历史回测的接受验证。总交互上限 275200，单次 episode 400 步，无自动延长，不打开最终 TEST。

配置：`configs/descent_stable_forward_2s.json`、`configs/stable_landing_recovery_20260914.json`。运行入口：`runs/experiments/stable_landing_recovery_20260914/INDEX.md`。TensorBoard 使用已有 CPU 日志桥读取原始 metrics.jsonl/episode_metrics.jsonl；不改变生产依赖。

CPU 验证覆盖：完整 100 步才成功、首次接触不计时、低速/倾斜/离地/禁止接触重置计时，旧下降与训练配置回归。训练完成与否以执行状态文件为准；通过 CPU 测试不代表已学会稳定落地。
