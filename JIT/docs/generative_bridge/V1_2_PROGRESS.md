# v1.2 进度快照

旧训练已按用户要求停止，历史结果保留。新流程 `campaign_0001` 使用代码 `16d4641`，当前等待 GPU 空闲执行基础模型名义验证，G/PPO 新训练均为0。

指定源为 `phase_u_v4_speed2_roll400_missed200_9977856_seed820701_20260826/checkpoints/transition_4988928`。零训练绑定保留 Actor/normalizer；64个固定观测动作完全一致。170项CPU测试通过；尚无本阶段GPU验证或研究性能结果。

三臂分别为A：PPO+keep；B：累计教案+PPO+keep；C：累计教案+预热+PPO+keep。各128,000 PPO步，C为预先指定主候选。上限4,421,600物理步、24小时、22,000 G更新和2,000预热更新。等PPO步数不等于等总成本，最终TEST未打开。

历史丢失根见 [lost_roots_v1_1.csv](lost_roots_v1_1.csv)：recovery_0002丢失core9/protected3/new1，series第1轮丢失core8/protected7/new0。共28条记录、24个不同根，全部与原采用报告匹配；没有把这些旧源标签导入新源。缺失字段写null。

详细身份、预算与限制见 [V1_2_PROGRESS.json](V1_2_PROGRESS.json)。本页是日期快照，实时状态在服务器 `generative_bridge_v1_2_20260928/campaign_0001/status.json`。
