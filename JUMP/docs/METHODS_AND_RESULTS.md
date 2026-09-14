# 方法与结果台账

| 方法/阶段 | 目的与差异 | 状态与结果 | 原始证据 |
|---|---|---|---|
| 冻结JIT Actor外部触发资格 | 原机器人/观测/动作，窄障碍，先真实接地后可控延迟触发，逐子步全任务判据 | completed；12设置只有2条unique轨迹，0资格，全部在触发前yaw_limit，216控制步 | [首批索引](../runs/qualification/initial_20260914T114808Z/analysis/INDEX.md) |
| 接近初始位置单变量诊断 | 仅initial.x由1.8改为2.5，两个策略各null-trigger一次 | completed；仍0/2资格，pi0 .675s roll_limit、repair0000 .390s yaw_limit，54控制步 | [追加索引](../runs/qualification/near_start_20260914T115254Z/analysis/INDEX.md) |
| 同状态观测/后端兼容性诊断 | 排除host最终forward与旧Warp传感采样差异 | 建议下一阶段，尚未执行；不能把上述失败全归因于原Actor | PROJECT.md，VALIDATION.md |
| 真仿真准备与触发窗口 | 同一真实前缀比较零准备与转向/驱动准备 | 设计已批准；资格通过后实施 | PROJECT.md |
| 集成任务结果预测器 | 94D输入，5×MLP，预测完整成功和真实离地状态 | 未实现、未训练 | PROJECT.md |

既有JIT跳跃见证和策略回测仅作为底层候选背景，旧first_valid_landing不能迁移为新任务成功。新项目保留失败、未知和全部成本，有限搜索没有成功不等于物理不可行。

首批signal0下出现触地后再次腾空，但没有满足.1s连续双轮接地资格，全部触发时间为null。不同delay的重复轨迹不是独立种子，不能推断选时收益或窗口宽度。新主线继续前需要先验证准备控制与触发接口；世界模型/上层训练预算尚未使用。
