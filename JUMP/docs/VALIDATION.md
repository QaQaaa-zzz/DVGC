# 当前验证

日期：2026-09-14。新任务：前向行驶下窄障碍完整腾越。旧 JIT 和 STTW_CONTROL 未修改。

- 基线 JIT action/observation：15 passed。
- 新项目70项行为测试＋继承通知3项：`PYTHONPATH=JUMP/src:JIT/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python -m pytest JUMP/tests JIT/tests/test_error_notifications.py -q` → **73 passed in 4.41s**。覆盖预算预留/错误计费/启动错误封口、oracle事件链、接触辅助函数、接地资格/触发辅助类及图件。runtime完整FIFO/reset不是这些辅助类测试的独立证明。
- 两个原始checkpoint身份/载入/76D推理及场景编译已通过；上述检查积分0步。
- 首批真实动态资格实验：12个声明情况全部完成、仅2条不同控制轨迹，0接地资格、0完整成功、12任务失败、0未知。实际216控制步/828物理步。没有实际触发，不能据此比较六个delay。
- 单变量近起点追加：只把初始x从1.8m改为2.5m，两个null-trigger情况完成；仍0接地资格。实际54控制步/213物理步。
- 模型检查：nq12/nv11/nu4，总质量5.091kg，payload2kg，原执行器及摩擦保留，仅显式改变障碍几何。

| 情况 | 失败终点 | 原因 | 控制步/物理步 |
|---|---:|---|---:|
| 初始x1.8，pi0，每个触发设置都未到触发门 | .405s | yaw_limit | 每条21/81；六条126/486 |
| 初始x1.8，repair0000，每个触发设置都未到触发门 | .285s | yaw_limit | 每条15/57；六条90/342 |
| 初始x2.5，pi0，null trigger | .675s | roll_limit | 34/135 |
| 初始x2.5，repair0000，null trigger | .390s | yaw_limit | 20/78 |

总实际270控制步/1041物理步；部分终止控制按ceil(substeps/4)计费，未执行子步不称实际积分。PPO和监督优化器均0步。两批分别预留4800和800，余量已释放；历史账本不删除。

首批完整轨迹中pi0在.080s首次轮触地、.240s双轮净空均>2mm；repair0000在.075s首次轮触地、.190s双轮净空均>2mm，signal始终0。最大连续双轮接触观测跨度均仅.005s，未达.1s资格。可描述为触地后无指令腾空，不能把初始化间隙或该未资格运动称为正式liftoff，更不能在本批区分关节动作与反弹的因果贡献。

实际动态轨迹逐项核对76D历史：初始全零，mask依次000→001→011→111；动作/速度/障碍距离/高度来自上一控制端点，未发现FIFO错位。这与原JIT15项基础测试、两个真实checkpoint/reset零步检查是三类独立证据。

CPU每子步后mj_forward使接触与传感值对齐当前状态，旧Warp RK4采样契约不同；尚未完成同后端/采样消融。因此结论限于“两个冻结Actor不能直接通过当前新资格设置”，不是旧策略能力消失、物理不可行或新主线失败。

- [首批全覆盖索引](../runs/qualification/initial_20260914T114808Z/analysis/INDEX.md)：12组PNG/PDF与12条JSONL，原图不覆盖。
- [近起点追加索引](../runs/qualification/near_start_20260914T115254Z/analysis/INDEX.md)：2组PNG/PDF与2条JSONL，包含yaw/wz。
- [可版本管理的紧凑证据与原件哈希](evidence/qualification_initial.json)。完整原始结果留在服务器。
- 两批watcher心跳、终态退出及completion_notified=true已核对，delivery_errors均为空；这是通知服务送达证据，不宣称用户已看到弹窗。
- 独立代码审查无阻断项；启动错误回执缺口已修复并复审通过，旧结果不重跑。
