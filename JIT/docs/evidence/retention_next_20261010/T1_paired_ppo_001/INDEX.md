# T1有限PPO：工程入口失败，机制筛查未完成

生产配置commit `2eb8b8f`，pilot seed `1010257101`。2026-10-11启动；31.45秒后reset子进程停止。`FourPoolReset`合并保存状态与新reset状态时发生数组形状不一致；失败在步进前。零物理调查已定位为随机键表示：新reset的uint32(2,)与快照恢复的标量typed key混合；纯CPU已复现。34份库存快照持久字段形状一致。

- 两臂正式训练均0/32000；工程PPO更新均0；没有学生终点评价。
- 新增charged保守保留200，D2累计1123052/2000000。实际完成步数没有子收据，不退款、不把预留当实际交互。
- `t1_passed=null`、`published_policy=null`；正式15运行与TEST关闭，无自动重试。
- TensorBoard6032 HTTP可访问，但reward/loss未产生，不能称训练标量验收通过。
- 桌面watcher已送达两条失败通知，delivery_errors为空。STTW原任务保持运行。

原运行与日志保留于[原运行目录](/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/T1_paired_ppo_001)。[状态](status.json)、[固定配置](plan.json)、[失败栈](reset.log)、[只读报告](summary.json)、[来源身份](source_manifest.json)。该失败属于工程接口，不能当作JIT机制负结果。

当前无新增评测轨迹，因此没有T1 XY图；不得称完整控制实验交付。T0使用既有轨迹的[重点根XY](../T0_saved_diagnostics_002/xy_focus.png)仍可查。

修复仅在四池共同边界把随机键统一为typed表示，保留key bits与PRNG实现。新增CPU测试调用真实构造器；历史测试通过__new__创建池，未覆盖这一合并步骤。修复后GPU检查尚未运行；原生产source_lock指向失败时版本，不能用修复后源码冒充原运行通过。
