# JIT 环境与奖励方法名称

## `jump_ori`

`jump_ori` 是本机 2026-09-28 Phase U speed2 从零训练所用的**环境、物理模型、重置、动作、奖励及训练配置**的固定名称。机器可读身份在 [jump_ori.json](jump_ori.json)，配置原件在 `/home/qy/DVGC/JIT/runs/code_snapshots/phaseu_speed2_dense10m_deferred_20260928/JIT/configs/phase_u_speed2_dense_10m_deferred_seed820701_20260928.json`。

- 源码提交：`a89b5020cca5fa9e93bb4d9c2fee4b889adfa19f`；该运行的配置文件 SHA-256：`6ccff2cb79adbfa6b14a80785f16b449ff8cef44ee4a5428a95f3f0af742a4ab`。
- 环境：`propulsion_ascent`、MuJoCo Warp、`orange_bike_4kg_horizontal.xml` 和 `reference_jump.csv`；配置原件记录完整的接触容量、事件、物理限制、观测和动作契约。
- 奖励实现：冻结快照的 `JIT/src/jit_dvgc/rewards.py`，SHA-256 `6ca5acd949326866c2e1645c645497a8b71d4bab5498fe1c35a4d1d9b6774e01`；逐项权重、裁剪、失败和终止罚以配置原件及其锁定代码为准。
- 示例训练运行：`/home/qy/DVGC/JIT/runs/phase_u/phase_u_v4_speed2_denseckpt_deferred_10002432_seed820701_20260928`，从零训练 10,002,432 步。此路径是来源例子，`jump_ori` 名称本身不选择 checkpoint，也不表示已达成落地任务。

此名称不回写旧运行、不修改奖励计算。后续若改变任何固定身份，另取新名称或版本，不移动 `jump_ori` 的指向。机器全局查询入口为 `jump_ori` 命令；也可直接读取 `/home/qy/DVGC/JIT/runs/model_registry/jump_ori/manifest.json`。
