# 同身份迁移接口：零物理预检

本轮只确认标准 checkpoint 的离线导出与保存观测推理一致性。**两引擎任务适配尚未就绪，未运行新物理、训练或 TEST，不是新的 sim2sim 成功结果。** 现有 01966080 属于另一个父源及 PhysX 训练谱系，不能替代本线学生。

- [身份、配置、运行时序与推理结果](preflight.json)：源为 retention π₀ 的 `bridge_baseline/checkpoints/transition_0`；Actor `a06acbc8…`、normalizer `db84aad7…` 与源同包核验。8条既有 D1 TRAIN 教案共525条76D观测，源 Brax 与导出 NPZ 的最大动作绝对差 `4.76837158203125e-7`，门槛 `3e-5`。教案为组合教师轨迹，这里仅复用其真实前置观测；不把教师动作当 π₀ 动作或成功能力。
- [现有 Isaac Actor 独立读取结果](external_actor_parity.json)：只读加载外部 `policy_runtime.py::Actor`，读取新导出的NPZ，使用同525条保存观测核对。该检查没有实例化 `Observation`、`PhysxTask` 或着陆scorer。
- [两引擎入口与剩余接口](engine_entrypoints.json)：旧导出器会写共享目录，MJX probe 固定另一源环境和 x=1.5 初态，PhysX任务尚无本线统一恢复判据。没有修改或调用这些物理入口。

新增入口为 `JIT/cli/export_sim2sim_policy.py`，必填 checkpoint、task-config、observations-manifest、新 output、repository，以及 payload／Actor／normalizer SHA。normalizer只从同一checkpoint获取；支持标准 `CheckpointPayload`，不猜测BC tuple格式。保存观测清单必须声明相同观测字段、XML身份、`final_test_used=false`及TRAIN／SOLVER_DEV／DEV角色；先检查全部角色，再打开观测文件。输出目录存在即拒绝覆盖。

时序来源需要分开：绑定的 XML 原文 `timestep=0.001`；当前核验的 `model.py::load_host_model` 覆盖为 `SIM_DT=0.005`，`env.py` 以 `CTRL_DT=0.020`、`SIM_DT=0.005` 创建runtime，4物理子步对应1控制步。JSON未声明这些时序字段。预检保存相关源码SHA并检查覆盖及乘法关系；没有实例化环境，不能把该静态核验称为运行时物理复现。

任务合同携带真实up/down配置及25步稳定恢复条件，保留前速≥0.5 m/s等门槛。坐标合同注明本体局部gyro、世界坐标角速度与 `v_root = v_com - omega × r_com` 恰好一次转换。**这些是目标适配器需遵守的合同；76D推理通过不证明历史/FIFO、观测适配、接触或两端scorer已一致。** checkpoint绑定的旧配置也不替代独立训练／评价runtime override声明。

验证：`JAX_PLATFORMS=cpu .../python -m pytest -q JIT/tests/test_sim2sim_export.py`，10项通过（先观察预期失败，再实现通过）；覆盖身份、配置、TEST拒绝、观测维度/非有限值、旧输出保护、实际推理与XML/runtime时序分离。该验证无物理成本，不并入T1交互预算。

原始绝对目录：`/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/sim2sim_preflight_001/`。当前完整bundle为 `pi0_export_with_timing/`，含NPZ与来源配置；早一版 `pi0_export/` 保留未覆盖。此compact目录只含小JSON与说明，不含权重或观测数组。后续T1标准checkpoint可通过同入口导出，目标端评价仍需单独冻结适配合同及预算。
