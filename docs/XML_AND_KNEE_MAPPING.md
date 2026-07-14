# 原始 XML 使用方式与 Knee 动作映射说明

## 1. 模型文件处理原则

本版本只使用：

```text
assets/orange_bike_4kg_horizontal.xml
```

代码不会再执行以下操作：

- 不生成 `orange_bike_runtime.xml`；
- 不删除 STL 或视觉 mesh 节点；
- 不把轮胎碰撞体从 ellipsoid 改成 cylinder；
- 不删除 XML 中的 `sdf_iterations`、`sdf_initpoints`；
- 不通过字符串重写 XML；
- 不通过环境变量切换到另一份 XML；
- 不改变关节范围、keyframe、执行器增益、力矩范围或接触参数。

环境在 `dvgc/env.py` 中直接执行：

```python
self._xml_path = str(Path(self._config.xml_path).resolve())
self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
```

因此 MuJoCo 会按照原 XML 的：

```xml
<compiler meshdir="meshes" .../>
```

读取原来的 STL。你的模型目录中只要保持：

```text
assets/
├── orange_bike_4kg_horizontal.xml
└── meshes/
    ├── base_link.STL
    ├── frontwheel.STL
    ├── rearwheel.STL
    ├── steer.STL
    ├── downarm.STL
    └── uparm.STL
```

即可。当前交付包没有伪造或替代这些 STL；你把自己原有的 `meshes` 目录放回对应位置即可。

## 2. 为什么旧 knee 正半轴会失效

原 XML 中：

```text
knee 初始位置 q0 = 2.5 rad
knee 上限      qmax = 2.5 rad
knee 下限      qmin = -1.5 rad
```

旧 residual 映射的正分支等价于：

```text
q_target = q0 + action_knee × (qmax - q0),  action_knee >= 0
```

由于：

```text
qmax - q0 = 2.5 - 2.5 = 0
```

所以：

```text
action_knee = 0.1  -> q_target = 2.5
action_knee = 0.5  -> q_target = 2.5
action_knee = 1.0  -> q_target = 2.5
```

整个正半轴都对应同一个执行器目标。PPO 即使改变正动作，实际控制量也不改变，这就是“正半轴死区”。

它与 XML 的 `kp=200`、`kv=5` 或 `forcerange=-50 50` 无关；根因是动作符号和初始关节限位冲突。

## 3. 新映射

本版本不修改 XML，只修改策略动作到原 XML knee 位置执行器目标的映射：

```text
q_target = clip(
    q_current - action_knee × Δq,
    qmin,
    qmax
)
```

其中默认：

```text
Δq = 0.20 rad / control step
qmin = -1.5 rad
qmax =  2.5 rad
```

动作含义固定为：

| `action_knee` | 目标变化 | 物理含义 |
|---:|---:|---|
| `+1.0` | `q_target = q_current - 0.20` | 最大幅度屈膝/flexion |
| `+0.5` | `q_target = q_current - 0.10` | 中等屈膝 |
| `0.0` | `q_target = q_current` | 保持当前角度 |
| `-0.5` | `q_target = q_current + 0.10` | 中等伸展/extension |
| `-1.0` | `q_target = q_current + 0.20` | 最大幅度伸展 |

例如在初始上限 `q_current=2.5` 时：

```text
action_knee = +0.25 -> q_target = 2.45
action_knee = +0.50 -> q_target = 2.40
action_knee = +1.00 -> q_target = 2.30
```

因此任何正动作都会产生不同且有效的屈膝目标，不再被 `2.5 rad` 上限全部裁剪。

当 knee 已经屈曲到 `q_current=2.2` 时：

```text
action_knee = -0.50 -> q_target = 2.30
action_knee = -1.00 -> q_target = 2.40
```

所以负半轴也能按幅度控制伸展。只有达到 XML 的真实关节上下限时才会饱和，这是物理限位，不是动作映射死区。

## 4. 为什么使用“相对当前角度”的增量目标

没有采用固定目标区间映射，原因是：

1. 零动作必须保持 reset/snapshot 中原有的 knee 状态，不能一 reset 就强制跳到另一个人为中值；
2. Takeoff、Flight、Landing 的合理 knee 基准不同，固定中值会给共享 Actor 引入阶段偏置；
3. 增量目标使动作近似具有“期望关节运动方向和强度”的含义；
4. `Δq=0.20 rad` 与 `kp=200` 配合时，满动作产生约 `40 N·m` 的静态比例项需求，实际输出仍受 XML 的 `±50 N·m` 执行器力矩范围限制，因此没有绕过执行器物理限制；
5. 参考轨迹中大量正 `action_knee` 与 knee 角度下降阶段相对应，新符号与参考数据的控制方向一致。

## 5. 所有入口如何保证使用同一映射

唯一映射实现在：

```text
dvgc/action_mapping.py
```

环境的训练、评估、可视化和认证均调用 `dvgc/env.py::_action_to_ctrl()`，后者再调用同一个 `knee_position_target()`。不存在单独的 train mapping 或 certify mapping。

映射版本写入统一配置和 policy manifest：

```text
steer_drive_hip_knee.incremental_positive_flexion.v2
```

策略加载时会同时校验：

- action mapping version；
- 原 XML SHA-256；
- config hash；
- candidate bank hash；
- downstream bank hash。

旧版策略若使用 `flex_positive.v1` 或 legacy mapping，不能与本版本直接混用，应重新训练或明确作为旧消融结果保存。

## 6. 原 XML 与 MJX 后端说明

原模型使用 ellipsoid 轮胎碰撞体。旧项目默认使用 `MJX-Warp + IMU/event contact estimation`，所以本版本恢复：

```text
impl = "warp"
contact_mode = "imu"
```

这保证不需要改碰撞几何。需要注意：默认 Warp 路径下，Actor 和阶段滤波器不读取私有 contact buffer；落地/支撑事件由 IMU、位置、速度和事件历史估计。

分支认证中的质量、摩擦和执行器力限扰动只修改运行时内存中的 `MjModel` 数值，用于 domain randomization；不会写回或生成任何 XML 文件。

若未来你的 MuJoCo/MJX 版本已经能够用原 ellipsoid 模型在 JAX backend 暴露所需 contact 数据，可以只在统一配置中切换 backend/contact mode，但仍然使用同一份原 XML，不能再生成几何替代模型。

## 7. 验证测试

新增测试 `tests/test_action_mapping.py`，验证：

- knee 位于 XML 上限时，正动作仍然产生连续不同目标；
- 零动作保持当前角度；
- knee 屈曲后，负动作按幅度伸展；
- 目标仍受原 XML `[-1.5, 2.5]` 限位裁剪。

`tests/test_source_contracts.py` 还检查：

- 环境直接调用 `MjModel.from_xml_path`；
- 唯一模型路径为 `orange_bike_4kg_horizontal.xml`；
- 工程中不存在 `prepare_runtime_xml`；
- 配置和代码中不存在 `orange_bike_runtime.xml`。
