# DVGC Clean Project V2 修改报告

## 一、本次修改的直接原因

上一版为了在缺少 STL 的临时审查环境中完成静态结构检查，额外生成了 runtime XML，并曾对视觉 mesh 和轮胎碰撞几何作运行时替代。这个处理不符合项目的真实使用条件：用户拥有完整 STL，模型文件本身就是唯一正式物理模型。

本版本已经彻底撤销该方案。原始 XML 与用户上传文件逐字节一致，SHA-256 为：

```text
ceda4f7e43895cee09bd7582d9b46688971740e74b6553615739f980eb098138
```

## 二、模型文件相关修改

### 已删除

- `assets/orange_bike_runtime.xml`
- `dvgc/xml_sanitize.py`
- `dvgc/assets.py`
- `prepare_runtime_xml()`
- 缺失 mesh 自动删除逻辑
- ellipsoid → cylinder 碰撞体转换逻辑
- XML option 属性删除逻辑
- `ORANGE_BIKE_XML` 环境变量替代路径

### 当前唯一模型路径

```text
assets/orange_bike_4kg_horizontal.xml
```

### 当前加载方式

```python
self._xml_path = str(Path(self._config.xml_path).resolve())
self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
```

训练、候选构造、评估、认证和自然起点评估均通过同一个 `OrangeBikeDVGC` 环境读取这一路径。

`cli.prepare_project` 现在只做原 XML 的只读哈希与结构审计，不再创建模型副本。

## 三、Knee 正半轴死区的根因

原 XML 的 knee 初始位置与上限相同：

```text
q_initial = 2.5 rad
q_max     = 2.5 rad
```

旧映射在正动作分支使用：

```text
q_target = q_initial + action × (q_max - q_initial)
```

因为括号内为 0，所以所有正动作都得到 `q_target=2.5`。这意味着 Actor 输出 `0.1、0.5、1.0` 时，执行器收到完全相同的目标，梯度探索在物理层面没有区别。

## 四、Knee 新映射

新映射定义为：

```text
q_target = clip(q_current - action_knee × 0.20, -1.5, 2.5)
```

动作顺序仍为：

```text
[steer, rear-wheel drive, hip, knee]
```

符号含义：

- `action_knee > 0`：减小 knee 角，屈膝；
- `action_knee < 0`：增大 knee 角，伸展；
- `action_knee = 0`：保持当前 knee 角。

这套映射不修改 XML，不绕过 joint limit，也不绕过 `±50 N·m` 的 actuator force range。

### 为什么改成增量目标

增量目标以当前 knee 状态为中心，因此：

- 自然起点的零动作不会把 knee 强制拉到人为中值；
- snapshot reset 后零动作保持 snapshot 的真实姿态；
- 同一共享 Actor 在 Takeoff、Flight 和 Landing 阶段具有一致动作语义；
- 正负动作都按幅度有效，只有到达真实 joint limit 才会饱和。

### 数值例子

初始 `q_current=2.5`：

```text
+0.25 -> 2.45
+0.50 -> 2.40
+1.00 -> 2.30
```

屈曲后 `q_current=2.2`：

```text
 0.00 -> 2.20
-0.50 -> 2.30
-1.00 -> 2.40
```

## 五、版本与旧策略兼容性

动作映射版本已经从：

```text
steer_drive_hip_knee.flex_positive.v1
```

更新为：

```text
steer_drive_hip_knee.incremental_positive_flexion.v2
```

旧 checkpoint 的动作语义与新环境不同，不能直接作为正式结果继续训练或认证。旧结果可以保留为历史消融，但新的 Landing → Flight → Takeoff → Approach 主链应使用 V2 mapping 重新训练。

Policy manifest 会校验 mapping version 和原 XML SHA-256，避免训练、评估和认证混用不同动作语义。

## 六、后端选择

由于原 XML 使用 ellipsoid 轮胎碰撞体，本版本默认恢复旧工程实际使用的：

```text
impl = warp
contact_mode = imu
```

这样不需要转换碰撞体。物理碰撞仍由原模型计算，但默认 phase/contact estimator 使用 IMU、位置、速度与事件历史，不把私有 contact buffer输入 Actor。

认证分支中的质量、摩擦、执行器力限缩放只发生在内存中的 `MjModel` 实例上，不会写回 XML。

## 七、修改文件清单

主要修改：

- `dvgc/config.py`
- `configs/default.json`
- `dvgc/env.py`
- `dvgc/action_mapping.py`（新增）
- `dvgc/model.py`
- `cli/prepare_project.py`
- `tests/test_action_mapping.py`（新增）
- `tests/test_model.py`
- `tests/test_source_contracts.py`
- `tests/test_optional_runtime.py`
- `README.md`
- `PROJECT_SUMMARY.md`
- `docs/XML_AND_KNEE_MAPPING.md`（新增）
- `docs/VERIFICATION_PROTOCOL.md`
- `docs/REMOVED_FILES.md`

删除：

- `assets/orange_bike_runtime.xml`
- `dvgc/xml_sanitize.py`
- `dvgc/assets.py`
- `docs/model_prepare_report.json`

## 八、验证结果

静态与单元测试：

```text
11 passed, 1 skipped
```

已验证：

- 上传 XML 与工程 XML 字节一致；
- 工程不存在 runtime XML；
- 环境直接使用 `MjModel.from_xml_path`；
- 正 knee 动作在 `q=2.5` 时仍产生不同目标；
- 负 knee 动作在屈曲后按幅度伸展；
- 零动作保持当前 knee 位置；
- 原 joint limit 仍然裁剪目标；
- Policy manifest 使用新 mapping version 与原 XML hash。

被跳过的是动态 MJX smoke test，因为当前审查容器没有 MuJoCo Playground，并且交付包不包含用户自己的 STL。将 STL 放入 `assets/meshes/` 并在训练服务器安装原有环境后，该测试会自动执行。

## 九、开始训练前必须执行

```bash
python -m cli.prepare_project
python -m pytest -q
```

确认 `tests/test_optional_runtime.py` 在你的训练环境中不再被跳过后，再运行短 rollout。不要直接从旧 checkpoint 开始长时间训练，因为动作映射版本已经变化。
