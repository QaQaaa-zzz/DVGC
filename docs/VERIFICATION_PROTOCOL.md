
## 模型一致性前置条件

- 训练、评估、认证和可视化必须直接读取 `assets/orange_bike_4kg_horizontal.xml`；
- 禁止创建或引用 runtime XML；
- policy manifest 中的 XML SHA-256 必须与当前原始 XML 一致；
- STL 由原 XML 的 `meshdir="meshes"` 解析，不允许删除视觉 mesh 或转换碰撞体；
- action mapping 必须为 `steer_drive_hip_knee.incremental_positive_flexion.v2`。

# DVGC 验证协议

1. `python -m cli.prepare_project` 后检查 XML/reference report。
2. `python -m pytest -q` 必须通过；动态环境缺失时只允许 optional runtime test 被 skip。
3. 每个 candidate bank 必须保存 source phase、training-only 标志和完整 PolicyState。
4. 训练不会改写 bank 标签。
5. 只有冻结 policy 的 `cli.certify` 可以更新 Chain/Final posterior。
6. Flight/Takeoff/Approach 缺少 downstream bank 时认证必须拒绝运行。
7. 至少 8 branches 后才能出现非 Unknown 标签；最多 32 branches。
8. build 和 audit seed namespace 不得相同。
9. audit 必须报告 precision、recall、coverage、calibration、false progress、missed success。
10. natural-start 评估只允许使用 Approach 完成后的冻结共享 Actor。
11. 任何阶段若下游固定评估退化，先做 rehearsal/回滚，不得继续向前扩展。
12. 训练 return、视频、单条轨迹和文件名都不是 Tube 认证证据。

## 本地执行顺序

```bash
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate
```

长 PPO 脚本只接受与当前源码指纹、配置 hash 和权威 XML hash 一致的 `docs/RUNTIME_GATE.json` PASS 报告。runtime gate 分别记录零/随机动作物理失败与 timeout，并执行 snapshot round-trip、short PPO compile/run/resume、policy save/load 和确定性推理检查。
