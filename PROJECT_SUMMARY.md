# OrangeBike DVGC 项目总结

## 1. 研究目标

本项目研究单轨双轮机器人完整跳跃中的阶段可达性、恢复能力与经验能力包线。当前 RA-L 主线不再采用“一个共享 Actor 按 Landing→Flight→Takeoff→Approach 顺序持续向前训练”的旧方案，而采用：

1. 独立阶段专家获得局部能力；
2. event-aligned next-stage entry 建立中间阶段证据；
3. 独立分支审计形成 proposal support 或正式 Tube；
4. phase-balanced distillation 合并专家行为；
5. phase-balanced Tube-RSI 训练最终共享 Actor；
6. 对冻结后的共享 Actor 重新进行端到端 Final-Recovery 认证，建立最终 JEL。

## 2. 权威模型与边界

- XML：`assets/orange_bike_4kg_horizontal.xml`
- 负载：4 kg
- hip/knee 执行器限幅：`±50 N·m`
- 动作顺序：`[steer, rear-wheel drive, hip, knee]`
- Actor 不读取 oracle contact；参考轨迹只用于候选范围、阶段 envelope 和物理诊断
- 不生成替代 XML，不扩大 matcher 半径，不把 proposal support 称为 Tube

## 3. 阶段证据的角色

### Landing

Landing 使用冻结策略完成稳定恢复，独立审计后的 Final-safe snapshot 构成正式 Landing Tube。

### Descent

Descent 以进入 Landing/C_L 并最终恢复为目标。独立审计后的 Final-safe snapshot 构成正式 Descent Tube；当前正式来源为 schema/provenance 已规范化的 Tube v6。

### Takeoff、Ascent、Apex

这三个阶段只要求进入下一个局部阶段：

- Takeoff→Ascent
- Ascent→Apex
- Apex→稳定物理 Descent entry

它们的 snapshot 经过冻结控制器和独立 branch 审计，但角色仍是 `stage_entry_certified_proposal_support`，不是正式 Tube。

## 4. 当前统一 RSI

五阶段 source bank 被转换为一个 `phase_balanced_tube_rsi_reset_bank`：

- 每个阶段总采样质量为 20%；
- 阶段内部先按 parent trajectory 等权，再按 parent 内状态等权；
- 所有输出 snapshot 均标记为 `training_only=True`；
- 输出记录移除嵌入的认证统计与 safe claim，仅保留来源哈希和角色；
- PPO 根据 `phase_rsi_stage` 为每个 reset 使用对应局部目标。

统一策略先由冻结专家 teacher action 进行 phase-balanced distillation，再执行有界 joint RSI PPO。当前 corrected pilot 为 4,096 effective steps，并要求：

- observation normalizer 冻结；
- Landing 和 Descent 固定 Final retention 不下降；
- 每个阶段动作漂移 RMS≤0.02、max≤0.05；
- 无 nonfinite；
- 上游或总体固定 Final 证据提升。

未达到门槛时停止，不自动进入无界追加训练。达到 `PASS_PROMOTE` 后才启动最终共享策略 JEL 审计。

## 5. 当前正式入口

```bash
bash scripts/start_corrected_apex_unified_rsi_followons.sh
```

关键代码：

- `cli/build_phase_balanced_tube_rsi_bank.py`
- `cli/build_phase_balanced_teacher_dataset.py`
- `cli/train_phase_balanced_distillation.py`
- `cli/preflight_phase_balanced_unified_rsi.py`
- `cli/train_phase_balanced_unified_rsi_pilot.py`
- `scripts/run_corrected_apex_unified_rsi_pipeline.sh`
- `scripts/run_final_shared_jel_audit.sh`

详细 snapshot 来源与 RSI 过程见 `docs/CURRENT_MAINLINE.md`。实时实验状态见 `docs/EXPERIMENT_STATE.md`。

## 6. 验证要求

在已有 Ubuntu 训练环境中执行：

```bash
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate
```

正式长训练必须保持 XML、配置、action mapping、policy、bank 和 source fingerprint 一致。snapshot restore 需要同时验证连续状态误差与离散事件字段一致性。

## 7. 归档

清理前完整仓库保存在：

```text
archive/pre-clean-20260731
```

当前分支只保留主线实现、必要构建工具、验证测试和当前实验状态；已结束的 controller、旧总控路线及一次性实验入口应通过归档分支和 Git 历史回溯。
