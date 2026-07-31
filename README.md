# OrangeBike DVGC

DVGC 当前主线面向精简的 IEEE RA-L 方法验证：独立阶段专家产生可追溯的阶段入口证据，随后进行 phase-balanced distillation、统一 Tube-RSI PPO，并对冻结后的最终共享 Actor 重新执行端到端 Final-Recovery 认证。

## 权威约束

- 模型：`assets/orange_bike_4kg_horizontal.xml`
- 负载：4 kg
- hip/knee 力矩限幅：`±50 N·m`
- 动作顺序：`[steer, rear-wheel drive, hip, knee]`
- 运行环境：`/home/qy/mujoco_playground/.venv/bin/python`
- 不重建或升级已有虚拟环境，不生成替代 XML，不扩大 Tube matcher 半径

## 当前方法主线

1. 分别训练和冻结 Landing、Descent、Apex、Ascent、Takeoff 等局部专家。
2. 中间阶段仅以有效 next-stage entry 作为局部目标；只有 Landing/Descent 的独立 Final-Recovery 证据可形成正式 Tube。
3. 将五阶段 snapshot 按阶段质量均衡组成 `phase_balanced_tube_rsi_reset_bank`。
4. 使用各冻结专家生成 teacher action，先进行 phase-balanced distillation。
5. 从均衡 reset bank 进行一次有界 joint RSI PPO；保持 normalizer 冻结，并约束各阶段动作漂移与 Landing/Descent retention。
6. 仅当统一策略通过提升门槛后，才进行独立的 Final 4→8→32 分支认证；最终冻结共享 Actor 的复认证结果才可称为 JEL。

当前入口：

```bash
bash scripts/start_corrected_apex_unified_rsi_followons.sh
```

其内部依次调用：

- `scripts/run_corrected_apex_unified_rsi_pipeline.sh`
- `cli.preflight_phase_balanced_unified_rsi`
- `cli.train_phase_balanced_unified_rsi_pilot`
- `scripts/run_final_shared_jel_audit.sh`

## 验证

在已有 Ubuntu 训练环境中执行：

```bash
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate
```

`runtime_gate` 必须与当前源码、配置和 XML fingerprint 一致。长 PPO 前必须通过模型加载、reset/step、snapshot round-trip、确定性推理和短 PPO compile/run/resume gate。

## 目录

- `dvgc/`：环境、snapshot、策略、rollout、认证与当前主线公共实现
- `cli/`：正式入口及仍需保留的研究构建工具
- `scripts/`：当前流水线、状态检查和基础预检
- `tests/`：源码合同、静态逻辑和运行时回归测试
- `docs/EXPERIMENT_STATE.md`：当前实验状态与下一步
- `docs/CURRENT_MAINLINE.md`：当前主线、RSI 和 snapshot 来源
- `archive/pre-clean-20260731`：清理前完整仓库快照

`runs/` 与 `artifacts/` 是本地训练资产，已由 `.gitignore` 排除，不应提交到源码历史。
