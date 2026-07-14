# OrangeBike DVGC Clean Project

这是删除历史过程脚本后保留的唯一正式工程。方法说明见 `PROJECT_SUMMARY.md`，旧文件处理见 `docs/REMOVED_FILES.md`。

## 1. 环境准备

在已有 MuJoCo Playground GPU 环境中安装本项目：

```bash
pip install -e .
python -m cli.prepare_project
python -m python -m pytest -q
```

本项目只读取 `assets/orange_bike_4kg_horizontal.xml`，不生成 runtime XML，也不修改碰撞几何。正式模型使用 4 kg 负载和 hip/knee `±50 N·m` 限幅，默认使用 `impl="warp"`、`contact_mode="imu"`；Actor observation 不读取 oracle contact。请将你已有的 STL 保持在 XML 指定的 `assets/meshes/` 目录。

模型与 knee 动作映射的完整说明见 `docs/XML_AND_KNEE_MAPPING.md`。

## 2. 单阶段命令

```bash
python -m cli.build_candidates \
  --phase landing \
  --target 96 \
  --bank artifacts/landing_candidates.pkl

python -m cli.train \
  --stage landing \
  --bank artifacts/landing_candidates.pkl \
  --run runs/landing

python -m cli.certify \
  --phase landing \
  --policy runs/landing/policy \
  --candidate-bank artifacts/landing_candidates.pkl \
  --output-bank artifacts/landing_tube.pkl

python -m cli.audit \
  --phase landing \
  --policy runs/landing/policy \
  --bank artifacts/landing_tube.pkl \
  --output runs/landing/audit.json
```

Flight、Takeoff 和 Approach 必须显式提供已认证下游 bank：

```bash
python -m cli.certify \
  --phase takeoff \
  --policy runs/takeoff/policy \
  --candidate-bank artifacts/takeoff_candidates.pkl \
  --downstream-bank artifacts/flight_tube.pkl \
  --output-bank artifacts/takeoff_tube.pkl
```

## 3. 完整顺序

```bash
bash scripts/run_backward_bootstrap.sh
```

脚本按 Landing → Flight → Takeoff → Approach → natural-start 顺序执行。后续阶段通过 `--resume` 继承前一阶段共享 Actor，并混入已认证下游 rehearsal。

## 4. 认证原则

- Candidate bank 与 downstream certified bank 是两个不同参数；
- `training_only=True` 的 velocity seeds 和 rehearsal states 永不参与认证；
- Chain 与 Final Recovery 分别统计；
- Chain 事件锁存，不读取最后一步瞬时值；
- Tube entry 使用下游 final-safe 状态的标准化距离；
- build 与 audit 使用不同 seed namespace；
- timeout 单独报告，不能写成物理 Failure；
- policy manifest 校验 action mapping、原始 XML、config 和 bank 版本；
- 全部入口直接读取 `orange_bike_4kg_horizontal.xml`，禁止 runtime XML 或替代几何。

## 5. 参考轨迹的允许用途

允许：候选范围、阶段姿态 envelope、动作方向/执行器诊断、消融参考。

禁止：逐点 CoM/姿态轨迹跟踪 reward、用“接近参考”替代经验可恢复标签、用 velocity-seeded 辅助状态进行正式认证。

## 6. 输出

- `artifacts/*_tube.pkl`：带 Chain/Final Beta posterior 的版本化 Tube；
- `runs/*/policy/`：不可变 policy bundle；
- `runs/*/audit.json`：独立 Tube 质量报告；
- `runs/natural_start_evaluation.json`：最终自然起点成功率；
- `docs/reference_report.json`、`docs/reference_phase_envelopes.csv`：参考轨迹审计；
- `docs/model_report.json`：模型结构审计。
