# 旧工程文件清理清单

旧压缩包中的 25 个文件不再作为工程入口。必要逻辑已合并到新的单一模块，其余历史实验代码全部删除。

| 旧文件 | 处理结果 |
|---|---|
| `orange_bike_dvgc_mjx.py` | 核心逻辑整理为 `dvgc/env.py`；删除 V2–V5 分支和重复默认配置。 |
| `dvgc_common.py` | 由 `dvgc/bank.py` 和 `dvgc/config.py` 取代。 |
| `dvgc_runtime.py` | 保留必要 PPO/推理工具为 `dvgc/runtime.py`。 |
| `dvgc_viability.py` | 重写为 `dvgc/viability.py`，使用 Physical-Belief 特征和 Final posterior。 |
| `takeoff_signals.py` | 精简保留为 `dvgc/signals.py`。 |
| `takeoff_reward_profiles.py` | 精简保留为 `dvgc/rewards.py`；仅一个正式 Takeoff reward。 |
| `mjx_xml_sanitize.py`、`xml_assets.py`、runtime XML 生成逻辑 | 完全删除。工程直接读取用户提供的原始 XML，不再清洗、复制或替换模型。 |
| `train_dvgc_ppo.py` | 重写为 `cli/train.py`。 |
| `certify_dvgc.py` | 重写为 `cli/certify.py`，强制显式 downstream bank、Chain/Final 双认证。 |
| `init_dvgc_bank.py`、`collect_dvgc_rollouts.py` | 合并为 `cli/build_candidates.py`。 |
| `evaluate_dvgc.py` | 重写为 `cli/evaluate.py`。 |
| `fit_viability.py` | 重写为 `cli/fit_viability.py`。 |
| `outer_loop_dvgc.py` | 删除固定周期外循环；由分阶段脚本和 `cli/relabel_plan.py` 取代。 |
| `visualize_dvgc_rollout.py` | 删除。可视化不是认证证据，后续应从统一 rollout trace 生成。 |
| `probe_dvgc_policy.py`、`probe_dvgc_contacts.py` | 删除。必要检查并入环境测试、认证输出和独立 audit。 |
| `audit_takeoff_clean_signals.py`、`audit_takeoff_guideline.py` | 删除。旧 guideline 单位/状态推进存在问题；参考分析统一由 `dvgc/reference.py` 完成。 |
| `check_warp_imu_backend.py` | 删除独立检查脚本；统一配置恢复为原模型所需的 MJX-Warp + IMU/event contact 路径。 |
| `clean_2m_video.py` | 删除，与 DVGC 方法无关。 |
| `smoke_test_dvgc.py` | 由 `tests/` 取代。 |
| `run_landing_v2.sh` | 删除，由 `scripts/run_backward_bootstrap.sh` 取代。 |
| `AGENTS.md`、`CLAUDE.md` 等过程说明 | 未进入交付工程；方法和运行说明统一在 `README.md` 与 `PROJECT_SUMMARY.md`。 |

旧实验结果如果要作为论文负消融，应放到工程外的只读归档目录，不应继续参与 import、配置或流水线。
