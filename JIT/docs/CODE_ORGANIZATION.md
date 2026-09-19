# JIT 当前代码组织

更新：2026-09-12。维护既有能力模块，禁止按日期/seed/checkpoint复制实现。目录迁移须先检查import、CLI、配置和工件来源依赖，不因文件位于flat目录就删除。

| 能力 | 当前模块（相对 JIT/src/jit_dvgc） | 职责与边界 |
| --- | --- | --- |
| 固定物理/观测/动作 | `constants.py`, `observation.py`, `action_mapping.py`, `unified_env.py` | 76D/106D、四动作、phase任务奖励；与XML/config绑定 |
| 历史多proposer生产 | `envelope_campaign.py`, `campaign_bank.py`, `dense_tube.py`, `dense_tube_runtime.py` | all_proposers_v1已完成，不从旧命令重启 |
| 真实采集与动作来源 | `acquisition/causal_jump.py` | 固定起点、完整tape、真实prefix/context |
| 候选与物理几何 | `analysis/capability_tube.py` | root/full物理单元、phase、绘图投影；不是连续域认证 |
| 残差网络 | `exploration_network.py` | frozen基础π、106D残差Actor/Critic、tanh Gaussian及动作合成 |
| 探索PPO及候选选择 | `exploration_training.py` | active-step loss、因果候选tick、baseline、初始/最终诊断 |
| 当前π新颖性 | `exploration_reward.py` | arrival/witnessed两个版本、每cell共享1信用、per-policy ledger |
| 准确续接与未知标签 | `exploration_continuation.py`, `unified_continuation_labels.py` | 完整context恢复、fresh_continuation时钟、冲突unknown |
| 候选池及延迟反馈 | `exploration_pool.py`, `exploration_reevaluation.py` | pending/witnessed、追加评价、不回放旧PPO |
| 有限探索/学习外循环 | `exploration_loop.py`, `exploration_loop_declaration.py`, `exploration_loop_support.py` | 声明、成本、两臂、pending支持与下一sourceπ |
| 后继π训练 | `iterative_probe_training.py`, `training/` | 固定起点/快照混合、phase/group、warm Actor+normalizer和fresh critic/optimizer |
| checkpoint诊断 | `checkpoint_comparison.py`, `analysis/checkpoint_discovery.py` | 固定TRAINpanel与锁定探索安排、离线共同预算视图 |
| 批量接续 | `continuation/device_rollout.py`, `unified_continuation_shards.py`, `policy_family_landing.py` | serial/device/vectorized、掩码、容量与计费；不追溯改变旧协议 |
| 结果交付 | `result_bundle.py`, `result_publishing.py` | 原始工件索引、精简报告、来源身份 |
| 旧监督探索 | `residual_exploration.py`及对应export/fit入口 | 保留可读性与历史结果；不是当前残差PPO训练器 |

`JIT/cli/`保持薄入口；`JIT/configs/`是声明而不是执行证据；`JIT/tests/`保留协议行为测试；`JIT/runs/`保存不可变运行产物，默认不Git。论文作者侧脚本、紧凑来源表和三格式图在 `JIT/docs/paper/`，不侵入训练模块。

旧“残差只实现监督warm start”“pending禁止进入所有训练支持”“自动checkpoint仍完全未实现”等描述已过时。当前仍未实现：critic质量罚项、扩散探索、跨policy通用条件explorer、收益自适应proposer分配、自动无限延长训练。
