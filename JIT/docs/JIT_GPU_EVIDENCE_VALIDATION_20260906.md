# JIT 小批 GPU 证据验收：运行与回传

本入口验证真实前缀、快照恢复、时间计数和串行/分片的一致性。它不运行 PPO、不补写历史标签、不选策、不把样本加入训练 Tube。默认只检查既有 Round1 π0；先运行这一轮，再根据结果决定修复或扩大探针对照。

## 直接运行

在保存真实 checkpoint、Tube 和配置的生产服务器执行。默认 GPU 为 0，可用 `--gpu 1` 指定另一张空闲 GPU。

```bash
cd /home/qy/DVGC &&
git switch agent/two-phase-soft-tube &&
git pull --ff-only origin agent/two-phase-soft-tube &&
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/validate_jump_evidence.py --gpu 0
```

脚本会在 `JIT/runs/evidence_validation/` 下新建带时间和 PID 的目录。最后打印：

```text
Return this file: .../results_to_send.zip
```

**把这个 ZIP 原样发回来即可。** 通过、出现差异、缺失工件或子进程失败都会打包；不必手动复制长终端输出。现有输出目录不能覆盖，原始 checkpoint 和历史 run 文件不修改。不要因为看到红色结果就重新训练或放宽阈值。

默认从本机 `JIT/runs/frozen_unified/` 搜索唯一的、formal config 名含 `round1` 的 `pi_0`。若找到多个或没有找到，会停在零交互 preflight，并在结果包中给出原因与候选路径。这时也可以先回传 ZIP；若已知正确文件，可显式运行：

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/validate_jump_evidence.py --gpu 0 \
  --frozen-policy /absolute/path/to/frozen_unified_policy.json
```

必须使用完整冻结记录及其真实 checkpoint/config/Tube 资产；不能把其他 checkpoint 改名为 π0。路径缺失不等于策略失败。无需在服务器安装新的测试环境，沿用现有生产解释器。

## 默认工作量和锁定内容

- 前缀种子 `9400001`，复用已有成功种子作工程检查，不算新的独立研究样本。
- 从既有 `_reset_jump_start_unified` 的完整 x=2.5 地面状态开始；保存 qpos/qvel、FIFO、动作、事件、计数、随机键、XML 和地面几何观测。
- Actor 使用冻结 checkpoint 的 deterministic inference。
- 一条真实前缀最多运行一个冻结 horizon；first valid landing 或终止后立即结束。
- 从真实 reached frames 选 upstream 首/中/末及 descending downstream 首/中/末，最多 6 个。保留 Apex 前后邻近帧，排除 post-landing、终止和仍上升的 downstream 帧。没有足够阶段覆盖时不会伪造状态补齐。
- 每次只运行一个子进程；默认每个 evaluator shard 最多 2 个候选，关闭子进程 JAX 显存预分配。首次 JIT 编译可能较慢；协调器每 30 秒打印当前阶段及日志路径。
- `label_seed=9521602`，各对照使用相同 candidate global index、seed 和冻结 horizon；不修改旧 wide/expanded 锁定实验。
- 默认 `atol=1e-6, rtol=1e-5`，离散事件/计数要求完全相同。这是预声明的工程数值容差，不是物理可行域的分辨率。
- 默认冻结 horizon=400、单策略、6 个状态时，上限 **17,200 次 env.step 调用**。默认总限额 20,000；各阶段也有执行上限。实际 landing/终止会提前结束，真实计数写入报告。编译耗时、reset 和模型 forward 不是 env.step 次数；不代表训练成本曲线已经闭合。

默认完整流程包含 12 个子进程：1 次前缀采集、6 次单状态重放、1 次串行标签、3 次分片标签、1 次合并。若阶段样本不足，进程数量相应减少。父进程不构建 GPU 环境。

## 四种接续，分别回答什么

每个候选的 reference 都通过保存的动作从同一个真实地面起点重新走到该帧，不用 RSI 伪造前缀。先比较其完整存储上下文与原 captured snapshot，逐项记录数值差异和精确哈希是否相同。

| 组别 | 初始化 | 用途 |
| --- | --- | --- |
| `raw` | 真实前缀结束时的 live simulator state | 未经快照重建的接续参考 |
| `preserved` | 保存再加载 snapshot，恢复原 episode/phase 计数 | 检查 snapshot 重建是否改变行为 |
| `raw_fresh` | live state 上直接应用现有 fresh-continuation 计数转换 | 单独隔离计数/行政状态转换的影响 |
| `restored_fresh` | snapshot 恢复后应用完全相同的转换 | 对照现有生产续接语义 |

每组继续到 first valid landing、终止或冻结 horizon。全部共享当前候选的 evaluator 和动作随机键。对比 qpos/qvel、ctrl、Actor 观测、FIFO、last action、随机键、相位/事件、终点及交互步数。保留计数的两组还逐帧比较 episode/phase/up-event 计数、start phase、累计回报和相位切换标记。

`counter_effect` 比较 raw 和 raw_fresh 的物理/Actor/事件/终点行为；已声明改变的计数、start phase、累计回报与相位切换标记不拿来制造“必然不相等”的数值失败，但其初值全部单独列出。这样才能发现计数改变是否实际延长存活时间、改变超时或落地结果。

另外记录 simulator `time`、`qacc_warmstart`、`act`、外力等可读取字段的初始差异。当前 snapshot 没有保存所有求解器内部状态；这些差异不能被省略成“完整状态已经一致”。本验收最多支持所测样本、策略、容差下的行为结论，不证明任意状态的重放等价。

## 串行与分片怎样比

使用同一个新建的小 catalog，调用现有生产 `label_unified_continuations` 和 `label_unified_continuation_shard`，通过已有严格 merger 还原全局顺序。对比候选/快照/Actor/payload/终点身份、标签、outcome、接触与终止事件、horizon/seed、每条交互数及 Actor 观测。

串行与分片的 execution metadata 和逻辑 protocol hash 本来可以不同；其完整协议各自验签后，逐个核对共享语义字段。缺行、重复、错序、改 seed/horizon/身份、labels 文件哈希漂移均不能通过。

低层标签接口仍使用历史 `split=train` 标记。本次 catalog 另有 `logical_role=engineering_validation` 和 `tube_admission_authorized=false`，这些数据只用于工程验收，不进入自适应 TRAIN、预测器拟合或 Tube。新 bank 采集协议检查也会拒绝把本次 catalog 当作正式 bank 采集。

## 看哪些结果

| 文件/字段 | 含义 |
| --- | --- |
| `summary.json` | 总状态、各验收 gate、各阶段退出码与成本 |
| `plan.json` | 事前锁定的策略、源代码身份、起点种子、容差、预算和样本规则 |
| `capture/start_state.json` | 实际完整起点和轮胎对地几何值 |
| `capture/prefix_frames.json`、`prefix_actions.json` | 真正走过的前缀，不是插值状态 |
| `evaluator_*/replay_*/report.json` | 首个差异帧、逐字段最大误差、四组终点、计数和未序列化 simulator 字段差异 |
| `evaluator_*/replay_*/trajectories.json` | 逐帧数值轨迹，用于后续定位 |
| `evaluator_*/serial/`、`shard_*/`、`merge/` | 原生产标签器输出及执行日志 |
| `failure.json`、各阶段 `process.log` | 缺文件、依赖、显存、运行错误等工程原因 |
| `results_to_send.zip` | 需要回传的包；只收集小 JSON/日志/说明，不包含 checkpoint、pickle 快照或视频 |

- `passed_on_sampled_states`：所有默认 gates 在这一小批状态上通过，**不是允许直接大训练**。
- `diagnostic_mismatch`：流程完成，但起点轨迹、阶段覆盖、恢复、计数行为或分片一致性存在问题。
- `engineering_error`：缺资产、依赖/显存/执行失败等；不生成物理负标签。未记录到完整计数的阶段会标记成本不完整。
- 退出码 0 对应小批全部通过；2 对应需回传分析的差异或工程问题。二者均有结果包。不要删除失败记录来重试计费。

下一步由结果决定：先定位恢复/时钟差异；通过后再补旧标签，核对更多失败/边界状态及其他冻结 evaluator，再开展已有策略的同预算探索对照。本轮不会自动改写 bank 中的 `snapshot_replay_equivalence_verified=false`，也不会自动启动 π4。

## 本地开发验证边界

开发环境执行 CPU 单测、受控数据比较、模拟子进程编排以及失败打包检查。生产 GPU、实际 checkpoint 和真实 rollout 的结论等待本次回传。测试记录在 [CURRENT_STATUS](CURRENT_STATUS.md) 中更新。
