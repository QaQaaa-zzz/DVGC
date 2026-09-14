# JUMP: finite preparation and jump timing

前向行驶约束下，面向窄障碍的准备动作与起跳触发联合决策。

当前实现是**第一阶段资格测试**：复用冻结 JIT Actor、原机器人/执行器/观测，通过独立的 host MuJoCo 运行器检查真实接地接近、外部触发、完整越障与落地后恢复。CPU 新协议的结果不能代表旧 MJX 回放等价。上层结果预测器和大规模训练尚未实现。

已完成三批资格/来源诊断，累计400控制步。最新找到原始上升专家pi_up_star：从x1.5在signal0下稳定前进1.84m/1s，当前pi0则偏航失败。上升专家仍未通过严格连续双轮接触资格；接触切换与运动稳定性需分别诊断。尚无实际触发或完整新任务成功，未启动新PPO。18组完整轨迹/PNG/PDF保留，详见当前验证。

继续追溯后找到更早的`transition_4988928`：旧自然起点轨迹包含跳上平台、驶下后沿、再落回地面前进的片段。也更正此前判断：当前pi_up_star并非只有上升证据，它同样有平台上落地行驶片段；最终失败不能否定此前落地。16条旧轨迹已完整出图，侧翻、航向与角速度问题保留，这些片段尚不满足新任务恢复标准。[详细核对](docs/VALIDATION.md)与[全部图件](runs/source_search/20260914/INDEX.md)可直接查看。本次新增仿真/训练均0步。

- [当前计划与进度](PROJECT.md)
- [验证范围](docs/VALIDATION.md)
- [方法与结果](docs/METHODS_AND_RESULTS.md)
- [2 亿控制步分阶段预算](configs/campaign.json)
- [首批 12 个情况、最多 4,800 控制步的声明](configs/qualification.json)

在此 Git 工作区根目录运行（复用 `/home/qy/mujoco_playground/.venv/bin/python`，无须安装新环境）：

```bash
PYTHONPATH=JUMP/src:JIT/src JAX_PLATFORMS=cpu /home/qy/mujoco_playground/.venv/bin/python JUMP/cli/run_qualification.py \
  --config JUMP/configs/qualification.json \
  --campaign JUMP/configs/campaign.json \
  --source-root /home/qy/DVGC \
  --output JUMP/runs/qualification/<new-run-name>
```

每次使用新的输出目录；runner 在第一步仿真前冻结声明、校验 checkpoint SHA 并预留预算。启动后立即开启 watcher，保留桌面 DBus 环境并验证心跳：

```bash
PYTHONPATH=JUMP/src:JIT/src /home/qy/mujoco_playground/.venv/bin/python JUMP/cli/watch_run.py \
  --active-run JUMP/runs/qualification/<new-run-name>/ACTIVE_RUN.json \
  --state-dir JUMP/runs/qualification/<new-run-name>/notifications
```

入口包括 `status.json`、`results.json`、`analysis/INDEX.md`、每个情况的 JSONL 物理子步记录及 PNG/PDF。程序错误保留为 unknown，并停止后续情况；确定的任务失败不会被当作程序错误。预算账本位于 `runs/campaign_ledger.json`，异常进程尚未核算的预留仍占预算。

所有 checkpoint/轨迹等大工件留在被忽略的 runs；Git 只保存源代码、配置和可审查的当前摘要。原 DVGC/JIT 工作区及原始结果不修改。
