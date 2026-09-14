# JUMP: finite preparation and jump timing

前向行驶约束下，面向窄障碍的准备动作与起跳触发联合决策。

当前实现是**第一阶段资格测试**：复用冻结 JIT Actor、原机器人/执行器/观测，通过独立的 host MuJoCo 运行器检查真实接地接近、外部触发、完整越障与落地后恢复。CPU 新协议的结果不能代表旧 MJX 回放等价。上层结果预测器和大规模训练尚未实现。

首轮尝试已完成：12个初始设置＋2个近起点追加，实际270控制步；均未通过接地准备资格，未发出跳跃触发。14组完整轨迹/PNG/PDF保留。下一步先核对后端/观测采样兼容性，再决定准备控制适配。详见当前验证。

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
