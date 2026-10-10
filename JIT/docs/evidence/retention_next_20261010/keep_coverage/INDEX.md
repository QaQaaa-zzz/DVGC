# B_keep_coverage 独立试点：实现、审计与准备

审阅基准 `7b467877`，当时实际HEAD匹配且工作树干净，未回退。实现提交 `47b4aad`，配置身份修正后的执行代码 `0ebfd3ce64a28d2fd76068bc55afce9c2ec7ee42`。本轮只准备，不执行。有效目录 `/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_keep_coverage_003`，当前 **prepared、0 charged、0真实学生BC更新**。早期001/002是零成本准备快照，保留不覆盖，003为本次交付入口。

新增 `run_retention_b.py prepare-keep-coverage --proposal --parent --output --repository`，新schema `jit_retention_B_keep_coverage_v1`。没有修改旧BASE、旧准备授权布尔或warmup优化器；旧prepare/run行为及旧keep路由兼容。新schema的run和直接worker都在GPU导入前检查新阶段授权与原时钟，缺任一项拒绝；按目录排他claim防并发重复执行，无自动重试/恢复。

源B必须已完成2000更新，历史费用从原costs复算为388036。π0/R5 checkpoint完整payload、身份文件、bank/frozen/config、物理XML与reference、新旧数据及执行Python代码均锁定；历史源码按旧提交核对，和当前执行源码锁分开。旧formal输入SHA是规范JSON身份，先验证它，再锁当前序列化字节；不将JSON缩进差异误判成奖励/物理变更。审计完全基于文件，不导入JAX/MuJoCo/Warp/Torch、不启动子进程或环境。

复用已有80回合数组字节：seed1010269101、16名义数值重复＋64随机TRAIN，400步上限、固定π0、全零请求。新角色manifest有完整80个TRAIN祖先、来源初态bank、seed、lane、名义同条件标志，与7个SOLVER_DEV祖先隔离；新独立TRAIN namespace只来自新回合，TEST保持未打开。不使用B_06/C_10或其他DEV失败状态/动作生成数据。名义组是1个物理条件的16次数值重复，随机keep采集不改变E的初始化协议。

**运行接线已实现，真实采集仍PENDING。** 将来获授权后仅采一次，正常失败保留标签和收费；工程异常不填0；只准入正式成功且没有物理失败冲突的完整真实轨迹，未知/冲突单列，不补seed。保留全部旧keep，另写新不可变npz/receipt/merged lock；名义/随机各50%采样质量，组内祖先均衡、祖先内轨迹均衡、轨迹内真实帧均匀。保存全部80结果、入池/排除理由、旧/新阶段观测数和着陆后观测数、名义重复条件及采样权重。当前没有新keep成功数，不能把CPU夹具数据写成TRAIN物理数据。

真实B_worker通过 `worker_keep_inputs()` 加载新merged receipt；新schema在合并尚未完成或哈希漂移时拒绝，绝不回退旧keep。旧schema仍读原keep。运行时保存 `worker_inputs.json` 绑定实际数据路径/哈希、教案、seed和初始化。学生原始π0＋fresh Adam/RNG，冻结整个normalizer/critic，Actor-only Adam1e-5/demo1/keep1/batch256/clip1；525条G8教案文件、祖先/前缀尾部权重、BC base seed和warmup逐步split/fold_in规则均保持原样。无BC2000/micro-last初始化，无E/G或PPO更新，无动作/物理/奖励/观测/成功标准变更。

保存0/100/500/1000/2000全部完整节点。新阶段独立重测5节点×15根×17世界：8根TRAIN与全部7根SOLVER_DEV；4个非零节点64题四格，原π0/R5完整任务基线凭原身份复用。7根中4个G参考仍NA，学生仍评价全部7根。选模仍全部7根主分数、最早平局、原B/D>5pp警戒及每节点最多一次固定源/候选确认；不调整阈值以求通过。选中候选8根固定两次复核。没有额外R5或四组合重测，它们保留原B证据，不超预算。

新report分支分别输出各节点TRAIN吸收、SOLVER_DEV迁移、四格N01/N10/净差、源/学生重复翻转、逐根转化矩阵与物理阶段，区分learner_last、stage_candidate、冻结R5和published_policy。未运行报告写PENDING/NOT_ESTABLISHED。已用DEV仍是开发数据，不能称新holdout或TEST。旧损失未降低或已吸收教案稳定丢失时保留π0/R5、结束本阶段，下一步仅考虑TRAIN学生自身访问状态诊断；无任何自动PPO或200轮入口。

|项目|新增charged上限|
|---|---:|
|80回合keep|32000|
|5节点×15根×17×400|510000|
|4节点×64完整DEV×400|102400|
|最多4次固定源/候选警戒复核|204800|
|选中候选8根两次复核|108800|
|合计|958000|

此前388036，全部执行后最多1346036／2000000；额外2000监督更新另计。原12小时起点 `1791622384.6936376`，截止 **2026-10-11T04:53:04.693638+08:00**。审计时约剩5小时25分，启动前必须重核，不重置。当前缺新阶段明确执行授权，故停在prepared。若授权到达时原时钟已过，仍拒绝，需另行明确有限时间边界。没有启动新TensorBoard/桌面watcher；只有将来授权run才启动独立6028与watcher并验证实际DEV奖励HTTP，不能把预留URL冒充已加载的训练标量。

定向回归 **60 passed /13.61s**：包括原入口、冻结初始化/完整状态、组/祖先/轨迹权重、DEV共祖先拒绝、失败排除与非有限拒绝、新worker实际使用merged loader、缺授权/过期/漂移拒绝、并发claim、阶段预算预留拒绝、prepare报告不编造效果，以及静态audit的禁止GPU模块与子进程检查。真实80回合输入已完成prepare/audit/dry-run/report，并再次对实际plan检查禁止导入/子进程、prepared零收费和缺数据拒绝。CPU通过不代表GPU采集、额外2000BC或可靠物理恢复已验证。

[actual-layout预算](budget_dry_run.json)、[实际audit](audit.json)、[零GPU/零仿真检查](static_verification.json)、[80回合祖先角色](TRAIN_keep_manifest.json)、[输入与源身份](input_summary.json)、[执行代码锁](code_lock.json)、[准备报告](prepared_summary.json)、[测试原文](tests.txt)。完整plan/input_lock与初态/request数组在服务器有效目录。

仅只读命令：

```bash
cd /home/qy/DVGC/runs/worktrees/jit-retention-first
export PYTHONPATH="$PWD/JIT/src" JAX_PLATFORMS=cpu
PY=/home/qy/mujoco_playground/.venv/bin/python
PLAN=/home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_keep_coverage_003/plan.json
"$PY" JIT/cli/run_retention_b.py audit --plan "$PLAN"
"$PY" JIT/cli/run_retention_b.py dry-run --plan "$PLAN"
```

首次准备命令已实际运行（不可复用同一output覆盖）：

```bash
"$PY" JIT/cli/run_retention_b.py prepare-keep-coverage \
  --proposal /home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_keep_coverage_preparation_001/proposal.json \
  --parent /home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_002/plan.json \
  --output /home/qy/DVGC/JIT/runs/experiments/retention_next_20261010/B_keep_coverage_003 \
  --repository /home/qy/DVGC/runs/worktrees/jit-retention-first
```

将来**只有新阶段获明确执行授权后**才能 `run --plan "$PLAN" --execute --authorization "$AUTH"`。授权凭证需schema `jit_keep_coverage_execution_authorization_v1`，绑定当前plan SHA，scope `collect80_and_BC2000_keep_coverage_only`、2000更新、958000上限、原起点、`source=explicit_user_instruction`及真实授权指令引用；本轮未生成它。即使有该凭证，也不能绕过原时钟和输入/代码锁。
