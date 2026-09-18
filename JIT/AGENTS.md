# JIT implementation guidance — 2026-09-12

Root [AGENTS](../AGENTS.md) governs. Read the [latest complete handoff](docs/CODEX_HANDOFF_20260911.md) and [roadmap](docs/JIT_TRAINING_ROADMAP.md) before editing.

User requirement (2026-09-14): every future JIT experiment launch must include a desktop error and normal-completion watcher. Notify once when all declared top-level execution/lineage statuses are completed; do not notify for individual child-stage completion. Start `PYTHONPATH=src python cli/watch_run_errors.py --active-run <experiment>/ACTIVE_RUN.json --state-dir <experiment>/notifications` in a detached process with desktop DBus environment preserved, and verify its heartbeat. The active manifest names current `execution` and `lineage` status files; update it on restart. The watcher follows that pointer, ignores previous failed attempts, deduplicates delivered errors, and retries failed notification delivery. It requires a live desktop notification service; it does not automatically survive a machine reboot or detect a killed process whose status was never updated. Verify notification delivery and watcher health instead of claiming popup visibility. Keep its log and notification_status.json with the experiment.

Production all_proposers_v1 is COMPLETE: pi_5/pi_6 trained and frozen, 9,296 historical campaign root cells, 256,000 new PPO transitions. First-success stopping, checkpoint comparisons, vectorized continuation engineering, frozen-base residual PPO, per-policy arrival rewards and pending/delayed learning now exist. The latest two-round pilot completed two128k policy trainings with zero delayed no-witness-to-witness gains. The old supervised residual and full-action experiments are superseded research stages, not current architecture. Diffusion, critic quality penalties, adaptive proposer allocation and automatic training extension remain unimplemented. See CURRENT_STATUS for current evidence.

Implementation priorities:

1. Explain current novelty saturation and candidate caps from saved data, including active/padding and effective residuals; no new simulation needed for this analysis.
2. Reuse phase geometry and audit root-grid sensitivity separately from5cm candidate/plot spacing. Preserve historical landing/failure conflicts and accepted replay limitations.
3. Preserve current first-success/unknown semantics and full-matrix comparison mode; never turn missing/error/conflict into negative or compare different context identities.
4. Use bounded128k–256k working budgets instead of another open-ended length sweep. A4-state panel is diagnostic, not proof that pending candidates were learned.
5. Predeclare fair same-mode residual/random and independent pending-enabled/witnessed-only successor comparisons before claiming an exploration or training advantage. Diffusion is optional later work.

Core files: envelope_campaign.py, campaign_bank.py, iterative_probe_training.py, dense_tube.py/dense_tube_runtime.py, acquisition/causal_jump.py, continuation shard modules, analysis/capability_tube.py, result_bundle.py/result_publishing.py. Extend these capabilities instead of making another iteration-specific duplicate pipeline.

Historical production reset recipe is20% complete fixed x2.5 start /80% witnessed snapshots. Current delayed recipe allocates25% of a phase's snapshot mass to pending when available, preserving phase/group balance. In the latest pilot pending training support is upstream only, giving about10% total reset probability. New pi uses Actor+normalizer initialization and fresh critic/optimizer. Keep support separate from verified envelope admission. No whole-Tube retention gate. Preserve complete history; legacy scans retain original identities/schema.

Respect the declared backend and budget for each stage.4096/8192/16384 vectorized capacity engineering is complete, but a38→39tick difference prevents assuming exact equivalence to locked serial experiments. Large capacity does not authorize competing jobs or replacing old scientific protocols. A CPU fixture pass is not a GPU rollout result.

Store each round's replot CSV/manifest, individual PNG/PDF/SVG, receipts and cost. Preserve observed gaps, missing evaluator outcomes and historical bank versions. Full images stay server-side; GitHub compact report absence of images is not a plotting failure.

## 离地点对齐图（2026-09-18，用户要求保存并复用）

- 使用 `PYTHONPATH=src /home/qy/mujoco_playground/.venv/bin/python cli/plot_liftoff_aligned.py --lineage <当前lineage目录> --output <新派生目录>`，在 JIT 目录执行；可沿 recovery/resume 祖先读取已完成轮次，零新增仿真，不覆盖原始记录。
- 同时输出原世界坐标与 `x_relative=x_world-x_liftoff` 对齐视图，只平移水平位置，保留真实根部高度；离地点高度不强行拉齐。前缀和接续保持分段，不画虚构重放连接。
- 默认离地诊断：先观察至少一个车轮间隙≤0，再取双轮间隙连续3个控制帧>0.01m的首帧；`--clearance` 和 `--hold-frames` 可显式配置。该诊断用于画图，排除初始悬空间隙，不能冒充正式物理子步离地事件或修改成功判据。未检出者须列入排除记录。
- 默认每轮按成功候选索引均匀抽取最多24条，每候选取首个成功attempt；`--per-round 0` 才是所有成功候选。报告已完成轮次、成功候选总数、抽取数及可对齐数，不把抽样投影称完整包线。
- 保留局部与完整恢复PNG/PDF/SVG、原始/对齐坐标NPZ、离地点CSV、见证/段索引JSON、来源哈希及INDEX。对齐坐标只比较形状，不参与世界坐标物理单元去重或可达性计数；不同prefix/suffix策略见证不等于单策略全程能力。

Do not open final TEST. TRAIN adaptation and development ACCEPTANCE are not final held-out performance. Data rows share ancestors and are not independent repetitions. Follow root authority for authorization, git safety and report retrieval.
