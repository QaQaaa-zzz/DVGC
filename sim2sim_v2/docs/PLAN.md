# Sim2sim V2 implementation plan
Approved design: user TASK.md, executed in isolated worktree. Historical evidence immutable.
1. P0: resolved config single source, frozen dependency paths, legacy equality and coefficient sensitivity; contact recorder with per-substep physics events.
2. P1: evaluate exact9904128 in MJX first with strict landing diagnostic; PhysX deterministic/stochastic, lanes/origins and reset factors, export parity, fixed action replay. Preserve first failure.
3. E1 prerequisite: offline normalizer/optimizer KL decomposition, rollback tests; only after P0/P1 gates train20 blocks (491520 learner transitions), evaluate10/20, conditional extend40.
4. E2: independent landing events/reward, offline ordering and finite tests then bounded training if E1 gates pass.
5. E3: true prefix suffix curriculum and takeoff retention; only if E2 ready. Physics/action/input changes separate unrun axes.
6. Report, actual budgets, local commit and locked shared ledger update; no push.
Diagnostics first declaration: <=64 scored episodes, <=400 control steps each, <=102400 scored physics steps for single lanes; full384 lane probes count all actual lanes separately and predeclare before launch. No PPO allowed until explicit diagnostic gates recorded.

2026-10-09 completion: P0 verified; P1 diagnostics executed, gate HOLD (layout/reset contact divergence unresolved). 26 focused tests pass. E1 prepared/offline tested, training unrun; E2/E3 unrun. See ../REPORT.md.

Latest user priority: same frozen model lands on both engines, no required forward speed. Verified transition_01966080 on one paired natural start; details runs/no_speed_dual_landing/REPORT.md. E1/E2/E3 remain unrun; no further training needed for this narrow engineering goal. Solver-history n1 complete, n2/n384 unrun.

2026-10-13 user-authorized dual stress panel: frozen dual-landing Actor01966080, exact existing hard1000 initial bank + round0099 explorer/atlas. 1000 paired cases per engine, onset0/5/10/15 x250, amplitude .25 x3 ticks. No retraining. Nominal source physics, calibrated target physics; no minimum forward speed. Per-engine maximum400000 scored control/1600000 physics; <=160 extra native-contact certification controls. Preserve shared sample seeds, actual adaptive pulse differences, traces and errors. Run small engineering compatibility check before full panel; count its cost separately. E1-E3 not part of this task. Existing FrozenNeighborhood and rsl_pulse infer reused; native batching and Warp shared capacity follow installed DATA_NON_VMAP. Source config/environment remains frozen.
