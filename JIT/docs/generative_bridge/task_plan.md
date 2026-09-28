# v1.1 implementation plan

Scope: user-provided merged v1.1 and reward supplement, base c98701a. No training launch.

- [x] Read update instructions, complete report and same-version YAML, reward supplement.
- [x] Inspect main checkout and registered worktrees: no existing generative_bridge package.
- [x] Implement/test strict contracts, nullable outcomes, pending invariance, trace admission and mixture.
- [x] Implement/test optional joint PPO/demo/keep adapter using existing restoration and numeric guard.
- [x] Implement/test H16 Flax DDPM/DDIM, proposal ranking and bounded full-state checkpoints.
- [x] Add opt-in preobs recording to existing evaluation; retain old defaults.
- [x] Implement/test resumable stage journal, cost reservations, prepare and equal-budget comparisons.
- [x] Audit explicitly named source artifacts without selecting production inputs; export evidence.
- [x] Run CPU regression/restore tests and review diff; report GPU and research not_run.
- [x] Initial components committed/pushed as715151f; continuation commits tracked below.

Constraints: original pending selection and reset rules; Actor 76D/256x3, action mapping,
H16, fixed generator structure, task reward and physics unchanged. Empty demo disables
sampling entirely. Unknown is never normal failure. TRAIN adopted Actor-only traces
remain distinct from teacher evidence. New experiments default execute=false.

Testing: pytest JIT/tests/test_generative_bridge_*.py and existing discovery reward,
retention, continuation and PPO numeric suites with JAX_PLATFORMS=cpu.
Production sources and all execution budgets remain unresolved pending explicit binding.

Production follow-up (not yet delivered): source-bound full teacher search/replay orchestration, initial G pretraining schedule, real adoption panels, GPU physics validation. CLI remains preparation-only. See REPORT.md for exact boundaries.

## 2026-09-28 continuation: complete and start authorized bounded training

User now requests full implementation and training. Reuse existing isolated branch.
Global constraints: preserve original Actor/physics/reset/reward/H16; teacher cannot gate pending PPO; no synthetic production data; exact adopted TRAIN success only; immutable history; bounded costs and notification watcher required. Source/budget choice requested asynchronously while implementation proceeds.

### Task 1: Generator pretraining
Implement a production pretraining stage reusing diffusion.py state/optimizer/checkpoint machinery, fixed source normalizer, real corpus and fixed dev fixtures. Configured defaults: AdamW batch256, at most20000 updates, LR1e-4, warmup1000, cosine decay to1e-5, weight decay1e-4, grad clip1, EMA.999, validation every1000. Preserve incremental1e-5 and full-state restore. Add deterministic scalar CPU behavioral tests and explicit costs/status. No GPU launch by subagent. Modify diffusion.py and its dedicated tests only; report interface for parent integration.

### Task 2: Production binding and teacher/student stages
Bind locked source/support/roots/ancestor partitions, production evaluation and32candidate teacher search/replay; prepare G bootstrap from true Actor traces; keep all original pending for PPO; attach demo/keep and acceptance callbacks; persist corpus and selected models. Wire CLI and bounded costs/watchers.

### Task 3: Validation and launch
Run CPU regression and bounded GPU semantic smoke, review all changes, predeclare selected source/budgets, launch permitted stage chain and verify real progress/notifications. Commit/push and report evidence, separating training running from complete.

Continuation status: pretraining8a5e43e and charged-budget fix23dc857 committed; production binding+orchestration implemented,109CPUregressions passed, source binding/retention read-only preflight passed. User delegated source/budget choice: source0093,698400physics,22000Gupdates,128000PPO,12h. GPU semantic smoke then training to run under existing resource gate/watchers.
