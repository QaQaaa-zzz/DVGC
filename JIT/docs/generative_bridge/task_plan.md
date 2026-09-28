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
- [ ] Commit/push isolated implementation branch and verify resulting status.

Constraints: original pending selection and reset rules; Actor 76D/256x3, action mapping,
H16, fixed generator structure, task reward and physics unchanged. Empty demo disables
sampling entirely. Unknown is never normal failure. TRAIN adopted Actor-only traces
remain distinct from teacher evidence. New experiments default execute=false.

Testing: pytest JIT/tests/test_generative_bridge_*.py and existing discovery reward,
retention, continuation and PPO numeric suites with JAX_PLATFORMS=cpu.
Production sources and all execution budgets remain unresolved pending explicit binding.

Production follow-up (not yet delivered): source-bound full teacher search/replay orchestration, initial G pretraining schedule, real adoption panels, GPU physics validation. CLI remains preparation-only. See REPORT.md for exact boundaries.
