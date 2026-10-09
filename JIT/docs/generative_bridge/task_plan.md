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

## v1.2 source replacement and cumulative learning — user authorized 2026-09-28

Old fixed-Actor series stopped by user request; preserve all prior evidence. New exact source is Phase U transition_4988928. Implement the supplied v1.2 design incrementally, retaining legacy defaults.

- [x] Stop old supervisor; preserve stop request and verified termination records.
- [x] Read complete v1.2 instructions and same-version YAML; preserve original input ZIP contents.
- [ ] Exact source binding, unchanged Actor/normalizer, CPU action parity, full-task nominal GPU gate.
- [x] Full-start-only three-step pulse and independent logical-episode sampling.
- [x] Cumulative student demos, independent retention reference, actor-only warmup and learning probes.
- [x] Finite stage runner: fresh source data, same-layout teacher search/replay, three student arms.
- [ ] Regression tests and bounded physical verification, immutable implementation commit.
- [ ] Launch permitted new stage with verified progress, TensorBoard and notifications.

A0 reserves 1,600 physical steps. A1 reserves 1,820,000; A2 reserves 2,600,000 including PPO/development/checkpoint selection; total 4,421,600 with 24h wall cap, no automatic retry allowance. G<=22,000 and BC<=2,000 updates. Stage B remains disabled until adoption/absorption/control gates hold. The three-arm stage is a mechanism comparison, not an equal-total-cost superiority claim. Failure of the required nominal gate blocks large teacher/student training.

A1 includes a separately drawn 64-episode generator development set, three acquisition steps outside the inherited 400-step suffix budget, fresh two-phase nominal reset support (<=80,400), teacher source rechecks and smoke. These are reserved explicitly before A1 begins.

Validation snapshot: 170 CPU tests passed; source action parity64/64 exact. Separate A/B/C config/trainer integration checks passed. GPU nominal queued at a0_nominal (no physical execution yet); remaining validation is not claimed complete. Source_phase bootstrapG stays cold; StageB learning-explorer stage remains disabled.

## 2026-10-09 finite continuation repair and training visibility

User authorizes continuing the original 200-round task, preserving every jump
model, and joining TensorBoard across rounds. Preserve failed original series
and existing checkpoints. R3-R27 inventory is derived from saved receipts, not
new physical evaluation.

1. Contain CPython finalization failure only after a last-valid incremental G
   worker normally returns, full state/lineage/cost validation passes, and files
   are fsynced. Training exceptions remain failures; the native library root
   cause remains unknown.
2. Audit the already executed R27 stages and publish one complete same-round
   bundle in a new metadata-only recovery container. Public continuation must
   reject it until all validations finish. Preserve old failed statuses and all
   charged work; do not rerun the 128000 student transitions or 2000 G updates.
3. Add grouped TensorBoard runs with offsets bound to learner initialization
   receipts. Preserve original logs, reject overlapping scalar coordinates, and
   carry future rounds onto the same cumulative learner-step axis.
4. Run focused CPU regressions, inspect real saved states, and perform budget
   dry-run. Continue R28-R204 (177 rounds) only after publication validation, with
   the original seven-day clock and resource gate. No additional round budget.
5. Check live HTTP scalars, desktop watcher, model index, Git delivery, and
   shared research ledger. Software checks and state inheritance are not proof
   of improved recovery or jumping capability.
