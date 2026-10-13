# FrontierBridge retention-constrained Actor-only implementation

User specification: `/home/qy/下载/FrontierBridge_agent_implementation_20261013.txt`.
The user explicitly excluded the 92.4% claim on 2026-10-13. It has no effect
on identity binding or selection. This is a separate stage, not a continuation
of D2, R73/R74, full200, or teacher-recovery100.

## Implementation plan and progress

- [x] Read current rules/state and verify the clean implementation worktree at
  `b6f951f8b93f580a23442fee147e754edc981e9e`.
- [x] Data layer: hash-bound real observation banks, parent action labels,
  ancestor/condition split checks, group-balanced weights, isolated pi0 identity.
- [x] Update layer: default-off hook in existing `warmup_actor`, one actual Adam
  proposal, six scaled candidates, atomic full-state rejection, attempt clock,
  checkpoint zero stop, full learner restore including rejected attempts.
- [x] Evaluation layer: separate parent/pi0 retention, stable repeated labels,
  independent absorption, required group gates and parent fallback.
- [x] New stage/CLI: identity locks, new authorization, no inherited D2 deadline,
  P0/P1/P2 common schedule and fixed data, separate cost ledger.
- [x] Integrate and review production adapters; run the complete listed CPU
  regression suite and zero-physics CLI dry-run, record exact counts.
- [x] Prepare verified implementation and the shared-state handoff. Exact commit
  and push outcome are recorded in the server evidence `git_delivery.json`.

## Contracts

`generative_bridge.retention_bank` exposes `build_retention_bank`,
`load_retention_bank`, `validate_retention_bank`, `audit_split_isolation`.
`generative_bridge.retention_guard` exposes `guard_metrics` and
`guarded_adam_step`. The update hook is
`update_guard={'bank': arrays, 'config': guard_config, 'identity': bank_identity,
'enabled': True}`; set enabled false for matched P1. Omitting the hook and new
attempt options preserves historical warmup.

The pure evaluation module exposes `evaluate_capability_groups`,
`select_checkpoint`, and independent existing pulse replay/collection adapters.
The orchestration has `prepare`, `audit`, `dry_run`, `run`, `worker`, `report`;
the thin entry is `JIT/cli/run_retention_consolidation.py`.
`retention_consolidation_runtime.py` dispatches the existing pulse backend;
`retention_consolidation_report.py` builds figures solely from saved receipts.

## Configuration and unresolved execution inputs

The checked-in config intentionally leaves parent, pi0, banks, calibration,
selection, evaluation and costs unset. Preparation/dry-run succeed as audits
and explicitly report `blocked`; they do not create an optimizer, call a
physical step, or search the teacher. No authorization file is generated.

Policy descriptors bind frozen wrapper plus checkpoint payload, config, XML,
Actor/normalizer/critic hashes, Actor schema/history, critic schema and action
transform. TRAIN_FIT and TRAIN_GUARD each cover old_nominal, old_random,
old_recovery and accepted_parent_recovery. Both are training information.
TEST arrays remain closed; DEV split metadata participates in leakage checks.
File hashes and canonical configuration hashes are distinct. Preflight uses the
repository loaders and checks the real pi0 serialization without a rollout.
A task_identity JSON binds physics, reward, termination, observation/history
and action transforms; its semantic hash binds the banks. Teacher manifests
require complete canonical source traces and verification receipts linking
root, context and trace bytes; BC frames must match those sources. Composite
teacher success cannot supply independent accepted_parent_recovery evidence.

Thresholds are normalized-action group/channel MSE and weighted absolute p99.
The quantile is the left inverse weighted empirical CDF, without interpolation.
Weights are normalized inside each group. The calibration receipt records
candidate thresholds, fixed parent and guard payload identities, TRAIN/DEV role
and costs; uncalibrated thresholds cannot run. A finite predeclared consecutive
rejection limit stops `blocked_by_guard`; thresholds are never auto-relaxed.

Selection config binds group-specific max_lost and fractional max_rate_drop,
mixed/unknown policy and min_expansion_gain, plus an explicit decision on
parent/pi0 gaps. Lost and gained remain separate. Candidate selection does
not publish/adopt a model. A parent/update0 fallback is no accepted improvement.
Verified absorption requires stable parent failure, stable teacher success and
independent student success; general gains remain separately counted.

## Cost and execution

P1/P2 use the same parent, seed/absolute attempt indexing, teacher weights,
fit bank, batches256/256, Adam1e-5/clip1, loss coefficients1/1 and checkpoint
schedule0/100/250/500/1000. 1000 is an attempt ceiling for each arm. Report
accepted/rejected counts and extra forward/backtracking costs separately.
Only one device is supported; no multi-device atomicity claim is made.

Evaluation panel cases are fixed independent complete episodes or complete
snapshot-context replays, each declaring case_id, condition_identity, group,
role, capacity, lane, horizon400 and three repeat seeds. Repeated executions
measure label variability; they are not independent training seeds. A teacher
composite is never counted as a student success. Existing witness eligibility
is preserved; numerical student evaluation is not formal envelope admission.
The current collect adapter supports complete deterministic episodes with zero
exploration residual. Nonzero external request-table admission is blocked in
this implementation; disturbed recovery uses complete snapshot replay. This
limitation does not remove cases from an already declared panel. Actual batch
capacity is bound before dispatch. Repeated collection uses the same condition
seed; snapshot time is bound to recorded source evidence. Renamed copies of
the same selected condition are not separate samples.

Reserve all capacities times horizon including padding and failures before
dispatch. BC attempts are not environment transitions. Record old acquisition,
reuse, new data, teacher queries, evaluations, active/padding and wall costs.
No new data/search/PPO is performed by this stage. Parent/pi0 evaluation
baselines are shared by P1/P2 only within the locked identical panel.

Launch requires an actual authorization record matching the plan hash, both
arm attempt budget, new physical budget and wall limit. The worker rechecks it.
TensorBoard binds loopback only; BC/reward is explicitly the negative
supervised loss, alongside loss and actual DEV results. The current arm's
scalar is verified over HTTP. Desktop watcher uses the scoped ACTIVE_RUN,
inherits DBus and must produce a live heartbeat. This does not prove popup
visibility. No automatic next stage or model publication is enabled.

## Verification and evidence

New tests are `test_retention_guard`, `test_retention_bank`,
`test_retention_consolidation`, `test_retention_consolidation_identity`,
`test_retention_consolidation_runtime`, `test_retention_consolidation_evaluation`.
Independent review regressions are in `test_retention_consolidation_review`.
Use `/home/qy/mujoco_playground/.venv/bin/python`, `PYTHONPATH=JIT/src`,
`JAX_PLATFORMS=cpu`. The attachment's twelve historical files are also required.
Test commands/counts and dry-run are recorded in the implementation evidence
directory under the server `JIT/runs/experiments`; raw training artifacts are
not committed. CPU toy optimizer tests are engineering checks, not GPU/MJX
closed-loop evidence. GPU microcheck and P0/P1/P2 physical pilot are not run
without the new explicit inputs, calibrated limits and budget.

## Verified implementation delivery (2026-10-13)

Final new/independent-review plus all twelve required legacy files:
**209 passed in 26.23 seconds; zero failed, zero skipped**. This includes a
read-only integration audit of the actual production pi0 checkpoint and the
real installed CPU JAX/Optax toy update/resume behavior. Python compilation,
CLI help and `git diff --check` also pass. No GPU/MJX rollout or training ran.
Separate synchronous sampling/proposal/diagnostic/guard durations are recorded
per attempt; compile cost belongs to the first measured call.

Evidence: `/home/qy/DVGC/JIT/runs/experiments/frontierbridge_implementation_20261013/`.
The final post-commit `dry_run/` contains parent_manifest, input_lock,
environment, plan, cost ledger and blocked audit. Preparation uses the
checked-in unresolved template: zero environment transitions, zero optimizer
updates and zero teacher searches. This blocked execution status is expected;
it does not imply a failed implementation. No budget or authorization is
inferred from the specification or old D2 allocation.

Changes from the suggested file list: orchestration/runtime, pure evaluation
and plotting are separated for independent verification; an extra identity
and independent review test module exercise installed artifact contracts.
The default historical BC path stays unchanged. No teacher/E/G/PPO/physics/
reward/network redesign, rollout expansion, model adoption or TEST opening.
Guard performance on real GPU banks and the P0/P1/P2 ability comparison remain
unverified; CPU fixture success is not evidence of knowledge absorption.

## Explicit training authorization and real startup adapter

The user subsequently requested training and explicitly selected frozen pi0 as
parent. The independent pilot is
`/home/qy/DVGC/JIT/runs/experiments/frontierbridge_pi0_pilot_20261013`.
This new declaration caps all calibration/micro/main physical work at
1,000,000 transitions and 12 hours; it inherits no old D2 balance. Calibration
uses a separate fresh Adam for20 updates plus one proposal for two micro guard
checks, never carrying its optimizer into P1/P2. Only0/20 are evaluated in
physics; full guard forward diagnostics also cover5/10. update20 must pass
every mandatory zero-old-loss gate before its measured group/channel MSE/p99
budgets can be frozen; failure blocks the main stage, with no auto-relaxation.

The real source adapter preserves525 teacher frames/eight TRAIN roots and
original weights. Banks contain4103 real frames from81 complete successful
Actor episodes, with FIT/GUARD reuse explicitly declared. Target GPU labels
and both inference graphs match exactly. Parent equals pi0, so the inherited
post-contact parent group does not imply new parent-relative capability.

Nominal fixed-start repeats collapse to one explicitly seen TRAIN_RETENTION
condition. That nominal gate is training-condition validation, not independent
DEV. Other conditions remain split-checked. Panel56 consists of this nominal
revisit,48 DEV retention conditions and seven solver roots (three verified
expansions, four unverified-teacher diagnostics). All retain true denominators.

Fixed external request tables are now admitted through exact array checks of
the requested actions, initial states, SHA and active-step timing. Identical
batch jobs are shared, without changing control/physical semantics. One policy
costs85,200 reserved transitions; main P0 plus eight nonzero checkpoints costs
766,800, with pi0 reference aliased to identical P0. Calibration reserves153,600,
so the combined reservation is920,400 within the independent cap.

New thin entries: `prepare_frontierbridge_banks.py` and
`run_retention_guard_calibration.py`. Real source prepare/audit exposed and
fixed repository-relative asset resolution: fallback is restricted to hashed
model assets, not arbitrary missing inputs. Final startup regression including
all prior required files: **234 passed in28.02 seconds, zero skips/failures**.
Actual process/progress, scalar HTTP verification and outcomes belong to the
pilot's live receipts and the locked shared research state.

The first real parent preflight completed76,800 charged transitions, then
stopped at zero optimizer updates: six reference conditions were mixed across
three executions. All49 conditions and their31 stable successes/12 stable
failures/six mixed labels remain in the original failed receipt. The explicit
new `baseline_mixed_policy=diagnostic_only` option reports all49 and gates the
43 stable frozen-parent conditions. Stable-reference student uncertainty,
losses, execution errors and timeouts still block; mixed reference conditions
never count as gains or absorption. The default remains blocking. This is used
development calibration, not independent TEST evidence. Only proven unspent
reservation is released; a fresh calibration and main stage retain total
reservation997,200 within the original one-million pilot cap. Calibration
receipts bind this policy and preserve seen-condition validation roles.


The user subsequently explicitly authorized bounded candidate diagnostic
training with nominal retention unverified and automatic adoption prohibited.
This is a new explicit branch, not a relaxation of the default strict receipt.
`continuation_eligible` is separate from adoption eligibility; diagnostic
checkpoints are never selected as an accepted improvement. Behavior calibration
uses a separate diagnostic schema without `calibrated=true` or closed-loop
qualification. Other mandatory stable-reference cohorts keep their zero-loss
and unresolved-student blocking gates.

All six complete parent numerical repetitions from both failed attempts are
reused with exact condition/controller/normalizer/physical context verification;
nominal labels remain [1,1,1,0,0,1]. Each actual student has three fresh repeats,
and exact reference/student counts are explicit per condition. P0 reuses the49
collect cases and executes only the seven solver cases. New calibration performs
20 fresh attempts and only one three-repeat physical check at attempt20
(76,800 maximum); its weights and Adam state never initialize the main arms.
Prior charged cost153,600 plus this calibration and the conservative766,800
main reserve is997,200, within the original cap. Actual fully completed main
physical maximum is690,000 because its49-case P0 cache adds no transitions;
therefore maximum actual cost920,400. All raw failures remain immutable.
