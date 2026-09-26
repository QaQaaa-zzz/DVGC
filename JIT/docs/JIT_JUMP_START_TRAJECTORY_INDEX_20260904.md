# JIT Fixed Jump-Start Trajectory Index — 2026-09-04

> New direction: [paper outline](JIT_PAPER_OUTLINE.md) and [review](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md). This index preserves original experiment and selection wording. Historical actor-selection statements do not define the new probe bank. The new workflow and correctness repairs remain pending; no new training was performed by the documentation review.

> Updated 2026-09-05. Older entries below retain their historical wording.
> Current authority is `JIT/docs/CURRENT_STATUS.md`: π3 is trained but its
> historical mixed-endpoint selection is quarantined, and no π4 is authorized.

## Purpose and boundary

This file is the lookup index for the paired `pi_0` / `pi_1` / `pi_2`
trajectory comparison.  Every listed rollout uses the same fixed ground
jump-start contract:

```text
root x = 2.5 m
default keyframe pose and declared initial velocity
authoritative env.step dynamics only
deterministic frozen unified Actor
no Tube/RSI state restoration
no training
seeds = 9400001, 9400002, 9400003
```

The primary trajectory gate ends scientifically at the first valid landing.
A later physical failure is retained in the trace and reported separately as a
full-recovery failure.  These results establish conditional performance from
the fixed jump start; they do not establish connection from the historical
natural reset, and the three seeds are not treated as three distinct initial
physical states.

## Common result root

```text
JIT/runs/pi_jump_start_seed_sweep_20260904/
```

Each seed directory contains:

- `declaration.json`: frozen policy, reset contract, and zero-training declaration;
- `reset_diversity.json`: audit of the fixed initial physical state;
- `report.json`: jump-to-landing and full-recovery outcomes;
- `canonical_trace.npz`: full simulator-state/action/metric trajectory for later numerical comparison;
- `canonical_trace.json`: trace metadata and recorded artifact identity.

## Frozen policies and intended use

| Policy | Config | Checkpoint | Why it is retained |
|---|---|---|---|
| `pi_0` | `JIT/configs/pi_unified_round1_natural10.json` | `JIT/runs/pi_unified/pi_unified_round1_natural10_10009600_seed821101_20260831/checkpoints/transition_10009600` | Historical iteration-0 baseline; compare early jump shape and recovery behavior against later policies. |
| `pi_1` | `JIT/configs/pi_unified_iter1_tube1_core_replay75_natural10.json` | `JIT/runs/pi_unified/pi_1_tube1_core_replay75_natural10_10009600_seed821101_20260903/checkpoints/transition_10009600` | Current proposal/continuation authority and reference trajectory family for the active jump-start method. |
| `pi_2` | `JIT/configs/pi_unified_iter2_c1_64x64_engineering_core75_natural10_20260904.json` | `JIT/runs/pi_unified/pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904/checkpoints/transition_10009600` | Historical unselected comparator; diagnose whether later realization reaches a different Apex/landing region without treating it as selected authority. |

## Per-seed trajectory paths and results

All nine rows passed the jump-to-first-valid-landing gate.  `pi_0` additionally
completed recovery in all three runs; `pi_1` and `pi_2` landed successfully but
later terminated at the roll limit.

| Policy | Seed | Report | Full trace | Result and comparison use |
|---|---:|---|---|---|
| `pi_0` | 9400001 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_0/seed_9400001/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_0/seed_9400001/canonical_trace.npz` | jump GO; recovery success; iteration-0 paired baseline |
| `pi_0` | 9400002 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_0/seed_9400002/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_0/seed_9400002/canonical_trace.npz` | jump GO; recovery success; iteration-0 paired baseline |
| `pi_0` | 9400003 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_0/seed_9400003/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_0/seed_9400003/canonical_trace.npz` | jump GO; recovery success; iteration-0 paired baseline |
| `pi_1` | 9400001 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_1/seed_9400001/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_1/seed_9400001/canonical_trace.npz` | jump GO; post-landing roll limit; active-authority comparator |
| `pi_1` | 9400002 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_1/seed_9400002/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_1/seed_9400002/canonical_trace.npz` | jump GO; post-landing roll limit; active-authority comparator |
| `pi_1` | 9400003 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_1/seed_9400003/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_1/seed_9400003/canonical_trace.npz` | jump GO; post-landing roll limit; active-authority comparator |
| `pi_2` | 9400001 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_2/seed_9400001/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_2/seed_9400001/canonical_trace.npz` | jump GO; post-landing roll limit; unselected-policy comparator |
| `pi_2` | 9400002 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_2/seed_9400002/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_2/seed_9400002/canonical_trace.npz` | jump GO; post-landing roll limit; unselected-policy comparator |
| `pi_2` | 9400003 | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_2/seed_9400003/report.json` | `JIT/runs/pi_jump_start_seed_sweep_20260904/pi_2/seed_9400003/canonical_trace.npz` | jump GO; post-landing roll limit; unselected-policy comparator |

The corresponding metadata path for any row is the trace path with `.npz`
replaced by `.json`; the predeclared protocol is in `declaration.json` in the
same seed directory.

## Compact comparison

| Policy | Jump GO | Full recovery | Mean Apex x (m) | Mean maximum root z (m) | Mean first-landing x (m) |
|---|---:|---:|---:|---:|---:|
| `pi_0` | 3/3 | 3/3 | 3.327924 | 0.568867 | 3.843207 |
| `pi_1` | 3/3 | 0/3 | 3.714613 | 0.649268 | 4.060181 |
| `pi_2` | 3/3 | 0/3 | 3.785959 | 0.732525 | 4.195559 |

This table is descriptive evidence for later paired trajectory comparison.  It
is not policy selection evidence and does not alter the locked historical
`pi_1` versus `pi_2` result.

## Active centerline artifact

The active real-frame acquisition scaffold is the locked `pi_0` trajectory:

```text
JIT/runs/nominal_centerline/pi_0_jump_start_centerline_v3_20260904/centerline.json
```

It contains 14 captured simulator frames/cells from `x=2.5` through `x=3.8 m`
and ends at the first valid landing.  It is both the longitudinal slice
scaffold and the reference for `pi_0` proposal acquisition.  The `pi_1` and
`pi_2` trajectories remain comparison evidence, not competing centerlines.

## Active family-landing iteration artifacts

Common root:

```text
JIT/runs/iteration_auto/pi_1_to_pi_2_pi0_centerline_family_landing_20260904/
```

| Artifact | Path | Purpose |
|---|---|---|
| Frozen every-slice plan | `frontier_plan_causal.json` | Declares the `pi_0` proposal policy, `pi_0/pi_1/pi_2` evaluator family, role split and first-valid-landing criterion. |
| TRAIN role | `frontier_train/role_manifest.json` | 527 reached candidates; 525 family positives. This is the only role allowed into replay. |
| TRAIN per-policy labels | `frontier_train/labels/per_policy/{pi_0,pi_1,pi_2}/summary.json` | Records controller-specific landing results: 485, 519 and 450 positives respectively. |
| CALIBRATION role | `frontier_calibration/role_manifest.json` | Disjoint holdout: 181/184 family positives; never embedded in replay. |
| ACCEPTANCE role | `frontier_acceptance/role_manifest.json` | Disjoint holdout: 181/184 family positives; never embedded in replay. |
| Physical capability evidence | `causal_jump_capability/summary.json` | Resolution-aware result: 382 unique TRAIN-positive root cells beyond the 14-cell centerline; holdouts have 134/135 positive root cells. |
| TRAIN replay Tube | `tube_2_policy_family_landing/manifest.json` | Retains 3119 Tube_1 rows and adds 525 verified TRAIN landing positives, for 3644 rows total. No fitted C-field or recovery label is used. |
| Tube-RSI smoke | `tube2_landing_replay_smoke/report.json` | GPU-backed 16-interaction smoke over both phases; GO before training. |
| State-disjoint role views | `disjoint_role_views/summary.json` | TRAIN-priority exact-state partition; CALIBRATION excludes 31 duplicate states and ACCEPTANCE excludes 40, with exact/near overlap zero. |
| Role isolation audit | `role_isolation.json` | Confirms exact and 0.01-tolerance near-state isolation across the derived role views. |
| Pre-training baseline lock | `acceptance_baseline_lock/baseline_lock.json` | Reuses the locked 3119-state pi_1 core and fixes six pi_1 landing failures from three parent groups before candidate training. |
| Formal candidate run | `../../pi_unified/pi_2_landing_replay_pi1_actor_warmstart_core75_natural10_10009600_seed821101_20260904/formal_report.json` | Exact 10,009,600-transition unified training; pi_1 Actor/normalizer warm start with fresh critic and optimizer. |
| Frozen candidate | `../../frozen_unified/pi_2_landing_replay_pi1_actor_warmstart_10009600_20260904/frozen_unified_policy.json` | Immutable pi_2 landing-replay candidate used by the locked gate. |
| Rejected recovery-mismatch diagnostic | `pi2_landing_replay_acceptance_gate/summary.json` | Preserved audit result: boundary passed but old recovery-based core criterion reported 10 upstream regressions; not used for selection. |
| Active landing gate | `pi2_landing_replay_acceptance_gate_landing_contract/summary.json` | Active contract result: 3119/3119 core landings, zero regressions, and 4/6 boundary gains across two parent groups. |
| Capability/realization decision | `pi2_landing_replay_capability_progression_represented_phases.json` | Prospective decision: envelope progressed and candidate authority eligible; only phases represented by locked negative states are required for boundary gain. |
| Selected policy | `../../iteration_selection/pi_2_landing_replay_20260904/selected_policy.json` | Selected next engineering iteration authority; not a final Actor, certified safe set, JCE, or JEL claim. |

Engineering failures are preserved beside the successful data with explicit
`engineering_error` names.  They document the rejected Warp stacked-state
batch path and the original sequential cross-policy GPU-memory accumulation;
they are not scientific negative samples.

## Tube3, π3 and predictor evidence — 2026-09-05

Common root:

```text
JIT/runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904/
```

| Artifact | Relative path | Purpose and current interpretation |
|---|---|---|
| Expanded plan | `frontier_plan_causal.json` | Locked π0 proposer, π0/π1/π2 evaluator family and wider declared scan. |
| TRAIN labels | `frontier_train/role_manifest.json` | 1,230/1,258 family positives; authoritative observed TRAIN outcomes. |
| Tube3 | `tube_3_policy_family_landing/manifest.json` | 4,803 rows; 1,159-row deduplicated raw increment over Tube2. |
| Strict isolation | `role_isolation_strict.json` | Outcome-blind derived holdouts; exact/near overlap and target-Tube overlap zero. |
| Causal capability | `causal_jump_capability/summary.json` | 713 new TRAIN root cells under fixed-jump-start family semantics. |
| Control geometry | `tube3_control_tube_geometry/summary.json` | +714 root and +897 full-physical all-state Tube cells. |
| π3 training | `../../pi_unified/pi_3_landing_replay_pi2_actor_warmstart_core75_natural10_10009600_seed821101_20260905/formal_report.json` | Completed 10,009,600 transitions; π2 Actor/normalizer warm start, fresh critic/optimizer. |
| π3 frozen manifest | `../../frozen_unified/pi_3_landing_replay_pi2_actor_warmstart_10009600_20260905/frozen_unified_policy.json` | Frozen trained checkpoint identity. |
| Historical gate | `pi3_landing_replay_acceptance_gate/summary.json` | Preserves 89 improvements/30 regressions, but mixes stable-recovery baseline with first-landing candidate and is not fair selection evidence. |
| Historical selection | `../../iteration_selection/pi_3_landing_replay_20260905/selected_policy.json` | Immutable engineering registration; quarantined from current π4 authority. |
| TRAIN realization | `pi3_train_realization/increment_summary.json` | Same-batch π0/π1/π2/π3 comparison and 1,061/1,159 π3 increment realization. |
| Predictor | `family_landing_predictor/summary.json` | Advisory upstream model; old ACCEPTANCE AUC 0.89249 and 6/9 accepted negatives. |
| Predictor field | `family_landing_predictor/upstream/field.npz` | Frozen model parameters used for the next pre-outcome scores. |

## Expanded predictor-audit round — incomplete

Common root:

```text
JIT/runs/iteration_auto/pi_3_to_pi_4_pi0_centerline_family_landing_predictor_audit_20260905/
```

The directory name does not authorize π4.

| Artifact | Relative path | Purpose/status |
|---|---|---|
| Expanded plan | `frontier_plan_causal_expanded.json` | Declares 0.1–0.7 m lookbacks and five strengths while retaining π0 proposer/family identities. |
| Predictor audit protocol | `predictor_audit_protocol.json` | Predeclares score-before-outcome transfer audit. |
| TRAIN catalog | `frontier_train/acquisition/{catalog,summary}.json` | Acquisition complete: 1,754 candidates. |
| CALIBRATION catalog | `frontier_calibration/acquisition/{catalog,summary}.json` | Acquisition complete: 583 candidates. |
| ACCEPTANCE catalog | `frontier_acceptance/acquisition/{catalog,summary}.json` | Acquisition complete: 574 candidates. |
| Locked TRAIN scores | `predictor_forward_scores_train.json` | 1,038 upstream scores; no outcome labels read. |
| Locked CALIBRATION scores | `predictor_forward_scores_calibration.json` | 342 upstream scores; no outcome labels read. |
| Locked ACCEPTANCE scores | `predictor_forward_scores_acceptance.json` | 333 upstream scores; no outcome labels read. |
| TRAIN failure | `frontier_train/labels/failure.json` | π0 long-lived evaluator failed after 1,409/1,754; not a scientific label. |
| CALIBRATION failure | `frontier_calibration/labels/failure.json` | π0/π1 preserved; π2 GPU allocation failure. |
| ACCEPTANCE failure | `frontier_acceptance/labels/failure.json` | π0/π1 preserved; π2 GPU allocation failure. |

Resume with independent evaluator shards of at most 600 candidates, strict merge
and the already locked scores. Do not reacquire or change seeds/horizon/endpoint.
