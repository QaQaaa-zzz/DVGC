# DVGC/JIT Technical Handoff — 2026-09-04

## Purpose

Active takeover guide after the historical `pi_1 -> C^1 -> raw Tube_2 -> pi_2` round and the subsequent **trajectory-centered Jump-Tube redesign**.

Read first:

```text
JIT/docs/CURRENT_STATUS.md
JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md
```

Historical quantitative context:

```text
JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md
```

The 2026-09-03 handoff is superseded historical context.

## 1. Current project definition

JIT now identifies a physically resolved state-space Tube around one successful real jump trajectory.

```text
successful rollout
-> real-frame centerline
-> 0.1 m x slices
-> local cross-section widening
-> continuation evidence
-> just-in-time replay curriculum
-> one unified Actor
```

No goal/jump intent is added to the Actor. Reward and Actor observation remain unchanged.

Scientific objects:

```text
F*  unknown physical/task feasibility
E_k cumulative successful real-dynamics evidence
J_k trajectory-centered Jump-Tube support
R_k single-policy realization coverage
```

A raw Soft Tube is replay/provenance; a Jump-Tube view is capability accounting. Neither is a certified safe/viable/reachable set.

## 2. Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- branch: `agent/two-phase-soft-tube`
- local repo: `~/DVGC`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee: +/-30 N m
- actions: `[steer, rear-wheel drive, hip, knee]`
- runtime expert switching: none
- final TEST/JCE/JEL: untouched

## 3. Historical completed artifact chain

```text
pi_up_star + pi_down_star
-> raw Tube_0 = 222
-> pi_0
-> C^0
-> raw Tube_1 = 3,119
-> pi_1 repair02
-> C^1 engineering path
-> raw Tube_2 = 3,776
-> pi_2
-> locked pi_1 vs pi_2 evaluation
```

Expert identities:

```text
pi_up_star
  9,977,856 transitions
  actor f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081

pi_down_star
  25,600 transitions
  actor 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

Historical raw Tube paths:

```text
Tube_0 JIT/runs/soft_tube/soft_tube_train_v1_20260828
Tube_1 JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
Tube_2 JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904
```

## 4. Existing physical-resolution evidence

All-state physical occupancy measured locally:

```text
Tube_0
  raw 222
  root cells 100
  full cells 112
  x slices 13

Tube_1
  raw 3119
  root cells 2142
  full cells 2404
  new root vs Tube_0 2042
  new full vs Tube_0 2292
  x slices 24

Tube_2
  raw 3776
  root cells 2446
  full cells 2871
  new root vs Tube_1 304
  new full vs Tube_1 467
  x slices 24
```

These are **all-state control-Tube counts**. Historical downstream-labelled states include late recovery and must not be described as descending Jump-Tube size.

## 5. Current pi_2 evidence

Locked source panel:

```text
pi_1 3115/3119
pi_2 3002/3119
strict regressions 115
upstream   423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
```

Locked pi_1-negative frontier:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
successful parent groups 3
baseline reproduction failures 0
```

Interpretation: real local capability progression plus strong upstream realization loss. pi_2 remains capability evidence but is not next authority.

## 6. New method contract

Nominal centerline:

```text
x start 2.5 m
x hard max 4.2 m
dx 0.1 m
actual terminal first valid landing if earlier
real captured frames only
no qpos/qvel interpolation
```

Branch semantics:

```text
pre-Apex            upstream
Apex-near           apex marker
post-Apex + vz < 0  downstream
post-landing         excluded from Jump-Tube frontier
```

Physical resolution:

```text
position 0.10 m
linear velocity 0.10 m/s
orientation 0.50 deg
root angular velocity 2.0 deg/s
joint angle 0.50 deg
joint rate 2.0 deg/s
wheel tangential speed 0.10 m/s
```

## 7. New production code

Nominal centerline:

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

Physical Tube:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

Jump-Tube semantic view:

```text
JIT/src/jit_dvgc/analysis/jump_tube_view.py
JIT/cli/analyze_jump_tube_view.py
```

Trajectory-centered frontier:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Automatic workflow preparation:

```text
JIT/cli/prepare_iterative_envelope_workflow.py
```

It now requires `--canonical-evaluation-report` and emits explicit centerline/source-geometry/source-Jump-Tube/frontier-revision/target-geometry/target-Jump-Tube stages.

## 8. Prospective frontier rule

```text
newest raw Tube shell
+ unique parent group
+ unique exact state
+ unique root_geometry_v1 cell
+ centerline-supported x slice
+ inside actual jump corridor
+ downstream root_vz < 0
+ no post-landing / late recovery
```

Selection is local per x slice and round-robin across slices; the old global-lowest-score selection is retired for prospective trajectory-centered rounds.

## 9. First commands after takeover

```bash
cd ~/DVGC
git pull --ff-only origin agent/two-phase-soft-tube

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli

$PY -m pytest -q \
  JIT/tests/test_nominal_jump_centerline.py \
  JIT/tests/test_jump_tube_view.py \
  JIT/tests/test_capability_tube_resolution.py \
  JIT/tests/test_resolution_frontier_parent_selection.py \
  JIT/tests/test_capability_progression.py \
  JIT/tests/test_iteration_policy_selection_capability.py \
  JIT/tests/test_iterative_envelope_automation.py
```

## 10. Immediate local scientific task

Do **not** train a new policy first.

1. Run one successful canonical natural evaluation of selected pi_1.
2. Build the nominal centerline.
3. Rebuild Tube_0/1/2 as filtered Jump-Tube views.
4. Quantify how much late downstream/recovery support is removed.
5. Inspect per-x cross-sections and gaps.
6. Only then prepare the next prospective workflow.

Exact commands:

`JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md`

## 11. Automatic workflow state

The code now integrates the trajectory-centered stages into future generated workflow configs.

However, no complete prospective run has executed this new DAG yet. Therefore:

```text
trajectory-centered automation code integrated = YES
trajectory-centered end-to-end experimental validation = NO
```

Do not claim a fully demonstrated automatic JIT cycle until one future workflow artifact completes these stages.

## 12. What not to do

Do not:

- train pi_3 before the Jump-Tube retrospective audit;
- run automatic replay-ratio repairs;
- add intent/reward changes;
- rewrite C_up^1 as formal PASS;
- promote pi_2 retrospectively;
- delete late recovery from immutable raw Tube history;
- count late recovery as Jump-Tube expansion;
- touch final TEST/JCE/JEL.

## 13. Next decision after filtered Tube geometry

- If historical downstream support collapses strongly after filtering: redesign future acquisition budgets around sparse jump slices.
- If Jump-Tube_2 still adds many new root cells but one Actor loses upstream coverage: investigate policy representation/retention.
- If multiple real trajectory branches emerge: only then consider a separate multi-centerline or goal-conditioned method version.
- If new Jump-Tube cell/slice growth saturates: define a stopping rule.
