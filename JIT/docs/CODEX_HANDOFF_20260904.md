# DVGC/JIT Technical Handoff — 2026-09-04

## Purpose

This is the active takeover guide after completion of the engineering `pi_1 -> C^1 -> raw Tube_2 -> pi_2` round and the resolution-aware capability-Tube method revision.

Read first:

```text
JIT/docs/CURRENT_STATUS.md
JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md
```

Historical quantitative context:

```text
JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md
```

The 2026-09-03 handoff is superseded historical context.

---

## 1. Current project definition

JIT is a **target-free real-dynamics capability-discovery and just-in-time curriculum** framework for one fixed single-track two-wheeled robot task.

Separate:

```text
F*   unknown conceptual physical/task feasibility
E_k  cumulative empirical successful real-dynamics evidence
T_k  resolution-aware sparse physical capability Tube
R_k  single-policy realization over cumulative support
```

The final runtime target remains one unified Actor. Experts/intermediate policies are discovery probes only.

A Soft Tube is replay/curriculum support, not a certified safe/viable/reachable set.

---

## 2. Current method decisions

### Target-free mainline

Do not add desired jump distance/apex intent now.

No reward or Actor-observation change is authorized in this method revision.

Reason: the current object is a nonzero-cross-section state-space Tube. Goal-conditioned JIT is deferred until the physical Tube geometry is understood.

### Physical resolution v1

```text
root x/y/z                      0.10 m
root vx/vy/vz                   0.10 m/s
roll/pitch/yaw                  0.50 deg
root angular velocity wx/wy/wz  2.0 deg/s
steering/hip/knee angle         0.50 deg
steering/hip/knee rate          2.0 deg/s
wheel tangential speed          0.10 m/s
phase                           discrete
```

Actor observation is not the capability metric space.

Two cell profiles:

- `root_geometry_v1` for macroscopic Tube geometry/frontier diversity;
- `full_physical_v1` for finer physical configuration diversity.

---

## 3. Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- branch: `agent/two-phase-soft-tube`
- local repo: `~/DVGC`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- simulation substep: 0.005 s
- control interval: 0.020 s = 50 Hz
- hip/knee actuator range: +/-30 N m
- action order: `[steer, rear-wheel drive, hip, knee]`
- runtime expert switching: none
- final TEST/JCE/JEL: untouched

No physics change was made by the resolution-aware revision.

---

## 4. Completed artifact chain

### Experts

```text
JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json
```

- pi_up_star: 9,977,856 transitions
- pi_down_star: 25,600 transitions

### Raw Tube_0

```text
JIT/runs/soft_tube/soft_tube_train_v1_20260828
222 raw snapshots = 117 upstream + 105 downstream
```

### pi_0

```text
JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json
```

### Raw Tube_1

```text
JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
3,119 raw snapshots = 222 retained + 2,897 raw expansion
```

### pi_1 repair02

```text
JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json
```

Engineering quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 groups
```

Historical strict formal PASS remains unclaimed because the old gate retains 3 baseline-reproduction mismatches.

### C^1

Upstream 64x64:

```text
AUC 0.6903137789904502
recall 0.5934515688949522
formal AUC>=0.70 FAIL
engineering-selected
```

Downstream 64x64:

```text
AUC 1.0
recall 1.0
formal calibration PASS
```

### Raw Tube_2

```text
JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904
3,776 raw snapshots = 3,119 retained + 657 raw expansion
```

Manifest SHA:

```text
135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff
```

Smoke: GO.

### pi_2

Run id:

```text
pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904
```

Completed 10,009,600 transitions.

---

## 5. Current pi_2 evidence

Locked Tube_1 panel:

```text
pi_1 = 3115/3119
pi_2 = 3002/3119
strict regressions = 115

upstream   423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
```

Locked pi_1-negative frontier:

```text
pi_2 success = 13/14
upstream = 4/5
downstream = 9/9
successful parent groups = 3
baseline reproduction failures = 0
```

Interpretation:

```text
local frontier progression = strong
upstream policy realization = degraded
```

pi_2 remains capability evidence but is not retrospectively selected as next authority.

---

## 6. Raw Tube cardinality claim boundary

Do not state that Tube_1 expanded the capability envelope 14x or Tube_2 expanded it another 21%.

Those ratios describe raw replay snapshot counts only.

The new required quantities are:

```text
unique_root_geometry_cell_count
unique_full_physical_cell_count
new_root_geometry_cell_count
new_full_physical_cell_count
x-slice coverage
nearest source-cell distances
```

These values do not exist until the local retrospective analyses are run.

---

## 7. New production capabilities

### Capability Tube reconstruction

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

Outputs:

```text
summary.json
entries.json
cells.json
x_slices.json
projected_points.csv
x-z / x-vx / x-pitch / x-z-vx plots
```

### Resolution-aware frontier parent selection

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Future parent rule:

```text
newest shell only
+ unique parent group
+ unique exact state
+ unique root_geometry_v1 cell
```

Default max 25 distinct parent cells per phase; minimum 5.

---

## 8. First commands after takeover

```bash
cd ~/DVGC
git pull --ff-only origin agent/two-phase-soft-tube

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli

$PY -m pytest -q \
  JIT/tests/test_capability_tube_resolution.py \
  JIT/tests/test_resolution_frontier_parent_selection.py \
  JIT/tests/test_capability_progression.py \
  JIT/tests/test_iteration_policy_selection_capability.py
```

Then analyze Tube_0/1/2 exactly as specified in:

```text
JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md
```

No new GPU training is needed for these retrospective projections.

---

## 9. Required retrospective outputs

```text
JIT/runs/capability_geometry/tube0_resolution_v1
JIT/runs/capability_geometry/tube1_vs_tube0_resolution_v1
JIT/runs/capability_geometry/tube2_vs_tube1_resolution_v1
```

Before any new training, review:

- raw vs root/full cell compression;
- new root cells per round;
- x-range/slice growth;
- phase distribution;
- Tube branches/lobes/gaps;
- nearest old-cell distance of new cells.

---

## 10. Automatic workflow status

Already generic:

```text
frontier roles
-> C^k
-> raw Tube construction
-> smoke/isolation
-> baseline lock
-> train/freeze
-> locked evaluation
-> capability progression
-> prospective policy selection
```

Newly implemented but not yet integrated into one prospective workflow DAG:

```text
physical Tube projection
resolution-aware cell accounting
x-slice geometry
resolution-distinct parent-plan revision
```

Therefore the next prospective iteration must explicitly do:

```text
analyze source Tube geometry
-> prepare outcome-blind plan
-> revise parent set by distinct root cells
-> run frontier roles
```

Do not claim fully integrated resolution-aware automation before the DAG records these stages.

---

## 11. What not to do next

Do not:

- train pi_3;
- run a 90/10 replay repair automatically;
- add intent/reward changes;
- reopen A/B warm-start studies;
- rewrite C_up^1 as formal PASS;
- promote pi_2 retrospectively;
- touch final TEST/JCE/JEL;
- interpret raw Tube counts as physical envelope volume.

---

## 12. Next scientific decision after Tube geometry

Only after Tube_0/1/2 cell analysis:

- if raw additions mostly collapse into occupied cells -> improve acquisition diversity;
- if many new root cells exist but one Actor loses coverage -> investigate policy representation/retention;
- if multiple successful Tube branches emerge -> reconsider latent/goal conditioning as a separate method version;
- if new-cell/x-slice growth saturates -> define a stopping rule.

---

## 13. Claim boundary

Supported now:

- exact raw Tube artifact counts;
- resolution contract and physical Tube analysis code;
- distinct-cell parent-selection code;
- pi_2 strong locked local frontier success;
- pi_2 upstream realization loss;
- target-free mainline remains unchanged.

Not supported until local analysis:

- unique Tube_0/1/2 physical cell counts;
- physical expansion percentages;
- Tube connectivity/branch structure;
- relation between pi_2 regressions and Tube branches.
