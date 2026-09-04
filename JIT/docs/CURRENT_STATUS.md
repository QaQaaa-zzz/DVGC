# Current JIT status — 2026-09-04

## Executive state

The engineering `pi_1 -> C^1 -> raw Tube_2 -> pi_2` round is complete. The project is now **paused before pi_3** while existing Tube_0/1/2 artifacts are re-measured in a resolution-aware physical capability space.

Current chain:

```text
pi_up_star + pi_down_star
  -> raw Tube_0 = 222 replay snapshots
  -> pi_0
  -> C^0
  -> raw Tube_1 = 3,119 replay snapshots
  -> pi_1 repair02 engineering authority
  -> C^1 engineering path
  -> raw Tube_2 = 3,776 replay snapshots
  -> pi_2 trained/frozen
  -> locked pi_1 vs pi_2 evaluation
  -> capability progression reinterpretation
  -> CURRENT: physical-cell Tube reconstruction + resolution-aware frontier redesign
```

Do not start pi_3 or a replay-ratio repair yet.

Final TEST/JCE/JEL remains untouched.

---

## Current JIT meaning

Keep these quantities separate:

```text
F*   conceptual physical/task feasibility under fixed dynamics
E_k  cumulative empirical successful real-dynamics evidence
T_k  resolution-aware sparse occupied physical capability cells
R_k  realization coverage of one unified policy over cumulative support
```

The latest policy is not the definition of physical feasibility. A later failure does not erase earlier successful evidence. Conversely, many exact snapshots in one small neighborhood do not constitute many independent capability states.

---

## Intent decision

**No intent is added now.** Reward and Actor observation remain unchanged.

Reason: the present research object is a target-free state-space Tube with nonzero cross-section. At one `x`, multiple heights, velocities and attitudes may all be valid. Adding a fixed desired distance/apex target would change the task toward goal-conditioned trajectory families and would require a separate reward/method redesign.

Goal-conditioned JIT is deferred until the target-free Tube geometry is understood.

---

## Physical capability resolution v1

Primary resolution contract:

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

Angular velocity is deliberately **2 deg/s**, not 5 deg/s.

Two cell profiles:

- `full_physical_v1`: fine physical deduplication;
- `root_geometry_v1`: root pose/twist only, used for Tube shape/frontier diversity.

Actor FIFO/history/last-action fields do not define physical cell identity.

Implementation:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

---

## What the new Tube analysis produces

For each raw Soft Tube:

```text
raw snapshot count
unique root-geometry cells
unique full-physical cells
duplicate fractions
phase-specific cell counts
0.10 m x-slices
cross-section statistics by x
CSV physical projection
x-z / x-vx / x-pitch / x-z-vx visualizations
```

For Tube_k relative to Tube_(k-1):

```text
new root-geometry cells
new full-physical cells
new cells by phase
new root cells per raw added snapshot
nearest source-cell distance in resolution units
```

This is the new basis for quantitative Tube expansion.

---

## Historical raw Tube counts — still valid, but demoted

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

```text
222 raw snapshots
117 upstream
105 downstream
```

Manifest SHA:

`c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

```text
3,119 raw snapshots
= 222 retained Tube_0
+ 2,897 raw expansion rows

upstream   427
downstream 2,692
```

Manifest SHA:

`817a980a5dd84f36507f7620d925ff9c68e982edfd82a80` is invalid shorthand; authoritative manifest SHA is:

`817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_c1_64x64_engineering_20260904` is invalid shorthand. Authoritative path:

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

```text
3,776 raw snapshots
= 3,119 retained Tube_1
+ 657 raw expansion rows

upstream   902
downstream 2,874
```

Manifest SHA:

`135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`

Do not report `3,119/222` or `3,776/222` as capability-envelope multipliers. Do not report `657/3119` as a capability-space growth percentage until cell analysis is complete.

---

## pi_1 / C^1 / pi_2 evidence

Selected engineering pi_1 repair02:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

Historical quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 parent groups
```

Historical strict formal Iteration-1 PASS is not claimed because 3 old boundary baseline-reproduction mismatches remain quarantined.

C^1:

```text
upstream 64x64:
  AUC 0.6903137789904502
  recall 0.5934515688949522
  original AUC>=0.70 formal gate FAIL
  engineering-selected only

downstream 64x64:
  AUC 1.0
  recall 1.0
  formal calibration PASS
```

pi_2 training:

```text
10,009,600 transitions
90% Tube / 10% natural
inside Tube: 75% retained Tube_1 / 25% raw Tube_2 expansion
```

Locked pi_1 vs pi_2 source-panel result:

```text
pi_1 3115/3119
pi_2 3002/3119
strict regressions 115

upstream:   423/427 -> 312/427
downstream: 2692/2692 -> 2690/2692
```

Locked pi_1-negative frontier challenge:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
successful parent groups 3
baseline reproduction failures 0
```

Interpretation:

- local frontier progression evidence: strong;
- upstream single-policy realization: substantially degraded;
- pi_2 remains capability evidence;
- pi_2 is not retrospectively selected as next authority.

---

## Resolution-aware frontier parent selection

New production capability:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Future parent rule:

```text
newest raw Tube shell only
+ distinct parent group
+ distinct exact state
+ distinct root_geometry_v1 cell
```

Default maximum: 25 distinct parent cells per phase.
Minimum before frontier execution: 5 distinct newest-shell root-geometry cells per phase.

This revision occurs before outcomes and preserves role assignment, policy/Tube identity, probe panel, seeds, physics, reward and TEST isolation.

---

## Automation maturity

Already generic:

```text
frontier role execution
C^k fitting/calibration
raw Tube construction
Tube-RSI smoke
role isolation
locked baseline
candidate train/freeze
locked paired evaluation
capability progression decision
prospective policy selection
```

Newly implemented but not yet exercised in a complete prospective automatic round:

```text
physical Tube projection
resolution-aware cell accounting
x-slice geometry/plots
resolution-aware distinct-cell frontier-parent revision
```

Therefore the next prospective iteration must first use:

```text
analyze source Tube capability geometry
-> prepare ordinary outcome-blind frontier plan
-> revise plan by distinct root geometry cells
-> run frontier roles
```

Do not claim end-to-end automatic resolution-aware JIT until a workflow artifact records these stages.

---

## Immediate local operator task

After pulling the branch, first run tests/compile, then analyze existing Tubes. No GPU training is required for this retrospective projection.

Recommended outputs:

```text
JIT/runs/capability_geometry/tube0_resolution_v1
JIT/runs/capability_geometry/tube1_vs_tube0_resolution_v1
JIT/runs/capability_geometry/tube2_vs_tube1_resolution_v1
```

Commands are maintained in `JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md`.

The results should answer whether the 2,897 and 657 raw added snapshots correspond to broad physical coverage or mostly dense sampling of existing neighborhoods.

---

## Immutable task identity

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
- actions: `[steer, rear-wheel drive, hip, knee]`
- runtime expert switching: none
- final TEST/JCE/JEL: untouched

Older +/-50 N m documentation was incorrect; the XML/runtime validator uses +/-30 N m. No physics file was modified.

---

## Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md`
5. `JIT/docs/JIT_CAPABILTY_PROGRESS_REPORT_20260904.md` is a misspelling; authoritative historical progress report is `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
