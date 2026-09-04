# DVGC Repository Instructions

## Current research truth — 2026-09-04

DVGC/JIT is an iterative **real-dynamics jump-capability discovery + just-in-time curriculum** project for one fixed single-track two-wheeled robot.

The current mainline is now **trajectory-centered** rather than globally target-free in state-space sampling:

```text
one successful real jump trajectory
-> real-frame nominal centerline
-> 0.1 m longitudinal x slices
-> local cross-section frontier widening
-> continuation evidence / Tube curriculum
-> one unified policy
-> locked capability + realization evaluation
```

No goal/intent variable is added to the Actor. Reward and task semantics remain unchanged. The centerline is a geometric scaffold for Tube identification, not a fixed controller target.

Scientific objects:

1. `F*`: unknown physical/task feasibility under fixed dynamics; JIT does not prove it.
2. `E_k`: cumulative successful real-dynamics evidence.
3. `J_k`: trajectory-centered empirical Jump-Tube support.
4. `R_k`: realization coverage of one unified Actor over accumulated support.

A raw Soft Tube remains TRAIN replay/curriculum support. A Jump-Tube view is the task-semantic subset used for jump-boundary accounting. Neither is a certified safe set, viability kernel, invariant set, reachability proof, or proof of the physical jump limit.

## Current chain

```text
pi_up_star + pi_down_star
  -> raw Tube_0 = 222
  -> pi_0
  -> C^0
  -> raw Tube_1 = 3,119
  -> pi_1 repair02 engineering authority
  -> C^1 engineering path
  -> raw Tube_2 = 3,776
  -> pi_2 trained/frozen
  -> locked pi_1 vs pi_2 evaluation
  -> physical-resolution analysis
  -> CURRENT: nominal centerline + filtered Jump-Tube_0/1/2 reconstruction
```

Do not train pi_3 or run replay-ratio sweeps yet. Final TEST/JCE/JEL remains untouched.

## Nominal Jump-Tube contract v1

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x spacing       = 0.1 m
actual terminal = first valid landing if earlier
```

Every centerline point must be one real captured simulator frame. qpos/qvel interpolation is forbidden.

Branch semantics:

```text
pre-Apex            upstream
Apex-near           apex marker
post-Apex + vz < 0  downstream
first valid landing terminal
post-landing         excluded from Jump-Tube frontier
```

Implementation:

- `JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py`
- `JIT/cli/build_nominal_jump_centerline.py`
- `JIT/src/jit_dvgc/analysis/jump_tube_view.py`
- `JIT/cli/analyze_jump_tube_view.py`

## Capability-state resolution v1

Actor observation space is not capability metric space. FIFO history, last action, acceleration and validity bits remain controller inputs but do not define physical-cell identity.

| Quantity | Resolution |
|---|---:|
| root x/y/z | 0.10 m |
| root vx/vy/vz | 0.10 m/s |
| roll/pitch/yaw | 0.50 deg |
| root angular velocity wx/wy/wz | 2.0 deg/s |
| steering/hip/knee angle | 0.50 deg |
| steering/hip/knee rate | 2.0 deg/s |
| wheel tangential speed | 0.10 m/s |
| phase | discrete |

Profiles:

- `root_geometry_v1`: primary geometric Tube/frontier diversity space;
- `full_physical_v1`: finer physical deduplication.

Implementation:

- `JIT/src/jit_dvgc/analysis/capability_tube.py`
- `JIT/cli/analyze_capability_tube.py`

## Historical raw Tube count semantics

```text
Tube_0 raw snapshots =   222
Tube_1 raw snapshots = 3,119
Tube_2 raw snapshots = 3,776
```

These are replay/provenance counts only.

Already measured all-state physical occupancy:

```text
Tube_0 root/full = 100 / 112
Tube_1 root/full = 2142 / 2404
Tube_2 root/full = 2446 / 2871
```

Those values are still not Jump-Tube sizes because historical downstream-labelled states include late recovery. Recompute filtered Jump-Tube views before making downstream capability claims.

## Trajectory-centered frontier rule

Exact SHA uniqueness and physical-cell uniqueness are necessary but insufficient.

Future pre-outcome frontier revision:

- `JIT/src/jit_dvgc/acquisition/resolution_frontier.py`
- `JIT/cli/prepare_resolution_aware_frontier_plan.py`

The CLI requires a locked nominal centerline.

Parent rule:

```text
newest source-Tube shell
+ unique parent group
+ unique exact state
+ unique root_geometry_v1 cell
+ x slice supported by centerline
+ inside actual jump corridor
+ downstream root_vz < 0
+ no post-landing/late-recovery frontier state
```

Selection is local/x-balanced: rank frontier candidates inside each x slice and round-robin across slices. Do not globally let one dense region consume the acquisition budget.

Role assignment remains outcome-blind `TRAIN/TRAIN/TRAIN/CALIBRATION/ACCEPTANCE`.

## Current pi_2 evidence

```text
source panel:
  pi_1 3115/3119
  pi_2 3002/3119

upstream:
  423/427 -> 312/427

downstream:
  2692/2692 -> 2690/2692

locked pi_1-negative challenge:
  pi_2 13/14
  upstream 4/5
  downstream 9/9
  successful parent groups 3
  baseline reproduction failures 0
```

Interpretation: local capability progression is real, while upstream single-policy realization degraded strongly. `pi_2` remains capability evidence but is not retrospectively promoted.

## Current automatic-iteration status

Generic infrastructure exists for:

```text
frontier roles -> C^k -> raw Tube -> smoke/isolation -> locked baseline
-> candidate train/freeze -> locked evaluation -> capability progression -> selection
```

New production capabilities exist for:

```text
successful rollout -> nominal centerline
physical Tube -> Jump-Tube semantic view
centerline + geometry -> x-balanced frontier plan revision
```

Do not claim end-to-end trajectory-centered automation until a prospective workflow artifact explicitly records these new stages.

## Immutable physical/task contracts

- branch: `agent/two-phase-soft-tube`;
- XML: `assets/orange_bike_4kg_horizontal.xml`;
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`;
- payload: 2 kg;
- control: 50 Hz;
- hip/knee actuator range: +/-30 N m;
- action order: `[steer, rear-wheel drive, hip, knee]`;
- runtime: one unified Actor, no expert switching;
- no silent XML/physics/reward/action/task-geometry/TEST changes.

## Data-role contract

- `TRAIN`: continuation fitting + qualifying replay expansion only;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development comparison only;
- final TEST/JCE/JEL: untouched.

Parent-group disjointness remains mandatory. Physical-cell/x-slice diversity supplements provenance isolation; it does not replace it.

## Repository/Git safety

- preserve unrelated work;
- never reset, clean, stash, rebase or force-push;
- use `/home/qy/mujoco_playground/.venv/bin/python`;
- keep CLIs thin and reusable logic under `JIT/src/jit_dvgc/`;
- compile/test structural changes before any cleanup.

## Current authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
