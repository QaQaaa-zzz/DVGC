# Current JIT status — 2026-09-04

## Executive state

The historical engineering chain through `pi_2` is complete. The project is **paused before any pi_3-like training** because the old global Tube definition mixed the actual jump trajectory with late downstream/recovery states.

Current mainline:

```text
pi_up_star + pi_down_star
  -> Tube_0 / pi_0 / C^0
  -> Tube_1 / pi_1 repair02
  -> C^1 engineering path
  -> Tube_2 / pi_2
  -> locked pi_1 vs pi_2 evaluation
  -> physical resolution analysis
  -> trajectory-centered Jump-Tube redesign
  -> CURRENT: build one successful nominal centerline and re-audit Tube_0/1/2 as Jump-Tube views
```

Do not start pi_3, do not run a replay-ratio sweep, and do not touch final TEST/JCE/JEL.

## Scientific object

JIT now distinguishes:

```text
F*   unknown physical/task feasibility under fixed dynamics
E_k  cumulative successful real-dynamics evidence
J_k  trajectory-centered empirical Jump-Tube support
R_k  realization coverage of one unified policy
```

The runtime target remains one unified Actor with no expert switching.

A raw Soft Tube is replay/training support. A Jump-Tube view is the task-semantic subset used for capability accounting. Neither is a certified safe set or exact viability kernel.

## Why the method changed

Resolution-aware visualization showed that many historical downstream-labelled states sit far beyond the useful jump interval and form a dense late-recovery cloud. The old parent selector had no requirement that downstream remain descending and no longitudinal x-slice balancing.

Therefore the project no longer uses global low-score newest-shell expansion as the scientific Jump-Tube definition.

## New trajectory-centered contract

One successful full-chain real rollout defines a nominal geometric centerline:

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x spacing       = 0.1 m
actual end      = first valid landing if earlier
```

Every centerline point is a real captured simulator frame. No qpos/qvel interpolation is allowed.

Branch semantics:

```text
pre-Apex            upstream
Apex-near           apex marker
post-Apex + vz < 0  downstream
first valid landing terminal of Jump-Tube centerline
post-landing         excluded from Jump-Tube frontier
```

No goal/intent variable is added. Reward and Actor observation remain unchanged.

Implementation:

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

## Physical capability resolution v1

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

Profiles:

- `root_geometry_v1`: macroscopic Tube geometry/frontier diversity;
- `full_physical_v1`: finer physical deduplication.

Actor FIFO/history/last-action fields do not define capability cells.

## Historical raw Tubes

```text
Tube_0 =   222 raw snapshots
Tube_1 = 3,119 raw snapshots
Tube_2 = 3,776 raw snapshots
```

These remain immutable replay/provenance artifacts and must not be converted directly into capability-growth percentages.

Already measured all-state physical occupancy:

```text
Tube_0
  root cells 100
  full cells 112
  x slices 13

Tube_1
  root cells 2142
  full cells 2404
  new root vs Tube_0 2042
  new full vs Tube_0 2292
  x slices 24

Tube_2
  root cells 2446
  full cells 2871
  new root vs Tube_1 304
  new full vs Tube_1 467
  x slices 24
```

These are all-state physical Tube counts, not yet filtered Jump-Tube counts. Historical downstream counts are explicitly under re-audit.

## Jump-Tube semantic view

Implementation:

```text
JIT/src/jit_dvgc/analysis/jump_tube_view.py
JIT/cli/analyze_jump_tube_view.py
```

The view retains only:

```text
x on nominal centerline support
upstream inside corridor
downstream inside corridor AND root_vz < 0
no post-landing recovery
```

Late recovery can remain in raw replay but cannot contribute to Jump-Tube expansion claims or future frontier-parent eligibility.

## New frontier parent rule

Implementation:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The CLI now requires `--nominal-centerline`.

Prospective parent requirements:

```text
newest raw Tube shell
+ distinct parent group
+ distinct exact state
+ distinct root_geometry_v1 cell
+ x slice supported by nominal centerline
+ inside actual jump corridor
+ downstream root_vz < 0
+ no post-landing / late recovery
```

Selection is x-balanced: candidates are ranked locally inside each x slice and selected round-robin across slices instead of globally taking the lowest scores.

## Current pi_2 evidence

Training completed at 10,009,600 transitions.

Locked source panel:

```text
pi_1 3115/3119
pi_2 3002/3119
strict regressions 115

upstream:   423/427 -> 312/427
downstream: 2692/2692 -> 2690/2692
```

Locked pi_1-negative challenge:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
successful parent groups 3
baseline reproduction failures 0
```

Interpretation: clear local capability progression plus substantial upstream single-policy realization loss. `pi_2` remains capability evidence but is not retrospectively selected as the next policy authority.

## Immediate next tasks

```text
1. obtain one successful canonical pi_1 natural rollout
2. build nominal centerline from real trace frames
3. rebuild Jump-Tube_0/1/2 semantic views
4. quantify how much historical downstream support disappears after vz/x/landing filtering
5. inspect per-0.1m cross-sections and gaps
6. only then predeclare the next trajectory-centered frontier plan
```

Full commands and rationale:

`JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md`

## Automation maturity

Existing generic automation still covers:

```text
frontier roles -> C^k -> raw Tube -> smoke/isolation -> baseline lock
-> candidate train/freeze -> locked evaluation -> capability progression -> selection
```

New trajectory-centered production capabilities now exist for:

```text
successful rollout -> nominal centerline
physical Tube -> Jump-Tube view
centerline + geometry -> x-balanced frontier plan revision
```

A complete prospective automatic round has not yet exercised these new stages. Do not claim end-to-end trajectory-centered automation until a recorded workflow DAG contains them.

## Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- branch: `agent/two-phase-soft-tube`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee actuator range: +/-30 N m
- actions: `[steer, rear-wheel drive, hip, knee]`
- runtime expert switching: none
- final TEST/JCE/JEL: untouched

## Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
