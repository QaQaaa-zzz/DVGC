# JIT agent and maintenance rules

## Scope and safety

- `JIT/` is the active implementation area.
- Work only on `agent/two-phase-soft-tube` unless explicitly told otherwise.
- Never reset, clean, stash, rebase, force-push, overwrite or reformat unrelated user work.
- Use `/home/qy/mujoco_playground/.venv/bin/python`.
- Fixed task identity: `assets/orange_bike_4kg_horizontal.xml`, SHA `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`, 2 kg payload, 50 Hz control, hip/knee +/-30 N m, actions `[steer, rear-wheel drive, hip, knee]`.
- Unified runtime never switches experts.
- Final TEST/JCE/JEL remains untouched until method/stopping/final policy are frozen.

## Scientific contract — trajectory-centered Jump Tube

The current mainline studies a **state-space Tube around one successful real jump trajectory**.

```text
successful full-chain rollout
-> real-frame nominal centerline
-> x-indexed physical cross-sections
-> local frontier widening
-> continuation evidence
-> just-in-time replay curriculum
-> one unified Actor
```

Do not add goal/jump intent to the current Actor. The centerline is a geometry scaffold for capability identification, not a controller target and not a reward change.

Scientific objects:

- `F*`: conceptual physical/task feasibility, not proved by JIT;
- `E_k`: cumulative successful real-dynamics evidence;
- raw Soft Tube: exact replayable TRAIN support;
- `J_k`: trajectory-centered Jump-Tube support;
- `R(pi_k, J_k)`: realization coverage of one unified policy.

## Nominal centerline v1

Implementation:

```text
jit_dvgc.analysis.nominal_jump_centerline
JIT/cli/build_nominal_jump_centerline.py
```

Contract:

```text
x_min = 2.5 m
x_hard_max = 4.2 m
dx = 0.1 m
actual end = first valid landing if earlier
real captured frames only
no qpos/qvel interpolation
```

Semantics:

```text
pre-Apex            upstream
Apex-near           apex marker
post-Apex + vz < 0  downstream
first valid landing terminal
post-landing         excluded from Jump-Tube frontier
```

A centerline may only be built from a completed successful canonical natural evaluation. If the deterministic canonical rollout is not full-recovery successful, stop and predeclare another successful real-rollout source; never fabricate a centerline.

## Physical capability space v1

Actor observation is not capability metric space. FIFO history, last action and validity bits do not define capability cells.

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

- `root_geometry_v1`: primary geometry/frontier diversity;
- `full_physical_v1`: fine physical deduplication.

Implementation:

```text
jit_dvgc.analysis.capability_tube
JIT/cli/analyze_capability_tube.py
```

## Raw Tube versus Jump-Tube view

Historical raw counts remain immutable:

```text
Tube_0 222
Tube_1 3119
Tube_2 3776
```

They are replay/provenance counts only.

All-state physical occupancy already measured:

```text
Tube_0 root/full 100/112
Tube_1 root/full 2142/2404
Tube_2 root/full 2446/2871
```

Do not use those downstream counts as jump-envelope size until the task-semantic view is rebuilt.

Jump-Tube view implementation:

```text
jit_dvgc.analysis.jump_tube_view
JIT/cli/analyze_jump_tube_view.py
```

Filter:

```text
centerline x support only
upstream within corridor
downstream within corridor AND root_vz < 0
post-landing recovery excluded
```

Late recovery may remain in raw replay but cannot be frontier evidence or capability-growth accounting.

## Trajectory-centered frontier parents

Production implementation:

```text
jit_dvgc.acquisition.resolution_frontier
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The CLI requires `--nominal-centerline`.

Parent rule:

```text
newest raw Tube shell
+ parent-group unique
+ exact-state unique
+ root_geometry_v1-cell unique
+ centerline-supported x slice
+ inside actual jump corridor
+ downstream root_vz < 0
+ no post-landing/late recovery
```

Selection is **local by x slice**, not global across a phase. Rank weak frontier cells inside each x bin, then round-robin across bins so dense regions cannot consume the budget.

Role assignment remains pre-outcome:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE, repeat
```

## Historical chain and current position

```text
experts
-> Tube_0 / pi_0 / C^0
-> Tube_1 / pi_1 repair02
-> v3/v3b/v3c frontier evidence
-> C^1 64x64 engineering path
-> Tube_2 / pi_2
-> locked pi_1 vs pi_2
-> physical-resolution analysis
-> CURRENT: nominal centerline + filtered Jump-Tube_0/1/2 reconstruction
```

Current pi_2 evidence:

```text
pi_1 source panel 3115/3119
pi_2 source panel 3002/3119
upstream 423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
locked pi_1-negative frontier: 13/14, 3 successful parent groups
```

Interpretation: local capability progression plus severe upstream single-policy realization loss. pi_2 remains evidence, not next authority.

Do not train pi_3 yet.

## Continuation-model claim boundary

- `V_up/V_down`: bootstrap expert-conditioned continuation fields.
- `C_up^k/C_down^k`: exact-policy-conditioned continuation evidence for proposal/filtering.
- PPO critic is not a JIT continuation field.

Current C^1 remains:

```text
upstream 64x64 AUC 0.6903137789904502 -> original AUC>=0.70 formal FAIL, engineering-selected only
downstream 64x64 AUC 1.0 / recall 1.0 -> formal calibration PASS
```

Do not rewrite upstream as a formal pass.

## Data-role contract

- `TRAIN`: fit continuation + qualifying replay expansion;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development comparison only;
- final TEST/JCE/JEL untouched.

Parent groups remain disjoint across roles. Geometry/x-slice diversity supplements provenance isolation; it does not replace it.

## Automatic iteration status

Generic workflow already covers:

```text
frontier roles -> C^k -> raw Tube -> smoke/isolation -> baseline lock
-> candidate train/freeze -> locked paired evaluation -> capability progression -> selection
```

New trajectory-centered capabilities exist for:

```text
successful rollout -> centerline
physical Tube -> Jump-Tube view
centerline + source geometry -> x-balanced frontier plan revision
```

These have not yet been exercised in one complete prospective automatic round. Do not claim end-to-end trajectory-centered automation until the recorded DAG includes them.

## Repository policy

- modify/consolidate before adding new files;
- new source files only for durable capabilities;
- keep CLIs thin;
- preserve artifacts/provenance;
- deletion requires dependency closure + compile/import/targeted tests;
- no unrelated-tree cleanup.

## Current read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
