# JIT agent and maintenance rules

## Scope and safety

- `JIT/` is the active implementation area. Treat repository-root `dvgc/`, `cli/`, `scripts/`, and `tests/` as read-only unless the user explicitly changes scope.
- Work only on `agent/two-phase-soft-tube` unless explicitly told otherwise.
- Never reset, clean, stash, rebase, force-push, overwrite, or reformat unrelated user work.
- Use `/home/qy/mujoco_playground/.venv/bin/python`; do not reinstall or reconfigure the environment.
- Fixed task identity: `assets/orange_bike_4kg_horizontal.xml`, SHA `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`, 2 kg payload, 50 Hz control, hip/knee actuator range +/-30 N m, action order `[steer, rear-wheel drive, hip, knee]`.
- Unified runtime policies never switch experts.
- Final TEST/JCE/JEL stays untouched until method, stopping rule, and final policy are frozen.

## Scientific contract — target-free, resolution-aware JIT

JIT distinguishes:

- `F*`: conceptual physical/task feasibility under the fixed robot/task; not proved by JIT;
- `E_k`: cumulative empirical successful real-dynamics evidence;
- raw Soft Tube snapshots: exact replayable TRAIN support;
- resolution-aware physical Tube cells: sparse physical coverage summary;
- `R(pi_k, T_k)`: how much of cumulative support a single unified policy realizes.

The newest policy is a **capability probe + single-policy realization candidate**, not the definition of physical feasibility.

A later failed rollout does not erase an earlier successful capability observation. Conversely, merely adding many exact snapshots does not prove capability-envelope growth.

## Intent decision

Do not add goal/jump-intent conditioning to the current mainline.

The current research object is a target-free capability Tube: at the same longitudinal location, multiple heights, velocities, attitudes and internal states may all be valid continuation states. Adding a fixed desired distance/apex target now would change reward semantics and risk collapsing the study toward a narrower goal-conditioned trajectory family.

Goal-conditioned JIT is deferred as a separate future method version, not part of the present Tube reconstruction.

## Physical capability space v1

Actor observation is not the capability metric space. Controller-only variables such as FIFO history, last action and validity bits are excluded from capability-cell identity.

Declared resolutions:

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

Two profiles are authoritative:

1. `full_physical_v1` — root pose/twist + steering/hip/knee pose/rates + wheel tangential speeds. Use for fine physical deduplication.
2. `root_geometry_v1` — root pose/twist only. Use for macroscopic Tube geometry, x-slice coverage and frontier parent diversity.

This split prevents a tiny internal-joint difference from being reported as a new macroscopic jumping-envelope region.

Implementation:

- `jit_dvgc.analysis.capability_tube`
- `JIT/cli/analyze_capability_tube.py`

## Tube count semantics

Historical counts remain exact artifact counts:

```text
Tube_0 raw snapshots =   222
Tube_1 raw snapshots = 3,119
Tube_2 raw snapshots = 3,776
```

Do not convert these directly into capability-growth percentages.

The new primary retrospective questions are:

```text
How many unique full physical cells exist?
How many unique root geometry cells exist?
How many raw snapshots collapse into already occupied cells?
How many genuinely new root geometry cells did Tube_1 add over Tube_0?
How many genuinely new root geometry cells did Tube_2 add over Tube_1?
At which x slices did the cross-section actually grow?
How far are new cells from source cells in resolution units?
```

Until these are computed locally, Tube_1/Tube_2 cardinality growth is only replay-support growth.

## Tube geometry / visualization

Longitudinal progress coordinate: `root_x_m`, 0.10 m slices.

Each x slice summarizes unique root-geometry cells over:

- y/z;
- vx/vy/vz;
- roll/pitch/yaw;
- wx/wy/wz.

Generated geometry artifacts:

- `summary.json`
- `entries.json`
- `cells.json`
- `x_slices.json`
- `projected_points.csv`
- x-z / x-vx / x-pitch / x-z-vx plots when matplotlib is available.

Never fit a smooth cylinder merely to make the result look like a Tube. Narrowing, branches, disconnected lobes and gaps are meaningful outcomes.

## Resolution-aware frontier parents

Future frontier parent diversity is not satisfied by exact SHA uniqueness alone.

Production capability:

- `jit_dvgc.acquisition.resolution_frontier`
- `JIT/cli/prepare_resolution_aware_frontier_plan.py`

Parent rule:

```text
newest source-Tube shell
+ parent-group unique
+ exact-state unique
+ root_geometry_v1-cell unique
```

Default limit: up to 25 distinct parent cells per phase. Require at least 5 distinct newest-shell root-geometry cells per phase before frontier execution.

Role assignment remains outcome-blind before acquisition/label outcomes:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE, repeat
```

Resolution-aware cell separation supplements parent-group isolation; it does not replace provenance isolation.

## Current completed chain

```text
experts
  -> Tube_0 raw 222
  -> pi_0
  -> C^0
  -> Tube_1 raw 3,119
  -> pi_1 repair02 engineering authority
  -> v3/v3b/v3c frontier evidence
  -> C^1 64x64 engineering selection
  -> Tube_2 raw 3,776
  -> pi_2 trained/frozen
  -> locked pi_1 vs pi_2 comparison
  -> capability-progression analysis
  -> CURRENT: retrospective physical Tube reconstruction and resolution-aware frontier redesign
```

Current pi_2 evidence:

```text
source panel:
  pi_1 3115/3119
  pi_2 3002/3119

upstream:
  423/427 -> 312/427

downstream:
  2692/2692 -> 2690/2692

locked pi_1-negative frontier:
  pi_2 13/14
  upstream 4/5
  downstream 9/9
  3 successful parent groups
  baseline reproduction failures 0
```

Interpretation: strong local frontier progression, severe upstream realization loss, pi_2 retained as capability evidence but not retrospectively promoted.

## Continuation model claim boundary

- `V_up/V_down`: bootstrap expert-conditioned continuation fields.
- `C_up^k/C_down^k`: exact-policy-conditioned continuation evidence, useful for proposal/filtering, not existential controllability.
- PPO critic/value is not a JIT continuation field.

Current C^1:

- upstream 64x64 AUC `0.6903137789904502`, below original 0.70 formal gate; engineering-selected only;
- downstream 64x64 AUC 1.0, recall 1.0, formal calibration PASS.

Do not rewrite the upstream result as a formal pass.

## Data-role contract

- `TRAIN`: may fit continuation fields and contribute qualifying replay snapshots;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: development frontier comparison only;
- final TEST/JCE/JEL: untouched.

Parent groups remain disjoint across logical roles.

Historical Iteration-1 -> 2 near-observation evidence remains explicit:

```text
exact overlap across roles = 0
TRAIN <-> ACCEPTANCE near overlap at atol 0.01 = 0
TRAIN <-> CALIBRATION near overlap = 140
CALIBRATION <-> ACCEPTANCE near overlap = 157
```

That historical engineering continuation is not a general relaxation rule.

## Automatic iteration status

The existing workflow automates:

```text
frontier roles
-> C^k
-> raw Tube_(k+1)
-> smoke/isolation
-> baseline lock
-> train/freeze candidate
-> locked paired evaluation
-> capability-progression decision
-> prospective policy selection
```

The newly implemented capability-resolution layer is reusable but has not yet been executed as part of a completed prospective automatic round.

Before another iteration, use:

```text
analyze source Tube physical cells
-> create ordinary outcome-blind frontier plan
-> revise it with distinct root-geometry-cell parents
-> run frontier roles
```

Do not claim the generic workflow is fully resolution-aware until its recorded DAG includes these stages.

Do not launch pi_3 or a replay sweep before Tube_0/1/2 are re-measured in physical cell space.

## Repository policy

- modify/consolidate first;
- new source files only for durable capabilities;
- keep CLIs thin;
- preserve artifact paths/provenance;
- deletion requires dependency closure + compile/import/targeted tests;
- no unrelated-tree cleanup.

## Current read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
