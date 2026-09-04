# DVGC Repository Instructions

## Current research truth — 2026-09-04

DVGC/JIT is an iterative **target-free real-dynamics capability-discovery and just-in-time curriculum** project for one fixed single-track two-wheeled robot task.

The research now separates four objects:

1. **Physical/task feasibility `F*`** — the unknown set of states from which some admissible control behavior could complete the fixed task. JIT does not prove or exactly compute it.
2. **Cumulative empirical capability evidence `E_k`** — successful real-dynamics evidence accumulated across frozen experts and unified policies.
3. **Resolution-aware capability Tube `T_k`** — sparse occupied physical cells used to summarize and visualize empirical support; raw snapshot count is not a capability-volume metric.
4. **Single-policy realization coverage** — how much of cumulative support one unified Actor realizes on a locked panel.

The runtime target remains **one unified Actor** with no expert switching. Phase experts and frozen intermediate policies are discovery probes/data sources only.

A Soft Tube remains TRAIN curriculum/replay support, not a certified safe set, viability kernel, reachability proof, invariant set, or proof of the physical jump limit.

## Current scientific chain

```text
pi_up_star + pi_down_star
  -> bootstrap V_up / V_down
  -> raw Tube_0 snapshots = 222
  -> pi_0
  -> C^0
  -> raw Tube_1 snapshots = 3,119
  -> pi_1 repair02 engineering authority
  -> C^1 engineering path
  -> raw Tube_2 snapshots = 3,776
  -> pi_2 trained/frozen
  -> locked pi_1 vs pi_2 evaluation
  -> capability-progression reinterpretation
  -> CURRENT: resolution-aware physical Tube reconstruction for Tube_0/1/2
```

Do **not** call `222 -> 3,119 -> 3,776` a 14x/17x capability-envelope expansion. Those are raw replayable snapshot counts only. The new primary coverage quantities are unique physical/root-geometry cells and their x-slice geometry.

Final TEST/JCE/JEL remains untouched.

## Target-free policy contract

Do not add goal/intent conditioning to the current JIT mainline.

The user explicitly chose to keep the current method target-free because a fixed desired jump target would change the research object toward goal-conditioned trajectory families. No reward, Actor observation, or task semantics are changed in the current resolution-aware revision.

Goal-conditioned JIT may be studied later as a separate method version only after the target-free Tube geometry is understood.

## Capability-state resolution contract v1

Actor observation space is **not** the physical capability metric space. FIFO history, last action, acceleration and validity bits remain controller inputs but do not define capability-cell identity.

The first declared physical resolution is:

| Quantity | Resolution |
|---|---:|
| root x/y/z | 0.10 m |
| root vx/vy/vz | 0.10 m/s |
| roll/pitch/yaw | 0.50 deg |
| root angular velocity wx/wy/wz | 2.0 deg/s |
| steering/hip/knee angle | 0.50 deg |
| steering/hip/knee rate | 2.0 deg/s |
| front/rear wheel tangential speed | 0.10 m/s |
| phase | discrete upstream/downstream |

Implementation:

- `JIT/src/jit_dvgc/analysis/capability_tube.py`
- `JIT/cli/analyze_capability_tube.py`

The code emits two cell profiles:

- `full_physical_v1`: root pose/twist + joint pose/rates + wheel tangential speeds; used for fine physical deduplication;
- `root_geometry_v1`: root pose/twist only; used as the primary geometric Tube/frontier diversity space so a joint-only difference cannot masquerade as macroscopic envelope expansion.

Quantization is nearest-grid, half away from zero. Angular velocity resolution is **2 deg/s**.

## Tube geometry contract

For each Tube, capability analysis must report at least:

```text
raw_snapshot_count
unique_full_physical_cell_count
unique_root_geometry_cell_count
root/full duplicate fractions
x-slice count
x-slice cross-section statistics
phase-specific cell counts
```

When a source Tube is provided, also report:

```text
new_root_geometry_cell_count
new_full_physical_cell_count
new cells by phase
new root cells per raw added snapshot
nearest source-cell distance in declared resolution units
```

The longitudinal progress coordinate is root `x`, resolved at 0.10 m. Each x slice summarizes cross-section support over y/z, linear velocity, attitude and angular velocity. Visualizations must be based on unique root-geometry cells rather than raw snapshot density.

The implementation generates:

- `summary.json`
- `entries.json`
- `cells.json`
- `x_slices.json`
- `projected_points.csv`
- x-z, x-vx, x-pitch and x-z-vx plots when matplotlib is available.

Do not force the data to look like a smooth cylinder. A valid empirical Tube may narrow, branch, contain multiple lobes or expose gaps; those shapes are scientific evidence.

## Resolution-aware frontier rule

Exact state SHA uniqueness is no longer sufficient evidence of frontier diversity.

Before future frontier outcomes are collected, a frontier plan may be revised by:

- `JIT/src/jit_dvgc/acquisition/resolution_frontier.py`
- `JIT/cli/prepare_resolution_aware_frontier_plan.py`

The revised parent rule is:

```text
newest Tube shell only
+ distinct parent_group_id
+ distinct exact state SHA
+ distinct root_geometry_v1 cell
```

Default maximum is 25 distinct parent cells per phase. At least 5 distinct newest-shell root-geometry cells per phase are required. The role pattern remains outcome-blind `TRAIN/TRAIN/TRAIN/CALIBRATION/ACCEPTANCE`.

This is a pre-outcome protocol revision only; it does not inspect labels and does not change physics, reward, policy, probe outcomes or TEST.

## Current pi_2 evidence

The current pi_2 candidate completed 10,009,600 transitions.

Locked source-Tube panel:

```text
pi_1 baseline: 3115/3119
pi_2:          3002/3119
strict regressions: 115

upstream:   423/427 -> 312/427
downstream: 2692/2692 -> 2690/2692
```

Locked pi_1-negative frontier challenge:

```text
pi_2 success: 13/14
successful parent groups: 3
upstream: 4/5
downstream: 9/9
baseline reproduction failures: 0
```

Interpretation:

- strong local frontier progression evidence;
- severe upstream single-policy realization loss;
- pi_2 remains capability evidence;
- pi_2 is not retrospectively promoted to the next formal authority.

The next task is **not** a 90/10 replay sweep and **not** pi_3 training. First reconstruct Tube_0/1/2 under the physical resolution contract and determine how much independent state-space coverage actually grew.

## Current automation status

The existing generic k -> k+1 workflow already automates frontier roles, continuation fitting, Tube construction, smoke, isolation, locked baseline, training/freeze, paired evaluation, capability progression and prospective selection.

The new resolution-aware layer is implemented as reusable production capabilities but is **not yet evidence from a completed new iteration**. Before launching another automatic round, the operator must insert the following pre-frontier steps:

```text
analyze source Tube capability geometry
-> prepare ordinary outcome-blind frontier plan
-> revise that plan to distinct root-geometry-cell parents
-> only then run frontier roles
```

Do not describe the automatic workflow as fully resolution-aware until a future workflow artifact explicitly contains those stages.

## Immutable physical/task contracts

- branch: `agent/two-phase-soft-tube`;
- XML: `assets/orange_bike_4kg_horizontal.xml`;
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`;
- actual payload: 2 kg;
- simulation runtime: 0.005 s substep, 0.020 s control interval = 50 Hz;
- hip/knee actuator force/torque range in the authoritative XML: **+/-30 N m**;
- action order: `[steer, rear-wheel drive, hip, knee]`;
- no runtime expert switching;
- no silent physics/reward/action/snapshot/task-geometry/TEST changes.

The XML and runtime model validator both enforce +/-30 N m. Documentation corrections do not modify the robot model.

## Data-role contract

- `TRAIN`: may fit continuation fields and contribute qualifying replay snapshots;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: development frontier comparison only;
- final TEST/JCE/JEL: untouched.

Parent-group disjointness remains mandatory. Resolution-aware cell separation supplements it; it does not replace provenance isolation.

## Repository/Git safety

- preserve unrelated work;
- never reset, clean, stash, rebase or force-push;
- use `/home/qy/mujoco_playground/.venv/bin/python`;
- keep CLIs thin and reusable logic under `JIT/src/jit_dvgc/`;
- run compile/tests after structural changes before deleting anything.

## Current authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
