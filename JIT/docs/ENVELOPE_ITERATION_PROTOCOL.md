# JIT Resolution-Aware Capability-Tube Iteration Protocol

## Status — 2026-09-04

This document defines the active scientific contract after the completed engineering `pi_1 -> C^1 -> raw Tube_2 -> pi_2` round.

The current method is **target-free** and **resolution-aware**.

Two previous assumptions are retired:

1. raw Tube entry growth is not a capability-volume metric;
2. exact state SHA uniqueness is not sufficient geometric diversity.

Final TEST/JCE/JEL remains untouched.

---

## 1. Scientific objects

Use four distinct objects:

```text
F*  conceptual physical/task feasible set under the fixed robot/task
E_k cumulative empirical successful real-dynamics evidence
T_k resolution-aware sparse occupied capability cells
R(pi_k, T_k) single-policy realization coverage over cumulative support
```

JIT does not prove `F*`.

A Soft Tube artifact is still an exact replayable TRAIN dataset. The physical capability Tube is an analysis/coverage representation projected from those snapshots.

A later policy failure does not erase earlier successful capability evidence. A new exact snapshot does not automatically create a new capability cell.

---

## 2. Target-free method authority

Do not add jump intent to the current method.

The present research question is:

> Under fixed XML/task semantics, what state-space Tube of different physically meaningful jump states can be empirically supported and progressively explored?

At a fixed longitudinal `x`, multiple heights, velocities, attitudes and internal configurations may be valid. The Tube therefore has a cross-section.

A fixed desired jump distance/apex target would change the task and reward toward a goal-conditioned trajectory-family problem. That may be a later JIT version, but it is not part of this protocol.

Reward, Actor observation and success semantics remain unchanged in the current revision.

---

## 3. Immutable physical/task contract

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- simulation substep: 0.005 s
- control interval: 0.020 s = 50 Hz
- hip/knee actuator range: +/-30 N m
- action order: `[steer, rear-wheel drive, hip, knee]`
- no runtime expert switching
- no silent reward/action/snapshot/task-geometry changes
- no TEST/final evidence in development iterations

Older text that said +/-50 N m was documentation drift; the authoritative XML/runtime validator uses +/-30 N m.

---

## 4. Capability coordinate system

Actor observation space is not the physical capability metric space.

Controller-only history/action variables may influence policy decisions but do not define whether two physical robot states occupy a new capability region.

Capability coordinates v1 are derived from snapshot `qpos/qvel` and the authoritative model indices.

### 4.1 Root geometry coordinates

```text
root x, y, z
root vx, vy, vz
roll, pitch, yaw
root wx, wy, wz
phase
```

This profile is the primary geometry/frontier-diversity space.

### 4.2 Full physical coordinates

Root geometry plus:

```text
steering, hip, knee angles
steering, hip, knee rates
rear/front wheel tangential speeds
```

Wheel rotational angle/phase is intentionally excluded from cell identity; tangential speed is retained.

---

## 5. Resolution contract v1

```text
position                     0.10 m
linear velocity              0.10 m/s
orientation                  0.50 deg
root angular velocity        2.0 deg/s
joint angle                  0.50 deg
joint angular velocity       2.0 deg/s
wheel tangential velocity    0.10 m/s
phase                        discrete
```

Quantization method:

```text
nearest grid
round half away from zero
```

The 2 deg/s angular-velocity resolution is intentional. Do not revert to the earlier 5 deg/s proposal without a new method decision.

Resolution identities are self-hashed in analysis artifacts.

---

## 6. Raw Soft Tube versus physical capability Tube

Raw Soft Tube:

```text
exact snapshot
exact state SHA
exact provenance
sampling weight
continuation score
```

Physical Tube:

```text
snapshot -> physical coordinates -> quantized cell
```

Report both:

```text
raw_snapshot_count
unique_full_physical_cell_count
unique_root_geometry_cell_count
```

A Tube generation is allowed to contain multiple raw snapshots in one physical cell because replay density can still be useful for policy learning. But these duplicates do not count as independent capability expansion.

---

## 7. Expansion metric

For `Tube_(k+1)` relative to `Tube_k`, distinguish:

```text
raw snapshot growth
new full physical cells
new root geometry cells
new x slices
x-slice cross-section growth
nearest source-cell distance in resolution units
```

The primary macroscopic expansion measure is **new root-geometry cells**, not raw entries.

A new full-physical cell with no new root-geometry cell may represent useful internal posture diversity but must not be presented as macroscopic jumping-envelope growth.

No cell-count metric is a proof of continuous physical volume or viability.

---

## 8. Tube geometry and x slices

Longitudinal coordinate:

```text
root_x_m
```

Slice width:

```text
0.10 m
```

For each x slice, summarize unique root-geometry cells over:

```text
y, z
vx, vy, vz
roll, pitch, yaw
wx, wy, wz
```

Required outputs:

- occupied cell count;
- raw snapshot density;
- phase composition;
- min/q25/median/q75/max for cross-section coordinates;
- visual x-z, x-vx, x-pitch projections;
- visual x-z-vx 3D projection.

Do not force a smooth Tube shape. Branches, gaps, lobes and bottlenecks are valid evidence.

---

## 9. Frontier parent selection

Future parent diversity requires all of:

```text
newest source-Tube shell
unique parent group
unique exact state SHA
unique root_geometry_v1 cell
```

Implementation:

```text
jit_dvgc.acquisition.resolution_frontier
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Default maximum:

```text
25 distinct parent cells per phase
```

Minimum support before frontier execution:

```text
5 distinct newest-shell root-geometry cells per phase
```

Role pattern remains pre-outcome and deterministic:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE, repeat
```

Resolution-aware parent selection is not allowed to inspect continuation outcomes.

---

## 10. Frontier candidate acquisition

Candidates must still be generated only by authoritative real dynamics from valid parent snapshots.

No direct coordinate dilation or hand-edited qpos/qvel is allowed.

Resolution-aware parent diversity reduces redundant starting neighborhoods. Future acquisition should additionally report candidate-cell occupancy so the project can decide whether more trajectories are adding coverage or only density.

A future optimization may filter candidate duplicates before expensive labeling, but that optimization must preserve deterministic candidate/PRNG provenance and must be predeclared before outcomes.

---

## 11. Data-role isolation

Logical roles remain:

- `TRAIN`: continuation fitting and candidate Tube replay support;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development frontier comparison only;
- final TEST/JCE/JEL: untouched.

Parent-group disjointness remains mandatory.

Resolution-aware geometric cell separation is an additional diversity requirement; it does not replace logical-role isolation.

Historical Iteration-1 -> 2 near-observation engineering evidence remains historical and does not relax the prospective protocol.

---

## 12. Continuation authority

- `V_up/V_down`: bootstrap expert-conditioned continuation evidence.
- `C_up^k/C_down^k`: exact-policy-conditioned continuation evidence tied to frozen `pi_k`.
- PPO critic/value is not a JIT continuation field.

`C^k` is useful for proposal/filtering and empirical continuation ranking. It is not a proof that no other admissible controller could succeed.

Current C^1 remains mixed-claim:

```text
upstream AUC 0.6903137789904502 < 0.70 formal gate
engineering-selected

downstream AUC 1.0
formal calibration PASS
```

---

## 13. Raw Tube construction

The replay artifact remains structurally core-retaining:

```text
raw Tube_(k+1)
= every raw Tube_k snapshot retained exactly
+ qualifying logical-TRAIN replay snapshots
```

This is a reproducibility/training contract, not the capability metric.

After construction, the next mandatory scientific analysis is physical-cell projection and source-vs-target comparison.

---

## 14. Policy training and realization

Unified policy training remains one Actor, no expert switching.

Current mainline training configuration used:

```text
90% Tube reset
10% natural reset
inside Tube:
75% retained source Tube
25% newest raw expansion
```

This is historical/current training evidence, not a permanent solution to all future realization problems.

Do not automatically sweep replay ratios after a failed policy-realization result.

---

## 15. Candidate evaluation

Prospective evaluation separates:

### A. Frontier progression

Require:

- locked baseline identity;
- no boundary baseline-reproduction mismatch;
- nonzero new frontier successes;
- sufficient independent parent groups;
- successes in both phases.

### B. Policy realization retention

Current prospective engineering non-inferiority margins:

```text
max global locked-panel coverage drop = 5 percentage points
max per-phase coverage drop           = 10 percentage points
```

Zero strict paired regressions are not required.

These margins evaluate the next single-policy authority; they do not define physical feasibility.

---

## 16. Current pi_2 interpretation

Locked source panel:

```text
pi_1 3115/3119
pi_2 3002/3119
upstream 423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
```

Locked pi_1-negative frontier:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
3 successful parent groups
0 baseline reproduction failures
```

Therefore:

```text
local frontier progression: strong
upstream single-policy realization: degraded
pi_2 as empirical capability evidence: retained
pi_2 as retrospectively selected next authority: no
```

Before interpreting how large the discovered envelope became, remeasure Tube_0/1/2 in physical cell space.

---

## 17. Automatic workflow maturity

Existing generic orchestration already supports:

```text
frontier roles
-> continuation fields
-> raw Tube construction
-> smoke/isolation
-> baseline lock
-> train/freeze candidate
-> locked paired evaluation
-> capability progression
-> prospective selection
```

New resolution-aware capabilities are implemented but not yet demonstrated in a complete prospective automatic run.

Required pre-frontier vNext operator sequence:

```text
analyze source Tube physical geometry
-> prepare ordinary outcome-blind frontier plan
-> revise parent set by distinct root-geometry cells
-> run frontier roles
```

Only after a workflow artifact contains these stages may the project claim fully integrated resolution-aware automation.

---

## 18. Immediate retrospective analysis

Without new simulation, analyze:

```text
Tube_0
Tube_1 versus Tube_0
Tube_2 versus Tube_1
```

Primary questions:

1. How many raw snapshots collapse into the same physical cells?
2. How many genuinely new root-geometry cells were gained per round?
3. At what x slices did the Tube widen or extend?
4. Is upstream support continuous, narrow, branched or multi-lobed?
5. Are current upstream realization regressions concentrated around one branch or region?

No pi_3 training starts before this evidence is reviewed.

---

## 19. Stopping and final JCE/JEL

Future stopping criteria should eventually use resolution-aware evidence, for example:

- negligible new root-geometry cells;
- no new x-slice extension;
- cross-section saturation;
- repeated inability to extend geometry without unacceptable realization loss;
- resource budget.

Only after method/stopping policy and final policy are frozen should untouched final TEST/JCE/JEL be used.

The final claim remains empirical, not a formal reachability/safety certificate.
