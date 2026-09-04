# JIT Trajectory-Centered Jump-Tube Iteration Protocol

## Status — 2026-09-04

This is the active protocol after the completed engineering `pi_1 -> C^1 -> raw Tube_2 -> pi_2` round and the subsequent physical Tube visualization review.

The current method is:

> **trajectory-centered, resolution-aware, real-dynamics Jump-Tube identification with just-in-time curriculum generation.**

Retired assumptions:

1. raw Tube entry growth is not capability-volume growth;
2. exact state SHA uniqueness is not geometric diversity;
3. a global low-score newest-shell pool is not a sufficient jump-frontier definition;
4. all states labelled `downstream` are not automatically valid descending Jump-Tube states.

Final TEST/JCE/JEL remains untouched.

---

## 1. Scientific objects

```text
F*  conceptual physical/task feasible set under fixed robot/task
E_k cumulative successful real-dynamics evidence
J_k trajectory-centered empirical Jump-Tube support
R(pi_k, J_k) single-policy realization coverage
```

JIT does not prove `F*`.

A raw Soft Tube is an exact replayable TRAIN artifact. A Jump-Tube view is a task-semantic capability representation projected from those snapshots.

A later policy failure does not erase prior successful capability evidence. A new exact snapshot does not automatically create a new physical cell or a new Jump-Tube region.

---

## 2. No goal intent in the current mainline

Do not add desired jump distance/apex intent to the current Actor.

The centerline introduced below is **not** a goal input and does not alter reward/task semantics. It is a geometric scaffold used to index the state-space Tube longitudinally.

At a fixed x slice, multiple physically different successful continuation states may coexist. The Tube therefore keeps nonzero cross-section.

Goal-conditioned JIT remains a possible later method version only.

---

## 3. Immutable physical/task contract

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- simulation substep: 0.005 s
- control interval: 0.020 s = 50 Hz
- hip/knee actuator range: +/-30 N m
- action order: `[steer, rear-wheel drive, hip, knee]`
- one unified runtime Actor
- no runtime expert switching
- no silent reward/action/XML/task-geometry changes
- no final TEST/JCE/JEL use during development iterations

---

## 4. Nominal centerline authority

Every prospective trajectory-centered iteration must lock one successful full-chain real rollout before frontier outcomes.

Centerline v1:

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x step          = 0.1 m
actual terminal = first valid landing if earlier
```

Construction rules:

```text
real captured simulator frames only
nearest real frame per target x slice
maximum x mismatch <= 0.05 m
no qpos/qvel interpolation
```

Branch semantics:

```text
pre-Apex            upstream
Apex-near           apex marker
post-Apex + vz < 0  downstream
first valid landing terminal
post-landing         excluded from Jump-Tube frontier
```

Implementation:

```text
jit_dvgc.analysis.nominal_jump_centerline
JIT/cli/build_nominal_jump_centerline.py
```

If the canonical natural rollout is not full-recovery successful, the centerline builder must stop. Another successful real rollout may only be substituted through a new predeclared source; never fabricate or interpolate a path.

---

## 5. Physical capability coordinate system

Actor observation space is not the physical capability metric space.

Capability coordinates are derived from authoritative snapshot `qpos/qvel` and model indices.

### Root geometry profile

```text
root x/y/z
root vx/vy/vz
roll/pitch/yaw
root wx/wy/wz
phase
```

### Full physical profile

Root geometry plus:

```text
steering/hip/knee angles
steering/hip/knee rates
front/rear wheel tangential speeds
```

Wheel angle phase itself is excluded from cell identity.

---

## 6. Resolution contract v1

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

Quantization:

```text
nearest grid
round half away from zero
```

The 2 deg/s angular-velocity resolution is deliberate.

Implementation:

```text
jit_dvgc.analysis.capability_tube
JIT/cli/analyze_capability_tube.py
```

---

## 7. Raw/Control Tube versus Jump-Tube view

Raw Soft Tube:

```text
exact restartable snapshot
state SHA
provenance
continuation score
sampling weight
```

Jump-Tube view:

```text
raw Tube
-> physical projection
-> centerline corridor filter
-> downstream descending filter
-> post-landing exclusion
-> quantized occupied cells
```

Implementation:

```text
jit_dvgc.analysis.jump_tube_view
JIT/cli/analyze_jump_tube_view.py
```

Historical late recovery states are retained in raw replay/provenance but cannot contribute to Jump-Tube capability growth or frontier-parent eligibility.

---

## 8. Jump-Tube geometry

Longitudinal coordinate:

```text
root_x_m
```

Slice width:

```text
0.10 m
```

For every centerline-supported x slice, summarize the unique root-geometry-cell cross-section over:

```text
y/z
vx/vy/vz
roll/pitch/yaw
wx/wy/wz
```

The Tube is not forced into a cylinder. Narrowing, widening, branches, lobes and gaps are valid evidence.

Primary progression question:

> At which x slices did successful physically resolved cross-section support widen?

---

## 9. Frontier parent selection

Future parent eligibility requires all of:

```text
newest source raw-Tube shell
unique parent group
unique exact state SHA
unique root_geometry_v1 cell
x slice supported by the locked nominal centerline
inside the actual jump corridor
downstream root_vz < 0
no post-landing / late-recovery state
```

Implementation:

```text
jit_dvgc.acquisition.resolution_frontier
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The CLI requires `--nominal-centerline`.

Default maximum:

```text
25 distinct parent cells per phase
```

Minimum after all semantic filters:

```text
5 distinct newest-shell root-geometry cells per phase
```

### Local x-balanced selection

Do not globally rank an entire phase and take the lowest scores.

New rule:

```text
partition eligible parents by x slice
rank weak continuation frontier cells locally in each slice
round-robin across slices
only after every occupied slice gets an opportunity may a slice contribute again
```

This prevents a dense downstream/recovery region from monopolizing acquisition budget.

Role pattern remains deterministic and pre-outcome:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE, repeat
```

---

## 10. Frontier candidate acquisition

Candidates remain real-dynamics only.

Allowed:

```text
reset exact valid replay state
compute deterministic frozen pi_k action
apply bounded declared action perturbation
advance only through authoritative env.step
```

Forbidden:

```text
hand-edited qpos/qvel
direct coordinate dilation
outcome-dependent parent reselection
TEST-informed acquisition
```

Existing sparse single-/two-axis action perturbation machinery may be reused, but all generated candidates must remain compatible with the trajectory-centered task semantics before they contribute to Jump-Tube claims.

---

## 11. Data-role isolation

Logical roles:

- `TRAIN`: continuation fitting and qualifying replay support;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development frontier comparison only;
- final TEST/JCE/JEL: untouched.

Parent-group disjointness remains mandatory.

Physical cell / x-slice diversity supplements provenance isolation; it does not replace it.

---

## 12. Continuation authority

- `V_up/V_down`: bootstrap expert-conditioned continuation evidence.
- `C_up^k/C_down^k`: frozen-policy-conditioned continuation evidence.
- PPO critic/value is not a JIT continuation field.

`C^k` is an empirical proposal/filtering model, not proof of existential controllability.

Current C^1 historical truth remains:

```text
upstream AUC 0.6903137789904502 < original 0.70 gate
engineering-selected only

downstream AUC 1.0
formal calibration PASS
```

---

## 13. Raw Tube construction

Raw replay artifact remains core-retaining:

```text
raw Tube_(k+1)
= every raw Tube_k snapshot retained exactly
+ qualifying logical-TRAIN replay snapshots
```

This preserves training/provenance continuity but is not the Jump-Tube expansion metric.

After construction, two analyses are mandatory **before candidate policy training**:

```text
1. physical capability geometry of raw Tube_(k+1)
2. filtered Jump-Tube_(k+1) view relative to source Jump-Tube_k
```

If filtered Jump-Tube root-cell growth is zero, the automatic workflow must stop rather than training merely because raw rows increased.

---

## 14. Policy training

Unified policy training remains one Actor with no expert switching.

Historical/current reset mixture:

```text
90% Tube reset
10% natural reset
inside Tube:
75% retained source Tube
25% newest raw expansion
```

This remains an engineering training configuration, not a permanent theorem or automatic repair rule.

Do not sweep replay ratios automatically after realization failure.

---

## 15. Candidate evaluation

Prospective evaluation separates:

### A. Empirical frontier progression

Require:

- locked baseline identity;
- zero boundary baseline-reproduction mismatch;
- nonzero new frontier successes;
- sufficient independent parent groups;
- phase-aware support.

### B. Single-policy realization

Current prospective engineering non-inferiority margins remain:

```text
max global locked-panel coverage drop = 5 percentage points
max per-phase coverage drop           = 10 percentage points
```

Zero strict paired regressions are no longer the sole method objective.

These margins govern promotion of the next single-policy authority, not physical feasibility.

---

## 16. Automatic workflow vNext

`JIT/cli/prepare_iterative_envelope_workflow.py` now requires:

```text
--canonical-evaluation-report
```

The generated DAG includes:

```text
lock nominal centerline
-> analyze source physical geometry
-> build source Jump-Tube view
-> prepare ordinary outcome-blind frontier plan
-> revise to trajectory-centered x-balanced plan
-> frontier TRAIN/CALIBRATION/ACCEPTANCE
-> C^k
-> raw Tube_(k+1)
-> analyze target physical geometry
-> build target Jump-Tube view
-> REQUIRE new filtered Jump-Tube root cells > 0
-> smoke/isolation
-> baseline lock
-> candidate train/freeze
-> locked evaluation
-> capability progression
-> prospective policy selection
```

This is the intended prospective automatic method. It is implemented but not yet validated by a complete new run.

---

## 17. Current pi_2 historical interpretation

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
pi_2 as capability evidence: retained
pi_2 as next authority: no
```

Historical all-state Tube counts must now be filtered into Jump-Tube views before they are used for capability-size claims.

---

## 18. Current operator sequence

Before any new candidate policy:

```text
1. obtain successful canonical pi_1 natural rollout
2. lock nominal centerline
3. rebuild Jump-Tube_0/1/2 semantic views
4. inspect removed late downstream/recovery support
5. inspect per-x cross-sections and gaps
6. only then prepare/run the next prospective trajectory-centered workflow
```

No pi_3-like training starts before this evidence is reviewed.

---

## 19. Stopping and final JCE/JEL

Future stopping should use trajectory-centered evidence, including:

- negligible new Jump-Tube root cells;
- no new x-slice support;
- cross-section saturation;
- repeated inability to widen sparse slices;
- unacceptable realization loss;
- resource budget.

Only after method, stopping rule and final policy are frozen may untouched final TEST/JCE/JEL be used.

Final claims remain empirical rather than formal reachability/safety certification.
