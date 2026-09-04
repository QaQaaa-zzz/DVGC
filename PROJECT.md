# DVGC Project

## 1. Scientific objective

DVGC/JIT studies **iterative real-dynamics jump-capability discovery with just-in-time curriculum generation** for one fixed single-track two-wheeled robot.

The final deployment target is one unified Actor. Phase experts and frozen intermediate policies are discovery probes/data sources only.

The project separates:

```text
F*   unknown physical/task feasibility under fixed dynamics
E_k  cumulative successful real-dynamics evidence
J_k  trajectory-centered empirical Jump-Tube support
R_k  realization coverage of one unified Actor
```

JIT does not prove `F*`, a viability kernel, invariant set, or certified safe set.

---

## 2. Current method: trajectory-centered, not goal-conditioned

The current mainline does **not** add desired jump distance/apex intent to the Actor and does not change reward semantics.

Instead, JIT first locks one successful real full-chain jump trajectory and uses it as a geometric centerline. The method then widens the physically valid state cross-section around that trajectory at each longitudinal x slice.

This preserves the desired research object:

> a nonzero-thickness state-space Jump Tube around a real jump, rather than one fixed controller target trajectory.

---

## 3. Nominal Jump-Tube geometry

Centerline contract v1:

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x spacing       = 0.1 m
actual terminal = first valid landing if earlier
```

Every centerline point is a real captured simulator frame from one successful canonical natural rollout. No qpos/qvel interpolation is allowed.

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
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

---

## 4. Physical capability coordinate system

Actor observation is not the capability metric space. FIFO history, last action, acceleration and validity flags remain controller inputs but do not define whether two physical states are meaningfully distinct.

Resolution contract v1:

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

- `root_geometry_v1`: root pose/twist, primary macroscopic Tube/frontier geometry;
- `full_physical_v1`: root geometry + joint pose/rates + wheel tangential speed, fine deduplication.

Implementation:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

---

## 5. Raw/Control Tube versus Jump Capability Tube

A historical Soft Tube is an immutable replay/provenance artifact. It may contain late landing/recovery states because a unified controller must still finish the task.

A Jump-Tube view is the task-semantic subset used to measure jump-boundary support.

Jump-Tube filter:

```text
x on nominal centerline support
upstream inside corridor
downstream inside corridor AND root_vz < 0
post-landing recovery excluded
```

Implementation:

```text
JIT/src/jit_dvgc/analysis/jump_tube_view.py
JIT/cli/analyze_jump_tube_view.py
```

This separation prevents late recovery states from inflating jump-capability expansion metrics without deleting useful replay history.

---

## 6. Core JIT loop

Prospective mainline:

```text
selected pi_k
+ source raw/control Tube_k
+ locked successful nominal centerline
        ↓
physical projection / resolved cells
        ↓
Jump-Tube_k semantic view
        ↓
local frontier selection per 0.1 m x slice
        ↓
real-dynamics action perturbation
        ↓
TRAIN / CALIBRATION / ACCEPTANCE continuation evidence
        ↓
C_up^k / C_down^k
        ↓
qualifying replay expansion
        ↓
raw/control Tube_(k+1)
        ↓
Jump-Tube_(k+1) reconstruction
        ↓
measure cross-section widening by x slice
        ↓
train one unified pi_(k+1)
        ↓
locked frontier progression + policy realization evaluation
```

The curriculum is generated from current local frontier support, not from a manually fixed easy-to-hard schedule.

---

## 7. Trajectory-centered frontier selection

Implementation:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The CLI requires a nominal centerline.

Parent eligibility:

```text
newest raw Tube shell
+ unique parent group
+ unique exact state
+ unique root_geometry_v1 cell
+ x slice supported by nominal centerline
+ inside actual jump corridor
+ downstream root_vz < 0
+ no post-landing / late-recovery state
```

Selection is no longer global by lowest continuation score.

New logic:

```text
partition by x slice
rank weak frontier cells locally
round-robin across slices
```

This ensures that one dense region cannot monopolize acquisition budget.

Role assignment remains outcome-blind:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE, repeat
```

---

## 8. Historical completed chain

### Experts

```text
pi_up_star
  9,977,856 transitions
  actor f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081

pi_down_star
  25,600 transitions
  actor 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

### Raw Tube_0

```text
222 raw snapshots
117 upstream
105 downstream
```

Retrospective all-state physical occupancy:

```text
100 root cells
112 full cells
13 occupied x slices
```

### pi_0

```text
10,009,600 transitions
actor 43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029
```

### Raw Tube_1

```text
3,119 raw snapshots
= 222 retained Tube_0 + 2,897 raw expansion
```

All-state physical occupancy:

```text
2,142 root cells
2,404 full cells
2,042 new root cells vs Tube_0
2,292 new full cells vs Tube_0
24 occupied x slices
```

These counts include historical late downstream/recovery support and are now under Jump-Tube re-audit.

### pi_1 repair02

Historical engineering quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 parent groups
```

Historical formal PASS is not claimed because three old baseline-reproduction mismatches remain quarantined.

### C^1

```text
upstream 64x64:
  AUC 0.6903137789904502
  recall 0.5934515688949522
  original AUC>=0.70 gate FAIL
  engineering-selected only

downstream 64x64:
  AUC 1.0
  recall 1.0
  formal calibration PASS
```

### Raw Tube_2

```text
3,776 raw snapshots
= 3,119 retained Tube_1 + 657 raw expansion
```

All-state physical occupancy:

```text
2,446 root cells
2,871 full cells
304 new root cells vs Tube_1
467 new full cells vs Tube_1
24 occupied x slices
```

Again, these are all-state control-Tube counts, not final Jump-Tube expansion counts.

### pi_2

Training completed at 10,009,600 transitions.

Locked source panel:

```text
pi_1 3115/3119
pi_2 3002/3119

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

Interpretation: clear local capability progression plus substantial upstream single-policy realization degradation. pi_2 remains empirical capability evidence and is not retrospectively selected as the next authority.

---

## 9. Why raw snapshot growth is not capability growth

Historical cardinalities:

```text
Tube_0 222
Tube_1 3119
Tube_2 3776
```

These describe replay support only.

Primary future progression metrics:

```text
unique Jump-Tube root cells
unique Jump-Tube full cells
new cells relative to source
occupied x slices
new cells by x slice
cross-section change by physical coordinate
```

The most important question becomes:

> where along x did the successful physical cross-section widen?

---

## 10. Current project position

```text
experts                          DONE
Tube_0 / pi_0 / C^0             DONE
Tube_1 / pi_1                   DONE
C^1 engineering path            DONE
Tube_2 / pi_2                   DONE
locked pi_1 vs pi_2             DONE
physical resolution analysis    DONE
trajectory-centered code        IMPLEMENTED
nominal centerline artifact     NEXT local task
filtered Jump-Tube_0/1/2        NEXT local task
prospective x-balanced frontier NOT RUN
pi_3-like candidate             NOT AUTHORIZED
```

The next work is geometric reconstruction and semantic audit, not another PPO run.

---

## 11. Automatic iteration state

Generic automation already supports:

```text
frontier roles
-> C^k
-> raw Tube
-> smoke/isolation
-> baseline lock
-> candidate training/freeze
-> locked evaluation
-> capability progression
-> prospective selection
```

New trajectory-centered production capabilities now support:

```text
successful canonical rollout -> centerline
physical Tube -> Jump-Tube view
centerline + geometry -> x-balanced frontier plan revision
```

A full prospective trajectory-centered automatic round has not yet been executed. Do not claim end-to-end automatic trajectory-centered JIT until a recorded workflow DAG contains these stages.

---

## 12. Immediate next work

1. Run one successful canonical natural rollout using the selected pi_1 authority.
2. Build the nominal centerline from real captured frames.
3. Recompute Tube_0/1/2 as Jump-Tube views.
4. Quantify how many historical downstream cells are removed by the x/vz/landing semantics.
5. Inspect `x -> cross-section` growth and identify sparse slices/gaps.
6. Only then predeclare a new trajectory-centered frontier plan.
7. Inspect the x-bin distribution before any TRAIN/CALIBRATION/ACCEPTANCE run.
8. Keep pi_3 paused until the new geometry is validated.

Detailed commands and rationale:

`JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md`

---

## 13. Immutable task contract

- branch: `agent/two-phase-soft-tube`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee actuator range: +/-30 N m
- actions: `[steer, rear-wheel drive, hip, knee]`
- runtime: one unified Actor; no expert switching
- final TEST/JCE/JEL untouched

---

## 14. Authority documents

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_TRAJECTORY_CENTERED_JUMP_TUBE_REPORT_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
