# DVGC Project

## 1. Scientific objective

DVGC/JIT studies **iterative real-dynamics jump-capability discovery with just-in-time curriculum generation** for one fixed single-track two-wheeled robot task.

The project separates four scientific objects:

1. **Physical/task feasibility `F*`** — the unknown set of states from which some admissible control behavior could complete the fixed task. JIT does not prove or exactly compute this set.
2. **Cumulative empirical capability evidence `E_k`** — successful real-dynamics evidence accumulated across frozen experts and unified policies.
3. **Resolution-aware capability Tube `T_k`** — sparse occupied physical cells summarizing where empirical capability evidence/support exists.
4. **Single-policy realization coverage** — how much of cumulative support one unified Actor realizes on a locked evaluation panel.

The final runtime target remains **one unified policy**. Phase experts and frozen intermediate policies are discovery probes/data sources, not a runtime switching architecture.

A Soft Tube is empirical TRAIN/curriculum support. It is not a certified safe set, viability kernel, reachability proof, invariant set, or proof of the physical jump limit.

---

## 2. Why the Tube is target-free

The current JIT mainline intentionally does **not** add a desired jump-distance/apex intent to the Actor.

The research object is not one fixed target trajectory. At the same longitudinal location `x`, multiple combinations of height, velocity, attitude and internal configuration may all be valid continuation states. The desired object is therefore a **state-space capability Tube with nonzero cross-section**, not a narrow goal-conditioned trajectory family.

Adding intent now would require changing reward/task semantics and would open a different research question. Goal-conditioned JIT remains a possible later method version, but it is not part of the current resolution-aware reconstruction.

---

## 3. Core JIT loop

```text
frozen capability probe(s)
        ↓
real-dynamics frontier acquisition
        ↓
continuation evidence near success/failure transition
        ↓
cumulative empirical support
        ↓
resolution-aware physical Tube
        ↓
just-in-time replay curriculum
        ↓
train one unified policy
        ↓
locked evaluation
  A. did the empirical frontier move?
  B. does the unified policy retain enough prior coverage?
        ↓
repeat or open a new method decision
```

The curriculum is produced from the current frontier rather than a manually fixed easy-to-hard schedule.

---

## 4. Bootstrap and iterative chain

Bootstrap:

```text
Propulsion-Ascent expert pi_up
        +
Descent-Recovery expert pi_down
        ↓
freeze experts
        ↓
expert-conditioned V_up / V_down
        ↓
raw TRAIN Tube_0 snapshots
        ↓
unified pi_0
```

Iterative regime:

```text
selected pi_k + raw Tube_k
        ↓
physical projection + resolution-aware occupied cells
        ↓
outcome-blind newest-shell frontier plan
        ↓
resolution-aware distinct-cell parent revision
        ↓
TRAIN / CALIBRATION / ACCEPTANCE acquisition
        ↓
pi_k-conditioned continuation labels
        ↓
C_up^k / C_down^k
        ↓
raw Tube_(k+1) replay support
        ↓
physical Tube_(k+1) cell reconstruction
        ↓
train/freeze pi_(k+1)
        ↓
locked paired evaluation
        ↓
frontier progression + policy realization decision
```

Historical engineering overrides remain explicit and are never rewritten to look prospective.

---

## 5. Physical capability coordinate system

The Actor observation is **not** the capability metric space. FIFO history, last action, acceleration and history-valid flags are controller inputs, but do not define whether two physical states are meaningfully distinct.

Resolution contract v1:

| Quantity | Resolution |
|---|---:|
| root x/y/z | 0.10 m |
| root vx/vy/vz | 0.10 m/s |
| roll/pitch/yaw | 0.50 deg |
| root angular velocity | 2.0 deg/s |
| steering/hip/knee angle | 0.50 deg |
| steering/hip/knee rate | 2.0 deg/s |
| wheel tangential speed | 0.10 m/s |
| phase | upstream/downstream discrete |

Two complementary sparse-cell profiles are used:

- **full physical cell**: root pose/twist + steering/hip/knee pose/rates + wheel tangential speeds;
- **root geometry cell**: root pose/twist only, used for macroscopic Tube geometry and frontier diversity.

This means two exact snapshots can have different SHA-256 values but still belong to the same physically resolved Tube cell.

---

## 6. Tube geometry

The longitudinal progress coordinate is root `x`, discretized in 0.10 m slices.

For each `x` slice, the empirical Tube cross-section is described from unique root-geometry cells over:

```text
y, z
vx, vy, vz
roll, pitch, yaw
wx, wy, wz
```

The result should be inspectable as a real spatial/dynamical Tube rather than only an unstructured list of snapshots.

The data are not forced into a cylindrical shape. A valid Tube may narrow, widen, branch, split into lobes or contain gaps.

Production implementation:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

Outputs include cell maps, x-slice statistics, CSV projection and visualization plots.

---

## 7. Raw snapshot count is no longer the expansion metric

Historical artifact cardinalities remain exact:

```text
Tube_0 raw snapshots =   222
Tube_1 raw snapshots = 3,119
Tube_2 raw snapshots = 3,776
```

These numbers describe replay/support cardinality only.

The previous statements such as “Tube_1 is 14.05x Tube_0” or “Tube_2 adds 21.06% capability” are no longer accepted as capability-envelope claims.

The correct retrospective quantities are:

```text
unique root geometry cells
unique full physical cells
new cells relative to source Tube
duplicate fraction
new cells by phase
new cells by x slice
nearest source-cell distance in resolution units
```

Tube_0/1/2 must now be reprocessed before the project claims quantitative state-space expansion.

---

## 8. Resolution-aware frontier acquisition

A future frontier parent is not considered independent merely because its state SHA differs.

Before outcomes are generated, the parent plan can be revised by:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Required parent uniqueness:

```text
newest Tube shell only
+ unique parent group
+ unique exact state
+ unique root_geometry_v1 cell
```

Default maximum: 25 distinct parent cells per phase. Minimum: 5 distinct newest-shell root-geometry cells per phase.

This increases the chance that additional simulation budget explores genuinely different regions instead of repeatedly sampling one small neighborhood.

---

## 9. Completed evidence chain

### Experts

`pi_up_star`

- 9,977,856 transitions
- actor `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`

`pi_down_star`

- 25,600 transitions
- actor `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`

### Raw Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

```text
222 snapshots = 117 upstream + 105 downstream
```

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- 10,009,600 transitions
- actor `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`

### Raw Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

```text
3,119 snapshots
= 222 retained Tube_0
+ 2,897 raw expansion snapshots
```

### pi_1 repair02

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

Historical engineering quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 parent groups
```

Historical formal PASS is not claimed because the old gate retained 3 baseline-reproduction mismatches.

### C^1

Upstream 64x64:

```text
AUC = 0.6903137789904502
recall = 0.5934515688949522
formal AUC>=0.70 = FAIL
engineering selection = YES
```

Downstream 64x64:

```text
AUC = 1.0
recall = 1.0
formal calibration = PASS
```

### Raw Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

```text
3,776 snapshots
= 3,119 retained Tube_1
+ 657 raw expansion snapshots
```

These 657 rows must now be measured for unique physical-cell novelty before being described as 657 meaningful expansion states.

### pi_2

Training completed at 10,009,600 transitions.

Locked source-panel comparison:

```text
pi_1 = 3115/3119
pi_2 = 3002/3119

upstream:   423/427 -> 312/427
downstream: 2692/2692 -> 2690/2692
```

Locked pi_1-negative frontier:

```text
pi_2 = 13/14
upstream = 4/5
downstream = 9/9
successful parent groups = 3
baseline reproduction failures = 0
```

Interpretation: clear local frontier progression evidence, but strong upstream single-policy realization degradation.

pi_2 is capability evidence; it is not retrospectively selected as the next formal authority.

---

## 10. Automatic iteration state

The existing generic workflow already automates:

```text
frontier roles
-> continuation fields
-> raw Tube construction
-> smoke/isolation
-> baseline lock
-> candidate training/freeze
-> locked evaluation
-> capability progression
-> prospective selection
```

The new resolution-aware Tube and frontier-parent capabilities are implemented but have not yet been exercised in a complete new prospective automatic iteration.

Before another iteration:

```text
analyze source Tube physical geometry
-> prepare ordinary pre-outcome frontier plan
-> revise parent set by root-geometry cells
-> execute frontier roles
```

Do not launch pi_3 until this resolution-aware retrospective analysis is complete and the next prospective workflow is explicitly declared.

---

## 11. Immediate scientific task

Reprocess all existing Tubes without new simulation:

```text
Tube_0 -> physical cells / x slices
Tube_1 vs Tube_0 -> genuinely new cells
Tube_2 vs Tube_1 -> genuinely new cells
```

Then answer:

1. How much of the 2,897 Tube_1 raw expansion is physically independent at the declared resolution?
2. How much of the 657 Tube_2 raw expansion is physically independent?
3. Which x slices actually gained cross-section coverage?
4. Is the upstream Tube one connected band, a narrow manifold, or multiple branches/lobes?
5. Does the pi_2 upstream realization loss coincide with a newly explored branch of the Tube?

Only after those questions are answered should the project decide whether it needs more frontier acquisition, different policy representation, or a new iteration.

---

## 12. Immutable task contract

- branch: `agent/two-phase-soft-tube`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control interval: 0.020 s = 50 Hz
- runtime simulation substep: 0.005 s
- hip/knee actuator range: +/-30 N m
- action order: `[steer, rear-wheel drive, hip, knee]`
- no runtime expert switching
- final TEST/JCE/JEL untouched

Older documentation that stated +/-50 N m was inconsistent with the authoritative XML and runtime model validation. The model itself has not been changed.

---

## 13. Authority documents

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`
