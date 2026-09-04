# DVGC/JIT Capability Progression Report — 2026-09-04

## Purpose

This document preserves the completed expert -> pi_2 progression evidence.

For the current method definition and next operator steps, use:

```text
JIT/docs/CURRENT_STATUS.md
JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md
```

The key later correction is:

> Tube entry count is raw replay-support cardinality, not physical capability-volume growth.

Therefore the exact history `222 -> 3,119 -> 3,776` remains valid, but it must not be interpreted as a physical-envelope multiplier.

---

## 1. Completed chain

```text
pi_up_star + pi_down_star
-> bootstrap V_up / V_down
-> raw Tube_0
-> pi_0
-> C^0
-> raw Tube_1
-> pi_1 repair02
-> v3/v3b/v3c frontier evidence
-> C^1 engineering selection
-> raw Tube_2
-> pi_2
-> locked pi_1 vs pi_2 comparison
```

Final TEST/JCE/JEL remains untouched.

---

## 2. Experts

`pi_up_star`:

```text
9,977,856 transitions
actor SHA f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081
```

`pi_down_star`:

```text
25,600 transitions
actor SHA 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

Frozen expert manifest:

```text
JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json
```

The experts are bootstrap capability probes/data sources, not runtime switching controllers.

---

## 3. Raw Tube_0 and pi_0

Tube_0:

```text
JIT/runs/soft_tube/soft_tube_train_v1_20260828
222 raw snapshots = 117 upstream + 105 downstream
```

Manifest SHA:

```text
c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b
```

pi_0:

```text
JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json
10,009,600 training transitions
```

Actor SHA:

```text
43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029
```

pi_0 established that one unified Actor could be trained from the two-phase bootstrap support.

---

## 4. Raw Tube_1 and pi_1

Tube_1:

```text
JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
```

```text
3,119 raw snapshots
= 222 retained Tube_0
+ 2,897 raw expansion snapshots

upstream   427
downstream 2,692
```

Manifest SHA:

```text
817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80
```

Selected pi_1 repair02:

```text
JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json
```

Engineering quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260
successful parent groups 4
```

Historical strict formal Iteration-1 PASS remains unclaimed because the old gate contains 3 baseline-reproduction mismatches from the historical PRNG hierarchy.

Warm-start A/B studies remain closed historical evidence.

---

## 5. pi_1 frontier and C^1

v3 TRAIN:

```text
1,031 total
upstream   821 = 785 positive + 36 negative, 9 parent groups
downstream 210 = 182 positive + 28 negative, 3 parent groups
```

v3b upstream CALIBRATION:

```text
739 = 733 positive + 6 negative, 3 parent groups
```

Downstream CALIBRATION:

```text
70 = 61 positive + 9 negative
```

v3c ACCEPTANCE:

```text
upstream   516 = 511 positive + 5 negative
downstream  70 =  61 positive + 9 negative
```

C^1 64x64:

```text
upstream:
  AUC 0.6903137789904502
  recall 0.5934515688949522
  original formal AUC>=0.70 FAIL
  engineering-selected

downstream:
  AUC 1.0
  recall 1.0
  formal calibration PASS
```

Do not state that all-phase C^1 formally passed.

---

## 6. Raw Tube_2 and pi_2

Tube_2:

```text
JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904
```

```text
3,776 raw snapshots
= 3,119 retained Tube_1
+ 657 raw expansion snapshots

upstream   902
downstream 2,874
```

Manifest SHA:

```text
135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff
```

Tube-RSI smoke: GO.

pi_2 run id:

```text
pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904
```

Training completed at 10,009,600 transitions.

---

## 7. Locked pi_1 vs pi_2 result

Source panel:

```text
pi_1 = 3115/3119
pi_2 = 3002/3119
strict regressions = 115
strict improvements = 2
```

Phase split:

```text
upstream:
  pi_1 423/427
  pi_2 312/427
  regressions 113

downstream:
  pi_1 2692/2692
  pi_2 2690/2692
  regressions 2
```

Locked pi_1-negative frontier:

```text
pi_2 = 13/14
upstream = 4/5
downstream = 9/9
successful parent groups = 3
baseline reproduction failures = 0
```

Scientific interpretation:

```text
local frontier progression = strong
upstream single-policy realization = degraded
```

pi_2 remains capability evidence but is not retrospectively promoted as the next formal authority.

---

## 8. Capability-progression semantic revision

The project now separates:

```text
frontier progression
from
single-policy realization retention
```

Prospective engineering non-inferiority margins:

```text
max global locked-panel coverage drop = 5 percentage points
max per-phase coverage drop           = 10 percentage points
```

Strict zero-regression remains a diagnostic rather than the definition of empirical envelope progression.

Because this semantic revision followed the observed pi_2 result, current pi_2 can only be analyzed retrospectively and cannot be selected retroactively under the new criterion.

---

## 9. Resolution-aware correction

The current method distinguishes:

```text
raw snapshots
unique root-geometry physical cells
unique full-physical cells
```

Resolution v1:

```text
position                     0.10 m
linear velocity              0.10 m/s
orientation                  0.50 deg
angular velocity             2.0 deg/s
joint angle                  0.50 deg
joint angular velocity       2.0 deg/s
wheel tangential speed       0.10 m/s
phase                        discrete
```

The physical Tube is sliced along root `x` at 0.10 m intervals.

Until the retrospective analysis is executed, the project does **not** know the exact physical-cell expansion from Tube_0 to Tube_1 or Tube_1 to Tube_2.

---

## 10. Current next step

Do not train pi_3 or change reward/intent.

Run:

```text
Tube_0 physical-cell reconstruction
Tube_1 versus Tube_0 cell comparison
Tube_2 versus Tube_1 cell comparison
```

Exact commands and interpretation:

```text
JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md
```

The next scientific decision must use those physical Tube results rather than raw cardinality alone.
