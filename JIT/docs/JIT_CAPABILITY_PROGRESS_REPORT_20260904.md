# DVGC/JIT Historical Capability Progression Report — 2026-09-04

## Purpose

This document preserves the completed engineering evidence from phase experts through `pi_2`.

It is **historical evidence**, not the active method definition.

Current method authority:

```text
JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md
```

Current scientific correction:

```text
RSI continuation success != forward reachability

J_k = R_k^forward ∩ V_k^continuation
```

Therefore the historical chain remains valuable for engineering provenance, continuation learning, replay curriculum and policy-realization analysis, but old RSI-anchored frontier results must not be retroactively labeled ground-connected Jump Capability.

---

## 1. Completed historical chain

```text
pi_up_star + pi_down_star
-> bootstrap V_up / V_down
-> raw Tube_0
-> pi_0
-> C^0
-> raw Tube_1
-> pi_1 repair02
-> v3/v3b/v3c frontier engineering
-> C^1 64x64 engineering path
-> raw Tube_2
-> pi_2
-> locked pi_1 vs pi_2 evaluation
-> physical-resolution retrospective analysis
```

---

## 2. Phase experts

```text
pi_up_star
  transitions 9,977,856
  actor SHA f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081

pi_down_star
  transitions 25,600
  actor SHA 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

Historical role: bootstrap propulsion/ascent and descent/recovery control/continuation evidence. They are not final runtime controllers.

---

## 3. Raw Tube_0

Path:

```text
JIT/runs/soft_tube/soft_tube_train_v1_20260828
```

Historical raw counts:

```text
222 snapshots
117 upstream
105 downstream
```

Retrospective physical-resolution analysis:

```text
100 unique root_geometry_v1 cells
112 unique full_physical_v1 cells
13 occupied x slices
```

Interpretation now:

> bootstrap replay/control support with significant local duplication; not a causal Jump-Capability set.

---

## 4. pi_0

```text
training transitions 10,009,600
actor SHA 43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029
```

Historical significance:

> first unified-policy demonstration that the two phase-specific support sources can train one Actor without runtime expert switching.

---

## 5. Raw Tube_1

Path:

```text
JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
```

Historical raw counts:

```text
3,119 snapshots
= 222 retained Tube_0 + 2,897 raw expansion
```

Retrospective physical-resolution analysis:

```text
2,142 root cells
2,404 full cells
2,042 new root cells vs Tube_0
2,292 new full cells vs Tube_0
24 occupied x slices
```

Historical all-state expansion was therefore real in a physical-cell occupancy sense, not merely SHA duplication.

However it must not be described as a 21x causal Jump-Capability expansion because:

- natural-start forward ancestry was not required for every expansion row;
- old downstream labels included landing/recovery states;
- the old acquisition definition was RSI-anchor based.

---

## 6. pi_1 repair02

Selected engineering authority:

```text
JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json
```

Actor SHA:

```text
85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a
```

Historical quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 parent groups
```

Historical formal Iteration-1 PASS is not claimed because three old baseline-reproduction mismatches remain quarantined.

`pi_1` remains the selected engineering authority for the current causal redesign unless/until a future prospectively validated causal iteration selects a successor.

---

## 7. Historical frontier engineering before C^1

Phase-specific v3/v3b/v3c work established useful engineering facts about continuation-label support and data-role design.

Important historical results include:

```text
v3 TRAIN
  upstream 821 = 785 positive / 36 negative
  downstream 210 = 182 positive / 28 negative

v3b upstream CALIBRATION repair
  739 candidates = 733 positive / 6 negative

v3c fresh ACCEPTANCE challenge
  upstream 516 = 511 positive / 5 negative
  downstream 70 = 61 positive / 9 negative
```

These artifacts remain valuable for continuation-model history and role-isolation methodology.

They are not natural-start-connected causal capability sets.

---

## 8. C^1 engineering path

Official engineering-selected network:

```text
76 -> 64 tanh -> 64 tanh -> 1
9,153 parameters per phase
```

Upstream calibration:

```text
AUC    0.6903137789904502
recall 0.5934515688949522
accepted negatives 0
all parent coverage PASS
```

Original formal gate required AUC >= 0.70, therefore upstream remained:

```text
formal calibration FAIL
engineering selection only
```

Downstream:

```text
AUC 1.0
recall 1.0
accepted negatives 0
formal calibration PASS
```

Do not rewrite the upstream result as a formal PASS.

---

## 9. Raw Tube_2

Path:

```text
JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904
```

Historical raw counts:

```text
3,776 snapshots
= 3,119 retained Tube_1 + 657 raw expansion
```

Retrospective physical-resolution analysis:

```text
2,446 root cells
2,871 full cells
304 new root cells vs Tube_1
467 new full cells vs Tube_1
24 occupied x slices
```

By phase, new root cells were:

```text
upstream   +199
downstream +105
```

This remains useful evidence that the second engineering round shifted new physical occupancy toward upstream support.

It is still not a causal Jump-Capability count because forward ancestry was not a prerequisite.

---

## 10. pi_2 training

Training completed at:

```text
10,009,600 transitions
```

Historical training used the declared Tube-RSI mixture with retained-core replay and natural resets.

No claim is made that this training configuration is a theorem or permanent optimal setting.

---

## 11. Locked pi_1 vs pi_2 realization result

Source panel:

```text
pi_1 3115/3119
pi_2 3002/3119
```

By phase:

```text
upstream
423/427 -> 312/427

 downstream
2692/2692 -> 2690/2692
```

Strict regression count:

```text
115
```

This remains valid evidence of substantial upstream single-policy realization loss.

---

## 12. Historical pi_1-negative frontier challenge

Result:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
successful parent groups 3
baseline reproduction failures 0
```

Current interpretation:

```text
VALID:
old-protocol continuation/frontier success evidence

NOT VALID:
natural-start-connected causal Jump-Capability proof
```

Reason: the old frontier acquisition began from RSI/Tube anchors. Short local perturbations were real dynamics, but the anchor itself was not required to have a ground-connected causal path.

This historical correction is central to the current paper narrative and must not be hidden.

---

## 13. What the historical chain did establish

The completed engineering work still established substantial reusable infrastructure and evidence:

1. two phase experts can seed a unified-policy pipeline;
2. one unified Actor can consume Tube-RSI curriculum without runtime expert switching;
3. continuation labels and separate continuation fields can be built with TRAIN/CALIBRATION/ACCEPTANCE roles;
4. core-retaining replay can repair severe old-state forgetting in at least one iteration;
5. raw snapshot count strongly differs from independent physical-cell count;
6. historical downstream phase labels can include late recovery and therefore need semantic filtering;
7. a candidate policy can gain local frontier behavior while losing large prior-state realization;
8. strict policy realization and cumulative capability evidence should be reported separately;
9. RSI is efficient for continuation testing but cannot establish reachability.

These findings motivate rather than invalidate the current causal JIT method.

---

## 14. What must be demonstrated next

The next evidence must be prospectively causal:

```text
locked natural-start centerline
-> every-x causal proposal plan
-> natural-start forward acquisition
-> ground-reachability provenance
-> continuation evaluation
-> Reachable ∩ Viable capability cells
-> per-x cross-section growth
```

Only after such causal evidence exists should the project train the next unified policy under the revised method.

---

## 15. Current authority documents

For current work use:

```text
AGENTS.md
JIT/AGENTS.md
JIT/docs/CURRENT_STATUS.md
JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md
PROJECT.md
JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md
JIT/docs/CODEX_HANDOFF_20260904.md
JIT/docs/CODE_ORGANIZATION.md
```

This file is last in the read order because it preserves history rather than defining the active method.
