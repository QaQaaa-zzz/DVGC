# Current JIT status — 2026-09-04

## Executive state

The project has completed the full engineering `pi_1 -> C^1 -> Tube_2 -> pi_2`
round and has revised the scientific meaning of the candidate gate.

Current chain:

```text
pi_up_star + pi_down_star
  -> Tube_0 = 222
  -> pi_0 frozen
  -> C^0
  -> Tube_1 = 3,119
  -> pi_1 repair02 selected as engineering authority
  -> v3 TRAIN / v3b CALIBRATION / v3c ACCEPTANCE
  -> C^1 64x64 engineering selection
  -> Tube_2 = 3,776
  -> Tube_2 smoke GO
  -> role-isolation engineering continuation
  -> pi_1 baseline locked before candidate training
  -> pi_2 trained/frozen at 10,009,600 transitions
  -> locked pi_1 vs pi_2 evaluation complete
  -> CURRENT: capability-progression semantics revised and coded
```

Current scientific result:

> `pi_2` demonstrates strong new local frontier capability but loses substantial
> upstream single-policy coverage over the cumulative Tube_1 support.  Therefore
> the capability evidence is retained, but `pi_2` is not retrospectively promoted
> to the next formal automatic policy authority.

Final TEST/JCE/JEL remains untouched.

---

## Scientific meaning now

Do not equate these three objects:

1. conceptual fixed-task physical feasibility `F*`;
2. cumulative empirical capability evidence `E_k` gathered across frozen probes;
3. current-policy realization coverage on `Tube_k`.

JIT does not prove `F*`.  A Tube is empirical TRAIN support/curriculum, not a
certified safe/viable/invariant set.  A later policy failing one prior state does
not erase the historical successful evidence for that state.

Future candidate decisions therefore separate:

```text
A. frontier progression
B. phase-aware policy realization retention
```

Zero individual paired regressions are no longer the envelope definition.

---

## Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- active branch: `agent/two-phase-soft-tube`
- expected local repository: `~/DVGC`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256:
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control rate: 50 Hz
- hip/knee torque: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`
- unified runtime: no expert switching
- final TEST/JCE/JEL: untouched

---

## Stable bootstrap chain

### Frozen experts

`pi_up_star`

- 9,977,856 transitions;
- actor SHA:
  `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`.

`pi_down_star`

- 25,600 transitions;
- actor SHA:
  `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`.

Frozen manifest:

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

```text
222 = 117 upstream + 105 downstream
```

Manifest SHA:

`c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- 10,009,600 transitions;
- actor SHA:
  `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`;
- payload SHA:
  `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`.

### C^0 and Tube_1

`C_up^0/C_down^0` passed the then-declared independent validation/calibration
path under frozen `pi_0`.

Tube_1:

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

```text
retained Tube_0 = 222
expansion       = 2,897
total           = 3,119

upstream   = 427  = 117 + 310
downstream = 2,692 = 105 + 2,587
```

Manifest SHA:

`817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`

Entries SHA:

`61c6796aaf4c4b1e43624c5cf06bce0d39736a6d1743c5142c6c250d23155ec9`

---

## pi_1 authority and closed Iteration-1 policy study

Selected engineering `pi_1`: repair02.

Frozen policy:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

Identity:

- actor SHA:
  `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`;
- payload SHA:
  `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`.

Historical quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260
successful parent groups 4
```

Historical formal PASS remains unclaimed because the old quickcheck contains 3
baseline-reproduction mismatches caused by the historical PRNG hierarchy.

Warm-start A/B and B checkpoint sweeps are historical/closed.  Do not reopen them
as the current mainline.

---

## pi_1 frontier evidence

Useful v3 TRAIN:

```text
total      1,031
upstream     821 = 785 positive + 36 negative, 9 parent groups
downstream   210 = 182 positive + 28 negative, 3 parent groups
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

---

## C^1 status

### Upstream

Selected engineering profile:

```text
76 -> 64 tanh -> 64 tanh -> 1
9,153 parameters
```

Metrics:

- threshold: `0.9835533512239714`;
- ROC AUC: `0.6903137789904502`;
- positive recall: `0.5934515688949522`;
- accepted negatives: 0;
- all calibration parents have accepted positive support.

The original AUC >= 0.70 requirement remains **not passed**.  The 64x64 model is
an explicit engineering selection, not a rewritten formal result.

### Downstream

Same 64x64 profile:

- 70 calibration candidates = 61 positive + 9 negative;
- threshold: `0.015432215517145933`;
- ROC AUC: 1.0;
- positive recall: 1.0;
- accepted negatives: 0;
- formal calibration PASS.

C^1 selection summary status:

`completed_engineering_selected`

---

## Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

Composition:

```text
retained Tube_1 = 3,119
new expansion   =   657
total           = 3,776

upstream   =   902 = 427 + 475
downstream = 2,874 = 2,692 + 182
```

Manifest SHA:

`135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`

Tube_1 -> Tube_2 cardinality increase:

- total: +21.06%;
- upstream: +111.24%;
- downstream: +6.76%.

These are entry-count changes, not physical state-space volume changes.

Tube_2 embeds:

```text
CALIBRATION rows = 0
ACCEPTANCE rows  = 0
TEST rows        = 0
```

### Tube_2 smoke

`status = completed`

`tube_rsi_smoke = GO`

16 total interactions in the smoke panel, 8 per phase, all finite.

---

## Role isolation for the current round

Engineering report:

`status = independent_for_candidate_training_engineering`

Exact state overlap:

```text
TRAIN <-> CALIBRATION = 0
TRAIN <-> ACCEPTANCE  = 0
CALIBRATION <-> ACCEPTANCE = 0
```

Parent groups are disjoint.

Near-observation overlap at `atol=0.01`:

```text
TRAIN <-> CALIBRATION       = 140
TRAIN <-> ACCEPTANCE        = 0
CALIBRATION <-> ACCEPTANCE  = 157
```

Therefore:

- candidate-training vs acceptance geometric isolation: preserved;
- all-role geometric isolation: not formally passed;
- explicit engineering override recorded.

Do not silently generalize this exception into the future automatic workflow.

---

## Locked pi_1 baseline

Before pi_2 training:

```text
source Tube_1 states          = 3,119
pi_1 baseline success count   = 3,115
status = locked_before_candidate_training
```

The later gate confirms phase baseline support:

```text
upstream   423/427
downstream 2692/2692
```

The locked baseline removed the historical boundary reroll/PRNG mismatch for this
round.

---

## pi_2 training

Run id:

`pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

Completed:

```text
requested transitions = 10,009,600
completed transitions = 10,009,600
```

Training contract:

```text
outer reset:
  90% Tube RSI
  10% natural

inside Tube:
  75% retained source Tube_1
  25% Tube_2 newest expansion
```

No expert switching, TEST, or validation use.

---

## pi_1 vs pi_2 locked result

### Strict old-core diagnostic

Overall:

```text
state_count                 3,119
pi_1 baseline success      3,115
pi_2 candidate success     3,002
strict regression_count      115
strict improvement_count       2
```

Overall panel coverage:

```text
pi_1 = 99.87%
pi_2 = 96.25%
global drop ≈ 3.62 percentage points
```

Phase split:

```text
upstream:
  pi_1 423/427 = 99.06%
  pi_2 312/427 = 73.07%
  regressions = 113
  coverage drop ≈ 25.995 percentage points

downstream:
  pi_1 2692/2692 = 100.00%
  pi_2 2690/2692 = 99.93%
  regressions = 2
  coverage drop ≈ 0.074 percentage points
```

The global number hides a severe upstream degradation because downstream dominates
Tube cardinality.

### Frontier/boundary evidence

Locked pi_1-negative challenge:

```text
state_count = 14
pi_2 success = 13
successful parent groups = 3
minimum groups = 2
baseline reproduction failures = 0
```

By phase:

```text
upstream   4/5
downstream 9/9
```

Old strict `iteration_accepted = false` because zero regression was required.

Revised scientific interpretation:

```text
frontier progression = strong
upstream policy realization = degraded
```

---

## New capability-progression semantics in code

Implementation:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

CLI:

`JIT/cli/analyze_capability_progression.py`

Prospective v1 decision:

### A. Frontier progression

Require:

- no baseline-reproduction mismatch;
- nonzero candidate boundary success;
- minimum parent-group diversity;
- success in both phases.

### B. Policy realization retention

Fixed locked-panel engineering proxy:

```text
max global coverage drop = 5 percentage points
max phase coverage drop  = 10 percentage points
```

A candidate may have nonzero strict paired regressions and still be selected if
coverage remains within the fixed non-inferiority margins and frontier progression
passes.

Current pi_2 expected retrospective classification:

```text
empirical_envelope_expansion_observed = true
global coverage margin = pass
upstream margin = fail
downstream margin = pass
candidate_policy_authority_eligible = false
```

Because the criterion was revised after seeing this pi_2 result, use
`--retrospective`.  Such an artifact cannot formally select pi_2.

---

## Updated automatic workflow

Future generated workflows now use:

```text
selected pi_k
-> frontier TRAIN/CALIBRATION/ACCEPTANCE
-> C^k
-> Tube_(k+1)
-> smoke
-> role isolation
-> lock baseline
-> train/freeze pi_(k+1)
-> locked paired evaluation
-> analyze_capability_progression
-> select only if frontier progression + policy realization both pass prospectively
```

The workflow does not automatically repair a failed policy.

### Current automation maturity

Implemented generically:

- orchestration/resume;
- newest-shell frontier roles;
- Tube retention/expansion;
- smoke;
- baseline lock;
- candidate train/freeze/evaluate;
- new capability decision;
- prospective selection.

But current pi_1 -> pi_2 evidence required manual scientific interventions:

- phase-specific v3 frontier redesign;
- v3b calibration repair;
- 64x64 C^1 engineering selection;
- upstream AUC engineering override;
- engineering near-observation isolation continuation.

Therefore current evidence is **not** proof of fully hands-off automatic JIT.

---

## Current position / what to do next

Do not start pi_3 and do not automatically launch a 90/10 replay repair.

Immediate next action after pulling current code:

```bash
$PY JIT/cli/analyze_capability_progression.py \
  --gate-summary ${GATE}/summary.json \
  --output ${ROOT}/pi_1_to_pi_2_capability_progression_retrospective.json \
  --retrospective
```

This should record frontier progression while refusing retrospective formal
selection of pi_2.

Next scientific method decision:

1. consider a goal-/intent-conditioned unified policy because current policy input
   does not specify desired jump behavior;
2. upgrade fixed-panel evaluation to multiple predeclared seeds per state so
   success probability/confidence can be estimated;
3. consider a frozen discovery-time policy archive so accumulated system
   capability is not limited to what the latest policy can probe;
4. then predeclare the next candidate/training method before running it.

No final TEST/JCE/JEL yet.

---

## Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
5. `JIT/docs/CODEX_HANDOFF_20260904.md`
6. `PROJECT.md`
7. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
8. `JIT/docs/CODE_ORGANIZATION.md`

The 2026-09-03 handoff is superseded historical context only.
