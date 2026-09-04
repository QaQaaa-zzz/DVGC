# DVGC Project

## Scientific objective

DVGC/JIT studies **iterative real-dynamics jump-capability discovery with
just-in-time curriculum generation** for one fixed single-track two-wheeled robot
task.

The project distinguishes three levels:

1. **Physical/task feasibility `F*`** — the unknown set of states from which some
   admissible control behavior could complete the fixed task. JIT does not prove
   or exactly compute this set.
2. **Cumulative empirical capability evidence `E_k`** — successful real-dynamics
   evidence accumulated across frozen phase experts and unified policies.
3. **Single-policy realization coverage** — how much of the cumulative support a
   particular unified Actor realizes on a locked evaluation panel.

The final runtime target remains **one unified policy**. Phase experts and frozen
intermediate policies are capability probes/data sources, not a runtime switching
architecture.

A Soft Tube is empirical TRAIN support/curriculum guidance. It is not a certified
safe set, viability kernel, reachability proof, invariant set, or proof of the
physical jump limit.

---

## Core JIT loop

```text
frozen capability probe(s)
        ↓
real-dynamics frontier acquisition
        ↓
continuation evidence near the success/failure transition
        ↓
cumulative empirical capability support
        ↓
just-in-time Tube curriculum
        ↓
train one unified policy
        ↓
locked evaluation
  A. did the empirical frontier move?
  B. does the unified policy still realize enough prior support?
        ↓
repeat or open a new method decision
```

The curriculum is generated from the current frontier rather than manually
prescribed as a fixed easy-to-hard schedule.

---

## Bootstrap phase

```text
Propulsion-Ascent expert pi_up
        +
Descent-Recovery expert pi_down
        ↓
freeze experts
        ↓
real handoff / continuation evidence
        ↓
expert-conditioned V_up / V_down
        ↓
TRAIN-only Tube_0
        ↓
unified Tube-RSI policy pi_0
```

The experts bootstrap capability evidence. They are not deployed together.

---

## Iterative phase

For selected frozen `pi_k` and cumulative `Tube_k`:

```text
selected pi_k + Tube_k
        ↓
outcome-blind newest-shell TRAIN / CALIBRATION / ACCEPTANCE
        ↓
real-dynamics frontier acquisition under pi_k
        ↓
pi_k-conditioned continuation labels
        ↓
fit C_up^k / C_down^k on TRAIN
        ↓
calibrate on disjoint CALIBRATION
        ↓
Tube_(k+1)
= every Tube_k entry retained exactly
+ qualifying TRAIN expansion
        ↓
Tube-RSI smoke + role isolation
        ↓
lock pi_k evaluation baseline before candidate training
        ↓
train/freeze unified pi_(k+1)
        ↓
locked paired evaluation
        ↓
capability-progression analysis
  A. frontier progression
  B. phase-aware policy realization
        ↓
prospectively select pi_(k+1) only when A + B pass
```

Historical engineering overrides remain explicit and are not rewritten to look
prospective.

---

## Why zero single-state regression is no longer the envelope definition

The current unified policy is reward-guided and does not receive an explicit
requested jump target such as desired distance, apex height, or landing position.
The same state may therefore admit multiple reasonable control responses.

A stochastic/reward-guided policy is not expected to reproduce one unique
successful rollout for every previously successful state.

Therefore:

- strict paired regressions remain useful behavioral diagnostics;
- a later failed rollout does not erase earlier empirical capability evidence;
- current-policy quality is measured by phase-aware coverage over cumulative
  support;
- a severe phase-specific coverage collapse still blocks automatic promotion to
  sole next-policy authority.

Future prospective selection uses the capability-progression v1 margins:

```text
max global Tube-panel coverage drop = 5 percentage points
max per-phase coverage drop         = 10 percentage points
```

These are engineering non-inferiority margins, not physical safety or feasibility
thresholds.

---

## Data-role contract

- `TRAIN`: may fit `C^k` and contribute qualifying Tube expansion;
- `CALIBRATION`: continuation-threshold calibration only;
- `ACCEPTANCE`: candidate-blind development frontier comparison only;
- final TEST/JCE/JEL: untouched until method, stopping rule, and final policy are
  frozen.

CALIBRATION and ACCEPTANCE rows never enter a Tube. Parent-group disjointness
remains mandatory.

---

## Completed evidence chain — 2026-09-04

### Frozen phase experts

`pi_up_star`

- 9,977,856 transitions;
- actor SHA-256:
  `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`.

`pi_down_star`

- 25,600 transitions;
- actor SHA-256:
  `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`.

Frozen manifest:

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

```text
222 TRAIN states
= 117 upstream
+ 105 downstream
```

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- 10,009,600 transitions;
- actor:
  `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`;
- payload:
  `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`.

### C^0 and Tube_1

`C_up^0/C_down^0` were fitted from frozen-pi_0 continuation evidence and passed
the then-declared independent calibration path.

Tube_1:

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

```text
retained Tube_0 = 222
new expansion   = 2,897
total           = 3,119

upstream   = 427  = 117 + 310
downstream = 2,692 = 105 + 2,587
```

Tube_1 contains about 14.05 times as many support entries as Tube_0. This is a
cardinality ratio, not a state-space volume ratio.

### pi_1 repair02

Frozen engineering authority:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

- actor:
  `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`;
- payload:
  `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`.

Historical quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 parent groups
```

Historical formal Iteration-1 PASS is not claimed because the old gate contains 3
baseline-reproduction mismatches from the historical PRNG hierarchy.

### Iteration-1 frontier evidence

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

### C^1

64x64 tanh was explicitly selected for engineering continuation.

Upstream:

- AUC `0.6903137789904502`;
- recall `0.5934515688949522`;
- original AUC >= 0.70 gate remains false;
- engineering selection only.

Downstream:

- AUC 1.0;
- recall 1.0;
- formal calibration PASS.

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

```text
retained Tube_1 = 3,119
new expansion   =   657
total           = 3,776

upstream   =   902 = 427 + 475
downstream = 2,874 = 2,692 + 182
```

Manifest SHA:

`135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`

Tube_1 -> Tube_2 entry growth is about 21.06%. Tube_2 contains about 17.01 times
as many entries as Tube_0.

Tube_2 RSI smoke is GO.

### pi_2

Training run id:

`pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

Training completed at 10,009,600 transitions with:

```text
outer reset: 90% Tube / 10% natural
inside Tube: 75% retained Tube_1 / 25% newest Tube_2 expansion
```

No expert switching, TEST, or validation data were used.

Locked comparison:

```text
Tube_1 source panel = 3,119 states
pi_1 success        = 3,115
pi_2 success        = 3,002
strict regressions  = 115
strict improvements = 2
```

Phase split:

```text
upstream:   423/427 -> 312/427  (~25.995 percentage-point drop)
downstream: 2692/2692 -> 2690/2692 (~0.074 percentage-point drop)
```

Locked pi_1-negative frontier challenge:

```text
pi_2 success = 13/14
parent groups = 3
upstream = 4/5
downstream = 9/9
baseline reproduction failures = 0
```

Current interpretation:

- empirical frontier progression: **strong**;
- pi_2 upstream single-policy realization: **degraded**;
- pi_2 is retained as capability evidence;
- pi_2 is **not retrospectively selected** under the newly revised prospective
  criterion.

---

## What the task has gained

1. Two phase-specific skills were converted into one unified JIT learning
   pipeline.
2. Empirical TRAIN support expanded from 222 to 3,776 entries while preserving
   prior Tube provenance structurally.
3. Two-phase frontier acquisition became informative, with both positive and
   negative continuation evidence.
4. `pi_2` demonstrated substantial new local frontier capability: 13/14 successes
   on states locked as `pi_1` failures, with success in both phases.
5. The pre-candidate locked-baseline protocol removed the old boundary PRNG
   reproduction mismatch in the current round.
6. The project identified a central research distinction: cumulative capability
   and latest-policy coverage are different quantities.

---

## Current automatic-iteration state

Future generic automation now contains:

```text
frontier plan
-> TRAIN
-> CALIBRATION
-> ACCEPTANCE
-> C^k
-> Tube_(k+1)
-> smoke
-> role isolation
-> lock baseline
-> train/freeze candidate
-> locked paired evaluation
-> capability-progression analysis
-> prospective selection only if frontier + policy realization pass
```

The completed pi_1 -> pi_2 round was not a pristine automatic round because it
required explicit interventions at frontier acquisition, C_up^1
architecture/calibration, and near-observation isolation.

Do not claim otherwise.

---

## Next scientific direction

Do not automatically treat 75/25 -> 90/10 replay as the answer.

The deeper representation problem is that the unified policy is not told what
jump behavior is desired. Before another candidate or pi_3 round, the next method
version should consider:

1. **goal-/intent-conditioned unified policy** — one runtime Actor, but with a
   low-dimensional requested jump outcome/behavior code;
2. **multi-seed probability-style policy coverage** — estimate success rates or
   confidence rather than one rollout per state;
3. **discovery-time frozen policy archive** — accumulate capability evidence using
   older successful probes without runtime policy switching;
4. only then decide whether a replay-only candidate remains scientifically useful.

No pi_3 work should begin until the next method version is explicitly declared.

---

## Immutable task contract

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA:
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b6f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`
- unified runtime: no expert switching
- final TEST/JCE/JEL: untouched until final method/policy/stopping decision

---

## Authority documents

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
5. `JIT/docs/CODEX_HANDOFF_20260904.md`
6. `PROJECT.md`
7. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
8. `JIT/docs/CODE_ORGANIZATION.md`

The 2026-09-03 handoff is superseded historical context only.
