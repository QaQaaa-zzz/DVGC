# DVGC Repository Instructions

## Current research truth — 2026-09-04

DVGC/JIT is an iterative **real-dynamics capability-discovery and just-in-time
curriculum** project for a fixed single-track two-wheeled robot task.

The current scientific model separates three objects:

1. **physical/task feasibility `F*`** — the unknown set of states from which some
   admissible control behavior could complete the fixed task; JIT does not prove
   or exactly compute this set;
2. **cumulative empirical capability evidence `E_k`** — successful real-dynamics
   evidence accumulated across frozen experts and unified policies;
3. **single-policy realization coverage** — how much of that cumulative support a
   particular unified Actor realizes on a locked evaluation panel.

The final runtime target remains **one unified Actor** with no expert switching.
Phase experts and intermediate frozen policies are capability probes/data sources.
A Soft Tube is empirical TRAIN support/curriculum guidance, not a certified safe
set, viability kernel, reachability proof, invariant set, or physical-limit
certificate.

Read the full current report:

`JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`

## Current completed chain

```text
pi_up_star + pi_down_star
  -> bootstrap V_up / V_down
  -> Tube_0 = 222
  -> pi_0
  -> C^0
  -> Tube_1 = 3,119
  -> pi_1 repair02 engineering authority
  -> v3 TRAIN / v3b CALIBRATION / v3c ACCEPTANCE
  -> C^1 64x64 engineering selection
  -> Tube_2 = 3,776
  -> Tube_2 RSI smoke GO
  -> engineering role-isolation record
  -> pi_1 baseline locked before candidate training
  -> pi_2 trained/frozen at 10,009,600 transitions
  -> locked pi_1 vs pi_2 evaluation complete
  -> capability-progression decision implemented
  -> CURRENT: pi_2 is frontier-progression evidence but is not retrospectively
              selected as the next formal policy authority
```

Final TEST/JCE/JEL remains untouched.

## Authoritative artifacts and evidence

### Experts

- `pi_up_star`: 9,977,856 transitions; actor
  `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`.
- `pi_down_star`: 25,600 transitions; actor
  `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`.

Frozen manifest:

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

- 222 TRAIN states = 117 upstream + 105 downstream.

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- actor:
  `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`;
- payload:
  `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`.

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

```text
3,119 total
= 222 retained Tube_0
+ 2,897 expansion

upstream   427 = 117 + 310
downstream 2692 = 105 + 2587
```

Manifest SHA:

`817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`

### pi_1

Engineering-selected repair02:

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

Historical formal Iteration-1 PASS is not claimed because the old quickcheck
retains 3 baseline-reproduction mismatches from the historical PRNG protocol.

### C^1

The current C^1 artifact is engineering-qualified rather than a clean all-phase
formal pass.

`C_up^1` 64x64:

- AUC `0.6903137789904502`;
- recall `0.5934515688949522`;
- original AUC >= 0.70 gate remains false;
- engineering selection/override only.

`C_down^1` 64x64:

- AUC 1.0;
- recall 1.0;
- formal calibration PASS.

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

```text
3,776 total
= 3,119 retained Tube_1
+ 657 new expansion

upstream   902 = 427 + 475
downstream 2874 = 2692 + 182
```

Manifest SHA:

`135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`

No CALIBRATION, ACCEPTANCE, TEST, or final-evaluation rows were embedded.
Tube_2 RSI smoke is GO.

### Current pi_2 evidence

Training run id:

`pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

Training completed at 10,009,600 transitions using:

```text
outer reset: 90% Tube / 10% natural
inside Tube: 75% retained Tube_1 / 25% Tube_2 newest expansion
```

Locked source-Tube panel:

```text
pi_1 baseline success 3115/3119
pi_2 success          3002/3119
strict regressions    115
strict improvements   2
```

Phase split:

```text
upstream:
  pi_1 423/427 = 99.06%
  pi_2 312/427 = 73.07%
  strict regressions = 113

downstream:
  pi_1 2692/2692 = 100.00%
  pi_2 2690/2692 = 99.93%
  strict regressions = 2
```

Locked frontier challenge:

```text
14 pi_1-negative states
pi_2 success = 13/14
successful parent groups = 3
upstream 4/5
downstream 9/9
baseline reproduction failures = 0
```

Interpretation:

- empirical local frontier progression: strong;
- pi_2 upstream single-policy realization: substantially degraded;
- do not describe pi_2 as “no capability improvement”;
- do not describe pi_2 as formally selected under the new prospective criterion.

## Revised capability-progression decision

Stable analysis:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

CLI:

`JIT/cli/analyze_capability_progression.py`

Future prospective candidate selection separates:

### A. Frontier progression

Require:

- zero baseline-reproduction mismatch;
- nonzero candidate frontier success;
- required independent parent-group support;
- candidate frontier success in both phases.

### B. Policy realization

Fixed method-level non-inferiority proxy:

```text
maximum global Tube-panel coverage drop = 5 percentage points
maximum per-phase coverage drop = 10 percentage points
```

Zero individual paired regressions are **not** required.

A candidate becomes the sole next automatic policy authority only when A and B
both pass prospectively.

Current pi_2 is retrospective relative to this revised decision and therefore may
be analyzed but may not be formally selected from a retrospective artifact.
`select_iteration_policy.py` enforces this.

## Automatic workflow

Future generated DAG:

```text
selected pi_k + Tube_k
  -> newest-shell TRAIN/CALIBRATION/ACCEPTANCE
  -> C^k
  -> Tube_(k+1)
  -> Tube-RSI smoke
  -> strict role isolation
  -> lock pi_k baseline before candidate training
  -> train/freeze pi_(k+1)
  -> locked paired panel evaluation
  -> capability-progression analysis
       frontier progression
       phase-aware policy realization
  -> prospectively select pi_(k+1) only if both pass
```

The workflow never auto-tunes a failure.

The completed pi_1 -> pi_2 round was **not** a pristine end-to-end automatic run:

- frontier acquisition required v3/v3b scientific repairs;
- 64x64 C^1 was selected engineering-mainline on reused data;
- C_up^1 remained below the original AUC gate;
- all-role near-observation isolation required an explicit engineering
  continuation after TRAIN <-> ACCEPTANCE near-overlap was verified zero.

Do not claim full prospective hands-off automation for this round.

## JIT meaning going forward

Use this definition:

> JIT is an iterative real-dynamics capability-discovery and just-in-time
> curriculum framework that accumulates empirical jump-capability evidence under
> fixed robot dynamics, uses the current frontier to train a single unified
> policy, and separately measures frontier progression and how much of the
> cumulative capability that policy can realize.

Consequences:

- earlier successful capability evidence is not erased by a later failed single
  rollout;
- the latest policy is not the definition of physical feasibility;
- `C^k` remains policy-conditioned proposal/filter evidence, not existential
  controllability proof;
- a future discovery-time frozen policy archive is compatible with one-policy
  runtime deployment;
- final physical-envelope/JCE/JEL claims still require untouched final evidence.

## Next scientific work

Do **not** automatically run a 90/10 replay repair just because the old strict
regression count is 115.

Before another candidate or pi_3 round:

1. generate the retrospective capability-progression artifact for current pi_2;
2. preserve pi_2 as frontier/capability evidence while keeping pi_1 as the last
   formally selected engineering authority;
3. evaluate a goal-/intent-conditioned unified policy so one Actor can express
   different desired jump behaviors explicitly;
4. upgrade future policy realization evaluation to multiple predeclared seeds per
   state and success-rate/confidence reporting;
5. consider a frozen discovery-time policy archive without runtime switching;
6. predeclare the next method version before new candidate outcomes are observed.

## Immutable physical/task contracts

- branch: `agent/two-phase-soft-tube`;
- XML: `assets/orange_bike_4kg_horizontal.xml`;
- XML SHA:
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`;
- actual payload: 2 kg;
- control: 50 Hz;
- hip/knee torque: +/-50 Nm;
- action order: `[steer, rear-wheel drive, hip, knee]`;
- no runtime expert switching;
- do not silently change physics/reward/action/snapshot/task geometry/TEST
  isolation.

## Data-role contract

- `TRAIN`: fit continuation fields and contribute qualifying Tube expansion;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: development frontier comparison only;
- final TEST/JCE/JEL: untouched.

Parent-group disjointness remains mandatory.  The current round's engineering
near-observation continuation is historical evidence, not an automatic rule
relaxation for future rounds.

## Repository and Git safety

- modify/consolidate existing production code first;
- new source files require a genuinely durable capability;
- keep CLIs thin and logic under `JIT/src/jit_dvgc/`;
- preserve path-bound provenance;
- do not delete without dependency-closure proof and compile/tests;
- preserve unrelated user work;
- never reset, clean, stash, rebase, force-push, or overwrite unrelated work;
- use `/home/qy/mujoco_playground/.venv/bin/python`.

## Current authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
5. `JIT/docs/CODEX_HANDOFF_20260904.md`
6. `PROJECT.md`
7. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
8. `JIT/docs/CODE_ORGANIZATION.md`

`JIT/docs/CODEX_HANDOFF_20260903.md` is superseded historical context only.
