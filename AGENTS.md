# DVGC Repository Instructions

## Current research truth — 2026-09-04

DVGC/JIT is an iterative **real-dynamics capability-discovery and just-in-time
curriculum** project for a fixed single-track two-wheeled robot task.

The project no longer defines the empirical jumping envelope as “the set that the
latest policy reproduces perfectly.”  Keep three layers separate:

1. **physical/task feasibility `F*`** — states from which some admissible control
   behavior could complete the fixed task; JIT does not prove this set;
2. **cumulative empirical capability evidence `E_k`** — successful real-dynamics
   evidence accumulated across frozen experts/policies and represented by
   provenance-bound Tube/frontier artifacts;
3. **single-policy realization coverage** — how much of that cumulative support a
   particular unified policy realizes on a locked evaluation panel.

The phase experts and later frozen unified policies are capability probes.  The
runtime/deployment target remains **one unified Actor** with no expert switching.
A learned Soft Tube is empirical TRAIN support/curriculum guidance, not a
certified safe set, viability kernel, reachability proof, invariant set, or
physical limit certificate.

Read the full scientific report first when taking over current work:

`JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`

## Current completed chain

```text
pi_up_star + pi_down_star
  -> bootstrap V_up / V_down
  -> Tube_0 = 222
  -> unified pi_0
  -> pi_0-conditioned C^0
  -> Tube_1 = 3,119
  -> pi_1 repair02 selected as engineering authority
  -> phase-specific frontier v3 / calibration v3b / acceptance v3c
  -> C^1 64x64 engineering selection
  -> Tube_2 = 3,776
  -> Tube_2 RSI smoke GO
  -> engineering role-isolation record
  -> pi_1 baseline locked before candidate training
  -> pi_2 trained/frozen at 10,009,600 transitions
  -> locked pi_1 vs pi_2 comparison complete
  -> capability-progression semantics revised in code
  -> CURRENT: pi_2 is frontier-progress evidence but not retrospectively selected
              as the next formal policy authority
```

Final TEST/JCE/JEL remains untouched.

## Key current evidence

### Experts

- `pi_up_star`: 9,977,856 transitions, actor
  `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`.
- `pi_down_star`: 25,600 transitions, actor
  `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`.

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

- 222 TRAIN states = 117 upstream + 105 downstream.

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- 10,009,600 transitions;
- actor `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`;
- payload `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`.

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

- retained Tube_0: 222;
- expansion: 2,897;
- total: 3,119;
- upstream: 427 = 117 + 310;
- downstream: 2,692 = 105 + 2,587;
- manifest
  `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`.

### pi_1

Engineering-selected repair02:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

- actor `85d6b4667364daf8e054158?` — do not use a partial value; authoritative actor:
  `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`;
- payload
  `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`;
- historical quickcheck Tube_0 222/222 and boundary 26/260 across 4 groups;
- historical formal PASS is not claimed because the old quickcheck retained 3
  baseline-reproduction mismatches from the old PRNG protocol.

### C^1

The Iteration-1 continuation stage is engineering-qualified, not a clean
all-phase formal pass.

`C_up^1` selected profile:

- `76 -> 64 tanh -> 64 tanh -> 1`;
- AUC `0.6903137789904502`;
- recall `0.5934515688949522`;
- original AUC >= 0.70 rule remains false;
- explicit engineering selection/override only.

`C_down^1`:

- same 64x64 profile;
- AUC 1.0;
- recall 1.0;
- formal calibration PASS.

Do not rewrite `C_up^1` as a formal AUC pass.

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

- source Tube_1 retained exactly: 3,119;
- new expansion: 657;
- total: 3,776;
- upstream: 902 = 427 + 475;
- downstream: 2,874 = 2,692 + 182;
- manifest
  `135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`;
- no CALIBRATION, ACCEPTANCE, TEST, or final rows embedded.

Tube_2 RSI smoke is GO.

### Current pi_2 result

Training run id:

`pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

Training completed at 10,009,600 transitions with:

- 90% Tube / 10% natural reset;
- inside Tube, 75% retained Tube_1 / 25% Tube_2 newest expansion;
- no expert switching;
- no TEST/validation.

Locked pi_1 -> pi_2 comparison:

```text
source Tube_1 states:        3,119
pi_1 baseline successes:     3,115
pi_2 candidate successes:    3,002
strict regressions:            115
strict improvements:             2
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

Locked boundary/frontier:

```text
pi_1-negative challenge states = 14
pi_2 successes                 = 13
successful parent groups       = 3
upstream                       = 4/5
downstream                     = 9/9
baseline reproduction failures = 0
```

Interpretation:

- **empirical frontier progression is strong**;
- **current pi_2 upstream policy realization is substantially degraded**;
- do not describe pi_2 as “no capability improvement”;
- do not describe pi_2 as the next formally selected authority either.

## Revised iteration decision semantics

The historical strict zero-regression gate remains valid as a diagnostic and as
reproducibility evidence for old selections.  Future automatic iterations use a
separate capability-progression decision:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

CLI:

`JIT/cli/analyze_capability_progression.py`

Future prospective policy selection asks two questions.

### A. Empirical frontier progression

Require:

- zero baseline-reproduction mismatch;
- nonzero candidate boundary success;
- required independent parent-group support;
- candidate success in both phases.

This is evidence that the local empirical frontier moved.  It does not require
zero single-state core regressions.

### B. Candidate policy realization

The v1 engineering proxy is fixed locked-panel coverage.  Prospective automatic
selection uses method-level non-inferiority margins:

```text
maximum global Tube coverage drop: 5 percentage points
maximum per-phase Tube coverage drop: 10 percentage points
```

These margins are fixed method values, not candidate-specific tuning.

A candidate becomes the sole next automatic policy authority only when both A and
B pass.

This prevents a large phase collapse from being hidden by Tube cardinality
imbalance while no longer demanding exact reproduction of every stochastic
single rollout.

### Current pi_2 under the revised semantics

The current result is a **retrospective** method reinterpretation because the new
criterion was defined after observing pi_2.

Expected classification:

```text
empirical_envelope_expansion_observed = true
global policy-realization margin      = pass
downstream phase margin               = pass
upstream phase margin                 = fail
candidate_policy_authority_eligible   = false
```

A retrospective decision artifact may document this evidence but may not select
pi_2 formally.  `select_iteration_policy.py` enforces that rule.

## Automatic workflow semantics

`JIT/cli/prepare_iterative_envelope_workflow.py` now generates the future generic
DAG as:

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
       A frontier progression
       B phase-aware policy realization
  -> select pi_(k+1) only if A + B pass prospectively
```

The workflow still never auto-tunes a failed candidate.  A failed policy
realization decision stops and returns control to a scientific method decision.

The current pi_1 -> pi_2 round was **not** a clean end-to-end automatic run:

- frontier acquisition required v3/v3b scientific repairs;
- C_up^1 required an explicit engineering 64x64 selection despite AUC < 0.70;
- all-role near-observation isolation required an explicit engineering
  continuation record after TRAIN <-> ACCEPTANCE near-overlap was confirmed zero.

Do not claim full prospective automation for this completed round.

## What JIT means going forward

Use this one-sentence definition:

> JIT is an iterative real-dynamics capability-discovery and just-in-time
> curriculum framework that accumulates empirical jump-capability evidence under
> fixed robot dynamics, uses the frontier to train a single unified policy, and
> separately measures frontier progression and how much of the cumulative
> capability that policy can realize.

Important consequences:

- earlier successful capability evidence is not erased because a later policy
  fails one paired rollout;
- the latest policy is not the definition of physical feasibility;
- C^k remains policy-conditioned and is a proposal/filter tool, not a proof of
  existential controllability;
- a future policy archive may be used for **discovery only** without changing the
  single-policy runtime requirement;
- final physical-envelope/JCE/JEL claims still require an untouched final
  evaluation after method and stopping decisions are frozen.

## Next scientific work

Do **not** automatically run a 90/10 replay repair merely because pi_2 has core
regressions.

The next method decision should address the deeper representation problem exposed
by pi_2:

1. generate the retrospective capability-progression artifact for pi_2;
2. preserve pi_2 as frontier/capability evidence but keep pi_1 as the currently
   selected engineering authority until a prospective next-authority decision is
   made;
3. evaluate a goal-/intent-conditioned unified policy so one Actor can express
   different desired jump behaviors explicitly;
4. upgrade future policy coverage from one rollout per state to a predeclared
   multi-seed success-probability/confidence evaluation;
5. consider a frozen policy archive for discovery-time frontier probing while
   keeping deployment single-policy;
6. only after the next method version is predeclared should another candidate or
   pi_3 iteration begin.

## Immutable physical/task contracts

- Work on `agent/two-phase-soft-tube`; do not modify `main` unless explicitly authorized.
- XML: `assets/orange_bike_4kg_horizontal.xml`.
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`.
- Actual payload: 2 kg.
- Control rate: 50 Hz.
- Hip/knee torque limits: +/-50 Nm.
- Action order: `[steer, rear-wheel drive, hip, knee]`.
- Unified runtime never switches experts.
- Do not silently change physics, reward semantics, action semantics, snapshot
  semantics, task geometry, collision geometry, or final TEST isolation.

## Data-role and claim isolation

- `TRAIN`: may fit `C^k` and contribute qualifying Tube expansion.
- `CALIBRATION`: threshold calibration only; never enters TRAIN/Tube.
- `ACCEPTANCE`: candidate-blind development comparison only; never trains C^k or
  enters a Tube.
- final TEST/JCE/JEL: untouched until method, final policy, and stopping decision
  are frozen.

Parent-group disjointness remains the primary role-separation contract.  Current
round engineering near-observation overlap is historical evidence, not a reason
to silently relax future automatic isolation.

## Repository-maintenance policy

1. Modify/consolidate existing production code first.
2. New production files require a genuinely new durable capability.  The new
   capability-progression analyzer qualifies because it changes stable decision
   semantics, not because it is called `pi_2`.
3. Iteration/run identity belongs in config/artifact metadata, not source-module
   names.
4. Keep `JIT/cli/` thin; reusable logic belongs under `JIT/src/jit_dvgc/`.
5. Preserve path-bound configs/frozen manifests/provenance.
6. Never delete Python/CLI/tests without proving dependency closure and running
   compile/import/targeted tests.

## Git and local-work safety

- Preserve unrelated user changes.
- Never reset, clean, stash, rebase, force-push, or overwrite unrelated work.
- Use `/home/qy/mujoco_playground/.venv/bin/python`.
- Keep formal runs/checkpoints/logs out of Git.
- Use focused commits.

## Current authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
5. `JIT/docs/CODEX_HANDOFF_20260904.md`
6. `PROJECT.md`
7. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
8. `JIT/docs/CODE_ORGANIZATION.md`

`JIT/docs/CODEX_HANDOFF_20260903.md` is historical/superseded and must not be used
as current operational truth.
