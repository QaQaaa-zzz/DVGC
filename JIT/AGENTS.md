# JIT agent and maintenance rules

## Scope and safety

- `JIT/` is the active implementation area. Treat repository-root `dvgc/`,
  `cli/`, `scripts/`, and `tests/` as read-only unless the user explicitly
  changes scope.
- Work only on `agent/two-phase-soft-tube` unless explicitly told otherwise.
- Never reset, clean, stash, rebase, force-push, overwrite, or reformat unrelated
  user work.
- Use only `/home/qy/mujoco_playground/.venv/bin/python`; do not reinstall or
  reconfigure the environment.
- Preserve the fixed task identity: `assets/orange_bike_4kg_horizontal.xml`, SHA
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`,
  2 kg payload, 50 Hz control, hip/knee +/-50 Nm, action order
  `[steer, rear-wheel drive, hip, knee]`.
- Unified runtime policies never switch experts.
- Final TEST/JCE/JEL evidence remains untouched until method, stopping rule, and
  final policy are frozen.

## Scientific contract — revised 2026-09-04

JIT is no longer defined as “the newest policy must reproduce every prior success
exactly.”  Keep these objects separate:

- `F*`: conceptual fixed-task physical feasibility; not proven by JIT;
- `E_k`: cumulative empirical capability evidence gathered through real dynamics;
- `Tube_k`: TRAIN-only structured support/curriculum derived from that evidence;
- `R(pi_k, Tube_k)`: how much of the cumulative support one unified policy
  realizes on a locked evaluation panel.

The project remains a single-policy deployment project, but the newest policy is
a **capability probe + realization candidate**, not the definition of physical
feasibility.

Consequences:

- a later policy's failed single rollout does not erase earlier successful
  capability provenance;
- strict paired regressions remain useful diagnostics;
- envelope progression and policy realization must be reported separately;
- a candidate can demonstrate new frontier capability yet still be unsuitable as
  the sole next-iteration policy authority because one phase loses too much
  coverage;
- Tube growth is not physical state-space volume and not a safety proof.

## Current authority

Read:

`JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`

before reconstructing the current chain.

Current completed state:

```text
experts
  -> Tube_0 222
  -> pi_0
  -> C^0
  -> Tube_1 3,119
  -> pi_1 repair02
  -> v3/v3b/v3c frontier roles
  -> C^1 64x64 engineering selection
  -> Tube_2 3,776
  -> pi_2 trained/frozen
  -> locked pi_1 vs pi_2 comparison complete
  -> CURRENT: frontier progression observed; upstream policy realization degraded
```

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

- retained Tube_1: 3,119;
- new expansion: 657;
- total: 3,776;
- upstream: 902 = 427 + 475;
- downstream: 2,874 = 2,692 + 182;
- manifest SHA:
  `135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`.

### Current pi_2 evidence

Training completed at 10,009,600 transitions using the declared 90% Tube / 10%
natural outer reset and 75% retained Tube_1 / 25% Tube_2 expansion inner replay.

Locked source-Tube panel:

```text
pi_1 3115/3119
pi_2 3002/3119
```

Phase coverage:

```text
upstream:   pi_1 423/427 -> pi_2 312/427
downstream: pi_1 2692/2692 -> pi_2 2690/2692
```

Locked frontier challenge:

```text
13/14 pi_2 successes
3 parent groups
upstream 4/5
downstream 9/9
baseline reproduction failures 0
```

Meaning:

- empirical local frontier progression: yes;
- upstream single-policy realization collapse: yes;
- current pi_2 retrospective promotion to formal next authority: no.

## Capability-progression decision API

Stable implementation:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

CLI:

`JIT/cli/analyze_capability_progression.py`

Future prospective v1 contract:

### Frontier progression

Require:

- zero boundary baseline-reproduction mismatch;
- nonzero candidate success;
- minimum parent-group diversity;
- candidate success in both phases.

### Policy realization

Fixed-panel non-inferiority proxy:

```text
maximum global coverage drop = 0.05
maximum per-phase coverage drop = 0.10
```

Zero individual regressions are **not** required.

A prospective candidate is eligible to become the sole next policy authority only
when both frontier progression and policy realization pass.

The current pi_2 result is retrospective with respect to this revised criterion,
so it may be analyzed but may not be formally selected using that retrospective
artifact. `select_iteration_policy.py` enforces this.

## Historical strict gate compatibility

Do not delete or rewrite historical strict artifacts.

- repair02/pi_1 historical selection remains reproducible through the old
  zero-regression path;
- its 3 boundary reproduction mismatches remain quarantined technical debt;
- future automation uses the new capability-progression stage after the locked
  paired evaluation.

Do not rewrite old reports to make them appear to have used the new semantics.

## Continuation semantics

- `V_up/V_down`: bootstrap expert-conditioned continuation only.
- `C_up^k/C_down^k`: exact-policy-conditioned continuation evidence for frozen
  `pi_k`; useful for frontier proposal/filtering, not existential controllability.
- PPO critic/value is not a JIT continuation field.
- The same state may fail under one policy and succeed under another.

Current C^1 claim boundary:

- 64x64 architecture selected engineering-mainline;
- upstream AUC 0.6903137789904502 remains below the original 0.70 formal gate;
- downstream AUC 1.0 formally passes;
- do not state that C^1 passed all original formal calibration rules.

## Data-role contract

- `TRAIN`: fit continuation models and contribute qualifying Tube expansion;
- `CALIBRATION`: threshold calibration only; never embedded in Tube;
- `ACCEPTANCE`: development frontier comparison only; never trains/calibrates C
  and never enters Tube;
- final TEST/JCE/JEL: untouched.

Parent-group disjointness remains mandatory.

Current Iteration-1 -> 2 engineering isolation evidence recorded:

```text
exact overlaps: 0
parent-group overlaps: 0
TRAIN <-> ACCEPTANCE near overlap at atol 0.01: 0
TRAIN <-> CALIBRATION near overlap: 140
CALIBRATION <-> ACCEPTANCE near overlap: 157
```

This historical engineering continuation does not silently relax the generic
future automatic isolation gate.

## Automatic iteration

Future generated DAG:

```text
selected pi_k + Tube_k
  -> newest-shell frontier plan
  -> TRAIN
  -> CALIBRATION
  -> ACCEPTANCE
  -> C^k
  -> Tube_(k+1)
  -> Tube-RSI smoke
  -> role-isolation audit
  -> lock pi_k baseline before candidate training
  -> train/freeze pi_(k+1)
  -> locked paired panel evaluation
  -> capability-progression analysis
       frontier progression
       phase-aware policy realization
  -> select pi_(k+1) only if both pass prospectively
```

Operator entry point remains:

`python JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute`

Automation rules:

- workflow config SHA is immutable after state creation;
- completed stages are revalidated, never silently rerun;
- failures stop the workflow;
- the workflow never auto-changes reward, replay ratio, PPO settings, model
  architecture, physics, acquisition panel, or capability decision thresholds;
- no final TEST/JCE/JEL stage is present;
- `Tube_(k+1)` retains every source Tube entry exactly and may add only qualifying
  TRAIN expansion.

The current pi_1 -> pi_2 round required explicit engineering interventions at C^1
and role-isolation, so do not describe that historical round as fully automatic.

## Next scientific question

Do not reflexively launch a 90/10 replay repair.

The current evidence points to a deeper representation problem: the unified
policy is reward-guided but not explicitly told which jump behavior/target it
should realize.

Before another candidate/pi_3 round, predeclare the next method version. Priority
questions:

1. add a low-dimensional goal/jump-intent condition while preserving one runtime
   Actor;
2. replace one-rollout-per-state policy realization with multi-seed success-rate
   estimation/confidence intervals;
3. consider a frozen discovery-time policy archive so empirical system capability
   evidence can use older successful probes without runtime policy switching;
4. then decide whether another same-representation training repair is still worth
   testing.

## Modify-first repository policy

1. Modify/consolidate existing production files first.
2. New production files require a genuinely durable capability.
3. Iteration IDs, retry numbers, seeds, and candidate names belong in configs and
   artifacts, not module names.
4. Keep CLIs thin; reusable logic goes in `JIT/src/jit_dvgc/`.
5. Preserve path-bound provenance and frozen artifacts.
6. Deletion requires proven dependency closure plus compile/import/targeted tests.

## Stable package boundaries

- `jit_dvgc.training` — unified PPO/preflight/freeze
- `jit_dvgc.tube` — Soft Tube/Tube-RSI
- `jit_dvgc.snapshots` — snapshot formats/pools
- `jit_dvgc.acquisition` — real-dynamics frontier acquisition
- `jit_dvgc.continuation` — continuation labels/fields
- `jit_dvgc.analysis` — paired evaluation + capability progression
- `jit_dvgc.workflow` — resumable orchestration

## Current read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
5. `JIT/docs/CODEX_HANDOFF_20260904.md`
6. `PROJECT.md`
7. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
8. `JIT/docs/CODE_ORGANIZATION.md`

The 2026-09-03 handoff is superseded historical context only.
