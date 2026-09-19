# 2026-09-02 Iteration-1 Repair Handoff

## Purpose

This document is the end-of-day handoff for the active JIT/DVGC envelope
iteration work on 2026-09-02.

It records:

- what was completed today;
- why the project spent time on fresh acceptance-bank acquisition before
  repaired `pi_1` training;
- which attempts are consumed evidence and must not be rerun/retuned;
- which failures were scientific versus engineering;
- the current authoritative route;
- the exact next scientific task.

For the compact resume marker read `docs/EXPERIMENT_STATE.md`. For the complete
current ledger read `JIT/docs/CURRENT_STATUS.md`.

## Start-of-day state

At the start of the day:

1. Tube_1 was complete with 3,119 TRAIN entries = 222 retained Tube_0 core +
   2,897 expansion states.
2. The first completed Tube_1 `pi_1` candidate had been frozen.
3. Its formal paired `pi_0 -> pi_1` gate had completed:
   - core preservation FAIL: 21 regressions;
   - boundary gain PASS: 12 gains across 5 parent groups;
   - iteration rejected.
4. A zero-interaction diagnosis showed retained-core replay dilution was a
   material mechanism.
5. The repaired iteration-1 method was predeclared as 50% retained core / 50%
   expansion inside each phase, with all other policy-training variables fixed.
6. The old 56-state boundary bank had been consumed by the rejected-candidate
   decision and diagnosis, so it could not be the sole fresh acceptance evidence
   for the repaired candidate.

The morning blocker was therefore **fresh pre-training acceptance-bank
readiness**, not PPO training.

## Scientific route followed today

```text
old rejected pi1
  -> diagnose core regression
  -> predeclare replay repair
  -> generate fresh pi0-negative acceptance evidence BEFORE repaired pi1
  -> if fresh bank PASS: train repaired pi1 once
  -> freeze repaired pi1
  -> paired acceptance gate
```

The project deliberately did not train repaired `pi_1` until the fresh bank was
ready.

## Step 1 — support-wide fresh readiness probe

Local acquisition:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_acquisition_20260902`

Local labels/readiness:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_labels_20260902`

Result:

- 659 fresh TRAIN candidates;
- 647 frozen-`pi_0` continuation successes;
- 12 frozen-`pi_0` continuation negatives;
- upstream: 12 negative states, only 2 negative parent groups;
- downstream: 0 negative states, 0 negative parent groups;
- required per phase: >=10 negative states and >=3 negative parent groups;
- **READINESS FAIL**.

Classification: **scientific/pre-training readiness failure**.

It is not an engineering error and must not be converted to PASS by lowering the
readiness rule.

This evidence is consumed.

## Step 2 — stronger single-axis shell

Because the first shell produced no downstream negatives, a stronger
real-dynamics single-axis shell was predeclared without repaired-candidate
information.

Config:

`JIT/configs/envelope_iter1_repair_acceptance_boundary_acquisition_extended_shell.json`

Protocol used:

- 10 upstream + 10 downstream parent anchors;
- strengths: 0.15 / 0.30 / 0.50;
- durations: 2 / 4 / 8 ticks;
- all 4 action axes and both signs;
- frozen `pi_0` only;
- no validation/TEST/training.

Local acquisition:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_extended_shell_acquisition_20260902`

Result:

- 1,272 fresh candidates;
- upstream 556 / downstream 716;
- acquisition interactions: 6,391 / 6,720;
- zero exact-state overlap with the first 659-state probe.

Frozen-`pi_0` labeling result:

- 58 negatives;
- upstream: 58 negative states but still only 2 parent groups;
- downstream: 0 negatives;
- **READINESS FAIL**.

Classification: **scientific/pre-training readiness failure**.

Important inference: increasing single-axis perturbation magnitude/duration did
not solve downstream boundary discovery. The next change had to target direction
family rather than blindly increasing shell strength again.

This evidence is consumed.

## Step 3 — zero-interaction mechanism diagnosis

The 1,272 catalog rows were joined with their frozen-`pi_0` labels without any
new environment interactions.

Key conclusion:

- all 716 downstream candidates succeeded;
- even the hardest downstream continuations were far from the 400-tick horizon;
- the failure pattern therefore did not support the hypothesis that the same
  single-axis direction family only needed still larger strength/duration;
- the acquisition family itself was insufficiently coupled for downstream
  recovery-boundary discovery.

A diagnostic display bug was also identified: grouping only by
`parent_group_id` could overwrite phase identity when the same parent identifier
appeared in both phases. The actual experimental result remained downstream
negative count = 0.

## Step 4 — production acquisition generalization

The existing acquisition capability was generalized rather than creating a new
iteration-specific module.

Implemented principles:

- configurable sparse action directions;
- `active_action_dimensions=1` retains historical one-axis behavior;
- `active_action_dimensions=2` systematically covers action pairs and sign
  combinations;
- iteration contract generalized from hard-coded `0 -> 1` to `k -> k+1`;
- real `env.step` dynamics retained;
- no direct `qpos/qvel` synthesis;
- no validation/TEST use.

This is reusable for later `pi_1 -> pi_2`, `pi_2 -> pi_3`, etc.

## Step 5 — two-axis acceptance acquisition

Predeclaration:

`JIT/configs/envelope_iter1_repair_acceptance_boundary_acquisition_two_axis.json`

Direction family:

- 4 actions;
- all `C(4,2)=6` action pairs;
- 4 sign combinations per pair;
- 24 sparse two-axis directions;
- strengths 0.15 / 0.30 / 0.50;
- durations 2 / 4 / 8;
- 10 upstream + 10 downstream parent anchors.

Local acquisition:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_two_axis_acquisition_20260902`

Completed result:

- 3,720 unique fresh TRAIN candidates;
- upstream: 1,560;
- downstream: 2,160;
- real acquisition interactions: 18,829 / 20,160 ceiling;
- exact-state overlap with both previous readiness probes: 0;
- training transitions: 0;
- validation/TEST/final data: none.

Classification: **acquisition PASS**.

Do not rerun this acquisition. It is the immutable source catalog for the fresh
acceptance bank.

## Step 6 — labeling CUDA/Warp engineering failure

Long single-process frozen-`pi_0` labeling failed with CUDA OOM inside Warp/MJX
collision allocation. The failure occurred after many serial continuation steps;
it was not a scientific negative/readiness result.

The code already reused a single compiled `jax.jit(env.step)`, so the earlier
paired-gate repeated-JIT-wrapper fix did not apply.

A Warp memory-maintenance attempt was added, including synchronization/mempool
maintenance telemetry. It did not fully solve the process-lifetime memory
pressure: long single-process labeling still exhausted device memory.

Classification: **engineering failure**.

Do not reduce:

- candidate count;
- 400-tick horizon;
- GPU use;
- task physics;
- scientific thresholds;
- labeling semantics.

The scientific 3,720-state catalog remained valid.

## Step 7 — process-sharded labeling recovery

To avoid wasting the completed acquisition, the same logical labeling job was
executed as four sequential independent Python/CUDA/Warp processes:

- shard 0: 930 candidates;
- shard 1: 930 candidates;
- shard 2: 930 candidates;
- shard 3: 930 candidates.

Each process used the same:

- frozen `pi_0`;
- exact catalog states;
- deterministic policy mode;
- protocol seed;
- 400-tick horizon;
- continuation success/failure semantics;
- physics and action semantics.

After each process exited, the CUDA/Warp process context was released. The four
outputs were merged in original global catalog order before one logical bank lock.

Merged root:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_two_axis_sharded_20260902/merged`

Scientific result:

- merged labels: 3,720;
- frozen-`pi_0` negatives: 260;
- upstream: 246 negatives across 4 parent groups;
- downstream: 14 negatives across 5 parent groups;
- Tube_1 overlap: 0;
- **fresh acceptance-bank readiness PASS**.

Classification: **scientific readiness PASS after engineering-equivalent
execution sharding**.

Important rule for future use: process sharding is allowed only as a non-adaptive
execution partition of one predeclared logical job. It must not change candidate
selection, seed, horizon, policy, label semantics, bank rule, or thresholds.

## Step 8 — repaired pi_1 formal training

After fresh-bank readiness passed, exactly one repaired iteration-1 policy run
was launched.

Config:

`JIT/configs/pi_unified_iter1_tube1_core_replay50_natural10.json`

Formal run:

`JIT/runs/pi_unified/pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902`

Training contract:

- 10,009,600 PPO transitions;
- fresh actor / critic / optimizer;
- seed 821101;
- Tube_1 support unchanged;
- 50% upstream / 50% downstream phase mixture;
- inside phase: 50% retained core / 50% expansion;
- outer reset mixture: 90% Tube / 10% natural;
- effective episode reset mass: 45% retained core / 45% expansion / 10% natural;
- no validation;
- no TEST;
- no expert switching.

Result: **training complete**.

Known final checkpoint payload SHA-256:

`ea93a534c2c6bb3bf145684cbea82df94fefa2df8099dcdcdd9492bd8007e205`

## Step 9 — repaired pi_1 freeze

Frozen candidate:

`JIT/runs/frozen_unified/pi_1_core_replay50_10009600_20260902/frozen_unified_policy.json`

Known frozen manifest file SHA-256:

`d5a1658530d475a67264aa5c621283d71c823200dbee6068f93413b93d06b7a8`

Freeze completion does **not** mean iteration acceptance.

The repaired policy is currently a frozen candidate comparison authority.

## Current state at end of day

```text
Tube_0 = complete
pi_0 = frozen accepted iteration-0 authority
C^0 = complete
Tube_1 = complete
first pi_1 = rejected
replay repair = predeclared and implemented
fresh acceptance bank = PASS, 260 states
repaired pi_1 = trained 10,009,600 transitions
repaired pi_1 = frozen
formal repaired paired gate = NOT YET RUN
C^1 = NOT AUTHORIZED
Tube_2 = NOT AUTHORIZED
pi_2 = NOT AUTHORIZED
TEST/JCE/JEL = untouched
```

## Exact next scientific task

Run exactly one formal repaired `pi_0 -> pi_1` paired acceptance gate.

### Policies

Baseline:

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

Candidate:

`JIT/runs/frozen_unified/pi_1_core_replay50_10009600_20260902/frozen_unified_policy.json`

### Core bank

Use all 222 Tube_0 core states.

Core rule is unchanged:

**PASS only if baseline-success -> candidate-failure regressions = 0.**

### Fresh boundary bank

Use the locked 260-state fresh two-axis frozen-`pi_0` negative bank from the
merged sharded labeling artifact.

The gate implementation/config must resolve each challenge snapshot through the
correct original two-axis acquisition provenance root rather than assuming that
snapshots live inside the merged label directory.

Boundary rules are unchanged from the original paired gate:

1. every locked baseline-negative challenge must reproduce as a baseline failure;
2. repaired `pi_1` must succeed on states from at least 2 distinct parent groups.

Do not increase or decrease `2` after seeing repaired-policy outcomes.

### Gate isolation

- deterministic continuation;
- 400-tick horizon;
- training transitions = 0;
- expert switching = false;
- validation = false;
- TEST/final = false.

## Decision after the next gate

### If core PASS and boundary PASS

Iteration 1 becomes accepted.

Then proceed in this order:

1. accept frozen repaired `pi_1` as the iteration-1 authority;
2. collect/freeze `pi_1` TRAIN boundary evidence under real dynamics;
3. fit `C_up^1` and `C_down^1` using the already frozen continuation
   architecture (`76 -> 8 tanh -> 1`); do not architecture-search again;
4. run a new independent fresh validation/calibration split;
5. build core-retaining `Tube_2` from retained Tube_1 core + accepted TRAIN
   expansion;
6. predeclare and train `pi_2` using generic iteration code;
7. freeze and gate `pi_2`;
8. repeat until a predeclared stopping criterion is met;
9. only then open final TEST/JCE/JEL.

### If either gate FAILS

1. preserve the completed gate artifact;
2. classify scientific failure separately from engineering failure;
3. diagnose with zero-interaction evidence first;
4. do not alter the consumed bank or gate threshold;
5. do not sweep replay ratios, PPO settings, reward, physics, network, or
   acceptance rule against the same gate;
6. predeclare any next repair before new candidate outcomes.

## Remaining code debt before unattended later iterations

The following are not blockers for the repaired iteration-1 gate itself, but they
must be resolved before fully automatic later iterations:

- `core_retaining_tube_iteration.py` still contains Tube_1 / iteration-0
  constants;
- shared continuation refit/fresh validation still depend on some
  upstream-specific evidence/CV helpers;
- generic `C^k -> Tube_(k+1)` construction remains incomplete;
- paired-gate input/provenance needs to consume the new fresh locked bank cleanly;
- workflow automation must stop on scientific failure and never auto-tune.

## Do not repeat tomorrow

Do **not**:

- rerun the 659-state support-wide probe;
- rerun the 1,272-state extended single-axis probe;
- rerun the 3,720-state two-axis acquisition;
- overwrite any completed/failed labeling directory;
- retrain the repaired `pi_1` before its gate;
- reuse the old 56-state bank as the sole repaired-candidate acceptance bank;
- lower readiness/gate thresholds;
- start `C^1`, `Tube_2`, or `pi_2` before paired-gate acceptance;
- touch final TEST/JCE/JEL.

## Runtime-artifact note

`JIT/runs/` is normally ignored by Git. The local runtime paths and hashes in
this handoff are provenance references, not claims that the large runtime files
are committed to the repository. Preserve the local artifacts until the
scientific chain is complete and all required provenance has been extracted.
