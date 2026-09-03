# Current JIT status — 2026-09-03

## Executive state

The project has finished the Iteration-1 initialization/replay ablations and is now authorized to proceed on the main JIT loop:

```text
Tube_0
  -> pi_0 frozen
  -> C^0
  -> Tube_1
  -> pi_1 candidate study complete
  -> repair02 selected as engineering pi_1 authority
  -> CURRENT: pi_1-conditioned frontier / C^1 / Tube_2 / pi_2
```

The selected engineering policy is **repair02**:

- frozen policy: `JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`
- actor SHA-256: `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`
- payload SHA-256: `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`
- Tube_0 retention: **222 / 222**
- upstream core: **117 / 117**
- downstream core: **105 / 105**
- historical 260-state boundary quickcheck: **26 / 260**, 4 parent groups

**Important claim boundary:** repair02 is selected for engineering continuation of the JIT iteration, but the historical Iteration-1 quickcheck is **not** claimed as a strict formal acceptance PASS. Its boundary report contains 3 baseline-reproduction failures caused by the historical PRNG/baseline-reproduction protocol mismatch. Register repair02 with the explicit engineering-quarantine flag so this distinction is machine-readable.

Final TEST/JCE/JEL remains untouched.

---

## Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- only active branch: `agent/two-phase-soft-tube`
- local repository: `~/DVGC`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control rate: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`
- unified policies never use expert switching
- validation/calibration/acceptance rows never enter a Tube
- TEST/final evidence may not influence Tube construction, threshold selection, policy training, checkpoint selection, or iteration stopping

The Soft Tube remains empirical training guidance/capability support. It is **not** a certified safe set, viability kernel, or formal invariant set.

---

## Stable bootstrap chain

### Frozen experts

`pi_up_star`

- 9,977,856 transitions
- actor SHA-256: `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`

`pi_down_star`

- 25,600 transitions
- actor SHA-256: `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`

Frozen manifest:

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

- TRAIN core: 222 = 117 upstream + 105 downstream
- manifest SHA-256: `c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`

### pi_0

Frozen authority:

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- 10,009,600 PPO transitions
- actor SHA-256: `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`
- payload SHA-256: `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`

### C^0 and Tube_1

`C_up^0/C_down^0` were trained from frozen-pi_0 continuation evidence and passed fresh validation. Tube_1 is already complete:

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

- retained Tube_0: 222
- expansion: 2,897
- total: 3,119
- upstream: 427 = 117 core + 310 expansion
- downstream: 2,692 = 105 core + 2,587 expansion
- manifest SHA-256: `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`
- entries SHA-256: `61c6796aaf4c4b1e43624c5cf06bce0d39736a6d1743c5142c6c250d23155ec9`

Tube_1 is a true core-retaining superset of Tube_0.

---

## Iteration-1 policy study — closed

Do not continue A/B or early-stop sweeps. The study is complete.

| policy/checkpoint | Tube_0 core | regressions | upstream | downstream | boundary | groups |
|---|---:|---:|---:|---:|---:|---:|
| **repair02** | **222/222** | **0** | **117/117** | **105/105** | 26/260 | 4 |
| B 1.024M | 217/222 | 5 | 112/117 | **105/105** | 33/260 | 3 |
| B 2.5088M | 206/222 | 16 | 101/117 | **105/105** | 28/260 | 4 |
| B 5.0176M | 214/222 | 8 | 109/117 | **105/105** | 25/260 | 4 |
| **B 7.5008M** | **217/222** | **5** | **112/117** | **105/105** | **42/260** | **4** |
| B 10.0096M | 212/222 | 10 | 107/117 | **105/105** | **46/260** | 4 |

No B checkpoint satisfies both:

```text
Tube_0 = 222/222
and
boundary > 26/260
```

Therefore:

1. `repair02` is the selected pi_1 engineering authority.
2. Warm-start A is discarded.
3. Warm-start B is retained only as an ablation/scientific diagnostic.
4. Do not spend more training/evaluation budget on B checkpoint selection.

### Scientific interpretation of B

B forgetting is not a monotonic function of training time:

```text
regressions: 5 -> 16 -> 8 -> 5 -> 10
```

Every B checkpoint retains downstream core at `105/105`; all core regressions occur upstream. Boundary gains are also overwhelmingly upstream.

The supported interpretation is therefore:

> Learning the new upstream boundary and retaining the old upstream control behavior interfere under naive pi_0 full warm-start. This is upstream expansion/retention policy interference, not a general descent/recovery failure and not simple monotonic overtraining.

Future method work may study warm-start plus explicit retention constraints (for example distillation/KL anchoring/constrained replay), but this is **not** part of the current mainline iteration.

---

## Historical Iteration-1 gate claim boundary

Historical repair02 quickcheck:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_repair02_quickcheck_20260903/summary.json`

Engineering result:

- core: 222/222, zero regression
- boundary candidate successes: 26/260
- successful parent groups: 4
- baseline reproduction failures: 3

The three reproduction failures come from a historical protocol mismatch between continuation-label PRNG hierarchy and the paired-gate re-roll hierarchy. They are **not** repair02 core regressions.

Use this language consistently:

- **engineering selection:** repair02 is selected as pi_1 and may generate C^1 / Tube_2 / pi_2;
- **publication/formal claim:** historical Iteration-1 strict acceptance is not claimed PASS under the old gate protocol.

`JIT/cli/select_iteration_policy.py --allow-baseline-reproduction-mismatch` exists specifically to encode this quarantine without rewriting history.

---

## Generic automatic k -> k+1 pipeline — implemented

The active branch now contains an iteration-generic workflow for `k >= 1`.

Operator entry point:

```bash
python JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute
```

`JIT/cli/prepare_iterative_envelope_workflow.py` generates one resumable `jit_iteration_workflow_v1` DAG with the following fixed stages:

```text
selected pi_k + Tube_k
  -> predeclare newest-shell frontier roles
  -> TRAIN frontier acquisition/labels
  -> disjoint CALIBRATION frontier acquisition/labels
  -> disjoint ACCEPTANCE frontier acquisition/labels
  -> fit + calibrate C_up^k / C_down^k
  -> build core-retaining Tube_(k+1)
  -> Tube-RSI smoke
  -> role-isolation audit
  -> lock exact pi_k core/boundary baseline before candidate training
  -> prepare pi_(k+1) fresh training config
  -> train pi_(k+1)
  -> freeze exact final checkpoint
  -> strict locked-baseline core/boundary gate
  -> select pi_(k+1) only if the strict gate passes
```

### Data-role contract

The frontier plan is outcome-blind and parent-disjoint before outcomes are observed:

- `train`: may fit C^k and contribute candidate Tube expansion states;
- `calibration`: threshold calibration only; never embedded in the Tube;
- `acceptance`: pre-candidate locked baseline audit only; never embedded in the Tube.

The automatic frontier probes **only the newest expansion shell of Tube_k**. It does not silently fall back to the full Tube if there is no outward shell.

### Tube_(k+1) retention contract

`JIT/src/jit_dvgc/iterative_tube.py` implements:

```text
Tube_(k+1)
  = every Tube_k entry retained exactly
  + new logical-TRAIN states
    with positive pi_k continuation label
    and C^k score strictly above the disjoint calibration threshold
```

For the immediate next round this means:

```text
Tube_2 retains all 3,119 Tube_1 states,
not merely the original 222 Tube_0 states.
```

The generated pi_2 training config then treats the whole Tube_1 support as the retained core and uses the selected repair02-style strong replay contract (75% retained core / 25% newest expansion inside Tube sampling, plus the existing 90% Tube / 10% natural outer mixture).

### Future strict gate fixes the historical PRNG debt

`JIT/src/jit_dvgc/iterative_acceptance_gate.py` changes future acceptance semantics deliberately:

1. pi_k core outcomes are evaluated and locked **before** pi_(k+1) training;
2. acceptance-role pi_k negatives are locked with their exact labeling PRNG identity;
3. the baseline boundary is not re-rolled after candidate training;
4. candidate core uses the exact same locked core seeds;
5. candidate boundary uses the exact same `labeling_seed -> candidate_index -> tick` PRNG hierarchy.

Therefore future `pi_k -> pi_(k+1)` gates do not depend on re-producing a historical negative with a different random-key protocol. A future workflow advances only when the strict gate artifact reports zero core regressions and sufficient boundary gain.

---

## Immediate operator launch: repair02 -> C^1 -> Tube_2 -> pi_2

### 1. Update the working tree

```bash
cd ~/DVGC
git fetch origin
git checkout agent/two-phase-soft-tube
git pull --ff-only origin agent/two-phase-soft-tube
git status --short

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python
```

Do not reset/clean/stash unrelated user files.

### 2. Register repair02 as selected engineering pi_1

```bash
$PY JIT/cli/select_iteration_policy.py \
  --frozen-policy JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json \
  --gate-summary JIT/runs/pi_unified_gate/pi_0_to_pi_1_repair02_quickcheck_20260903/summary.json \
  --output-dir JIT/runs/iteration_selection/pi_1_repair02_selected_20260903 \
  --allow-baseline-reproduction-mismatch
```

Required artifact:

`JIT/runs/iteration_selection/pi_1_repair02_selected_20260903/selected_policy.json`

It must record:

```text
engineering_selection = true
formal_acceptance_claim = false
baseline_reproduction_mismatch_quarantined = true
```

### 3. Generate the pi_1 -> pi_2 workflow

```bash
$PY JIT/cli/prepare_iterative_envelope_workflow.py \
  --selected-policy JIT/runs/iteration_selection/pi_1_repair02_selected_20260903/selected_policy.json \
  --source-tube JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901 \
  --tag 20260903 \
  --work-root JIT/runs/iteration_auto/pi_1_to_pi_2_20260903 \
  --config-out JIT/runs/iteration_auto/pi_1_to_pi_2_20260903/workflow.json
```

### 4. Dry-run the resolved plan

```bash
$PY JIT/cli/run_iteration_workflow.py \
  --config JIT/runs/iteration_auto/pi_1_to_pi_2_20260903/workflow.json
```

No scientific stage is executed without `--execute`.

### 5. Start the formalized automatic round

```bash
$PY JIT/cli/run_iteration_workflow.py \
  --config JIT/runs/iteration_auto/pi_1_to_pi_2_20260903/workflow.json \
  --execute
```

The runner is resumable. Re-run the same command after an engineering interruption; completed stages are reused only after their declared completion artifacts are revalidated. The workflow config SHA is immutable after state creation.

If a scientific gate fails, the workflow stops. It must never auto-change thresholds, reward, replay ratio, PPO settings, physics, architecture, or acceptance criteria to force progress.

---

## Current automatic stopping/decision rules

For each later round:

```text
pi_k selected/frozen
  -> C^k
  -> Tube_(k+1)
  -> pi_(k+1)
  -> locked-baseline core preservation + boundary gain
```

Advance only if the strict gate passes. Stop and diagnose if:

- core regressions appear;
- acceptance frontier lacks required two-phase/parent-group support;
- C^k cannot pass its fixed calibration contract;
- Tube expansion vanishes;
- boundary gain fails;
- implementation/provenance/isolation checks fail.

Broader project-level stopping can later be declared for saturation, negligible Tube growth, inability to expand without retention loss, physical-envelope target reached, or resource budget reached. Do not iterate merely because another round is mechanically possible.

---

## Repository implementation authority

Stable capabilities now live under the package APIs:

- `jit_dvgc.training` — unified PPO / formal preflight / policy freezing
- `jit_dvgc.tube` and `iterative_tube.py` — Soft Tube / Tube-RSI / iterative core retention
- `jit_dvgc.acquisition` — real-dynamics boundary acquisition
- `jit_dvgc.continuation` and `iterative_continuation_fields.py` — continuation labels/fields/calibration
- `jit_dvgc.analysis` — paired diagnostics/gates
- `jit_dvgc.workflow` — resumable orchestration
- `iterative_frontier_protocol.py` — outcome-blind train/calibration/acceptance role generation
- `iterative_acceptance_gate.py` — pre-candidate locked baseline and future strict gate

Contract regression coverage for the new generic path is in:

`JIT/tests/test_iterative_envelope_automation.py`

It covers newest-shell-only frontier behavior, full source-Tube retention with TRAIN-only expansion, and workflow stop-on-scientific-gate-failure semantics.

---

## Do not reopen these routes

Unless a later accepted iteration produces new evidence requiring a scientific decision, do not:

- rerun warm-start A;
- continue B checkpoint sweeps;
- tune B early stopping against the already-consumed quickcheck bank;
- return to linear C_up or architecture ladders;
- inject validation/calibration/acceptance/TEST rows into a Tube;
- make natural cold-start performance the main JIT blocker;
- rewrite reward/physics/task semantics to make a gate pass;
- claim the historical repair02 gate was formally PASS;
- use final TEST/JCE/JEL before final frozen-policy selection.

## Current one-line authority

> **repair02 is selected pi_1 for engineering continuation; historical Iteration-1 formal PASS remains quarantined; the active mainline is now `pi_1 -> C^1 -> Tube_2 -> pi_2` through the resumable locked-baseline automatic workflow.**
