# DVGC/JIT Codex Technical Handoff — 2026-09-03

This document is the long-form handoff for an agent taking over the active DVGC/JIT research line. It explains the scientific objective, implementation path, current evidence, code-control logic, exact next actions, stopping rules, and claim boundaries.

For live operational state, `JIT/docs/CURRENT_STATUS.md` remains the first source of truth. This handoff explains how the pieces fit together and what a new agent must preserve while continuing the project.

---

## 1. Executive summary

DVGC/JIT is an iterative **single-policy empirical jumping-capability-envelope identification** project for a single-track two-wheeled robot.

The project is **not** a two-expert deployment system. The two phase experts exist only to bootstrap the first empirical state support and continuation evidence. The deployable controller is always one unified Actor.

The active scientific loop is:

```text
phase experts
  -> bootstrap continuation evidence
  -> Tube_0
  -> unified pi_0
  -> pi_0-conditioned C^0
  -> Tube_1
  -> selected pi_1
  -> pi_1-conditioned C^1
  -> Tube_2
  -> pi_2
  -> strict capability gate
  -> if PASS: select pi_2 and repeat
```

As of 2026-09-03:

- Tube_0 is complete: 222 TRAIN states.
- pi_0 is frozen and authoritative for Iteration 0.
- C_up^0/C_down^0 passed independent validation/calibration.
- Tube_1 is complete: 3,119 TRAIN states = 222 retained Tube_0 + 2,897 expansion states.
- The Iteration-1 policy study, including A/B initialization and checkpoint sweeps, is closed.
- **repair02 is the selected engineering pi_1 authority.**
- repair02 preserves Tube_0 at 222/222 and has 26/260 successful historical boundary candidates across 4 parent groups.
- No warm-start B checkpoint preserves all 222 Tube_0 states while beating repair02's boundary result.
- The active mainline is now **repair02/pi_1 -> C^1 -> Tube_2 -> pi_2**.
- A generic resumable `k -> k+1` workflow is implemented for `k >= 1`.
- Final TEST/JCE/JEL evidence remains untouched.

The important historical qualification is that repair02 is an **engineering selection**, not a retroactive publication-level strict Iteration-1 PASS. The old quick gate contains 3 baseline-reproduction failures caused by an historical PRNG-protocol mismatch. That technical debt is quarantined rather than rewritten.

---

## 2. Repository and immutable task identity

Repository:

```text
QaQaaa-zzz/DVGC
```

Only active branch:

```text
agent/two-phase-soft-tube
```

Verified pre-handoff code baseline on 2026-09-03:

```text
99769986f209bc62c172e572807bb039fdfa6016
Generalize envelope iteration protocol after pi1 selection
```

The documentation changes that add/update this handoff will advance the branch HEAD, so do not use the SHA above as a future equality check. Use it as the code baseline that was reviewed before this handoff was written.

Local repository expected by the current tooling:

```text
~/DVGC
```

Python environment:

```text
/home/qy/mujoco_playground/.venv/bin/python
```

Authoritative task XML:

```text
assets/orange_bike_4kg_horizontal.xml
```

XML SHA-256:

```text
0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a
```

Immutable physical/task contracts:

- payload: 2 kg;
- control rate: 50 Hz;
- hip/knee torque limits: +/-50 Nm;
- action order: `[steer, rear-wheel drive, hip, knee]`;
- unified policies do not switch between phase experts at runtime;
- do not change task physics, reward meaning, action semantics, snapshot semantics, collision geometry, or task geometry during an iteration;
- the historical `4kg` token in the XML filename is not the current payload and is not a reason to rename or replace the XML;
- final TEST/JCE/JEL evidence must remain isolated until a final frozen policy and stopping decision are fixed.

The learned Soft Tube is empirical training/curriculum support. It is **not** a certified safe set, viability kernel, reachability proof, or formal invariant set.

---

## 3. Scientific objective

The research question is:

> Can a unified policy iteratively enlarge the empirically demonstrated state set from which it can continue a complete jump maneuver, while retaining competence on previously established support?

The project therefore optimizes two properties simultaneously:

1. **retention** — an accepted new policy must not lose previously established capability;
2. **expansion** — it must gain capability on previously failing frontier states.

A larger training Tube alone is not evidence of expansion. A higher PPO reward alone is not evidence of expansion. Training completion is not acceptance. A new policy becomes the next scientific authority only after the declared capability gate passes.

The eventual output is an **empirical policy-conditioned jumping capability envelope**, not a theorem of safety or invariance.

---

## 4. Core concepts and terminology

### 4.1 Phase experts

Two bootstrap experts are used:

- `pi_up_star`: Propulsion-Ascent expert;
- `pi_down_star`: Descent-Recovery expert.

They are data-generation/bootstrap authorities, not the final controller.

The physical transition near Apex/early descent is a handoff band in the maneuver, not a third expert.

### 4.2 Bootstrap fields V_up / V_down

`V_up` and `V_down` are expert-conditioned bootstrap continuation models used to construct Tube_0.

They must not be silently reused as the authority for later unified policies.

### 4.3 Unified policy pi_k

`pi_k` is the single unified Actor for iteration `k`.

For later iterations, the relevant continuation question is policy-conditioned:

```text
C(s | pi_k)
```

The same state can fail under `pi_k` and succeed under `pi_(k+1)`.

### 4.4 C_up^k / C_down^k

`C_up^k` and `C_down^k` are empirical continuation fields fitted to evidence generated by the exact frozen/selected `pi_k`.

The PPO critic/value function is **not** a JIT continuation field.

### 4.5 Tube_k

A Tube is a versioned empirical training support.

For later iterations the structural rule is:

```text
Tube_(k+1)
  = every Tube_k entry retained exactly
  + qualifying new logical-TRAIN expansion states
```

The next Tube is therefore a cumulative support, not a replacement level set.

### 4.6 Capability gate

After training `pi_(k+1)`, acceptance requires both:

```text
core preservation
+
boundary gain
```

For future rounds, this gate is based on a pre-candidate locked baseline so the baseline outcome is not re-rolled after the candidate has been trained.

---

## 5. Full implementation path

The end-to-end research path is:

```text
Propulsion-Ascent expert pi_up
        +
Descent-Recovery expert pi_down
        |
        v
freeze experts
        |
        v
real handoff / continuation evidence
        |
        v
bootstrap expert-conditioned V_up / V_down
        |
        v
TRAIN-only Tube_0
        |
        v
Tube-RSI unified pi_0
        |
        v
freeze pi_0
        |
        v
real-dynamics frontier evidence under pi_0
        |
        v
pi_0-conditioned C_up^0 / C_down^0
        |
        v
independent validation / calibration
        |
        v
core-retaining Tube_1
        |
        v
train and select pi_1
        |
        v
================ iterative regime ================
        |
        v
predeclare outcome-blind newest-shell frontier roles
        |
        +--> TRAIN
        +--> CALIBRATION
        +--> ACCEPTANCE
        |
        v
collect real-dynamics frontier evidence under selected pi_k
        |
        v
fit C_up^k / C_down^k on TRAIN only
        |
        v
calibrate thresholds on disjoint CALIBRATION only
        |
        v
build Tube_(k+1): retain ALL Tube_k + qualified TRAIN expansion
        |
        v
Tube-RSI smoke and role-isolation audit
        |
        v
lock pi_k core/boundary acceptance baseline BEFORE candidate training
        |
        v
fresh pi_(k+1) training
        |
        v
freeze exact final checkpoint
        |
        v
strict locked-baseline core-preservation + boundary-gain gate
        |
   +----+----+
   |         |
 PASS       FAIL
   |         |
select      preserve evidence
pi_(k+1)    stop and diagnose
   |
   v
repeat
        |
        v
when stopping rule is declared and final policy frozen:
independent final JCE/JEL evaluation
```

---

## 6. Data-role isolation

From the generic iterative regime onward, frontier data has three predeclared logical roles plus a final untouched role.

### TRAIN

May:

- fit `C^k`;
- contribute candidate expansion states to `Tube_(k+1)` if continuation-positive and above the calibrated threshold.

### CALIBRATION

May:

- calibrate the frozen `C^k` phase threshold.

May **not**:

- train `C^k`;
- enter a Tube;
- serve as candidate acceptance evidence.

### ACCEPTANCE

Used to lock baseline boundary outcomes before candidate training and to compare the candidate later.

May **not**:

- train `C^k`;
- calibrate `C^k`;
- enter a Tube.

### FINAL TEST/JCE/JEL

Untouched by iterative development.

It may not influence:

- Tube construction;
- threshold selection;
- policy training;
- checkpoint selection;
- iteration repair;
- iteration stopping.

Parent-group disjointness across TRAIN/CALIBRATION/ACCEPTANCE is required. Seed disjointness alone is not considered sufficient.

---

## 7. Completed bootstrap chain

### 7.1 Frozen experts

`pi_up_star`

- training transitions: 9,977,856;
- actor SHA-256: `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`.

`pi_down_star`

- training transitions: 25,600;
- actor SHA-256: `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`.

Frozen manifest:

```text
JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json
```

### 7.2 Tube_0

```text
JIT/runs/soft_tube/soft_tube_train_v1_20260828
```

Tube_0 contains:

```text
222 TRAIN states
= 117 upstream
+ 105 downstream
```

Manifest SHA-256:

```text
c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b
```

### 7.3 pi_0

Frozen authority:

```text
JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json
```

Identity:

- 10,009,600 PPO training transitions;
- actor SHA-256: `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`;
- payload SHA-256: `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`.

### 7.4 C^0 and Tube_1

`C_up^0/C_down^0` were fitted from frozen-pi_0 continuation evidence and passed fresh independent validation/calibration.

Tube_1:

```text
JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
```

Composition:

```text
retained Tube_0 = 222
new expansion   = 2897
total           = 3119

upstream   = 427  = 117 core + 310 expansion
downstream = 2692 = 105 core + 2587 expansion
```

Manifest SHA-256:

```text
817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80
```

Entries SHA-256:

```text
61c6796aaf4c4b1e43624c5cf06bce0d39736a6d1743c5142c6c250d23155ec9
```

Tube_1 is a true structural superset of Tube_0.

---

## 8. Iteration-1 policy study and what it proved

The first Tube_1 policy candidate did not preserve the old core. That rejection led to the retained-core replay repair and subsequent initialization/warm-start study.

The completed comparison is:

| policy/checkpoint | Tube_0 core | regressions | upstream | downstream | boundary | parent groups |
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

Therefore the study is closed:

1. `repair02` is selected as engineering `pi_1`.
2. warm-start A is discarded;
3. warm-start B is retained as an ablation/scientific diagnostic only;
4. do not spend more compute on B checkpoint selection or A/B warm-start variants during the current mainline.

### 8.1 Interpretation of B

B regressions across checkpoints were:

```text
5 -> 16 -> 8 -> 5 -> 10
```

This is not monotonic overtraining.

Every B checkpoint kept downstream core at `105/105`. All core regressions occurred upstream, and most boundary gain was also upstream.

The supported interpretation is:

> Naive full warm-start creates upstream expansion/retention interference: learning new upstream boundary behavior changes previously successful upstream behavior.

This is not evidence of a general descent/recovery failure.

A future method study may investigate warm-start plus explicit retention constraints such as distillation, KL anchoring, or constrained replay. That is **not** the current mainline and must not be reopened before the formal pi_1 -> pi_2 loop produces new evidence requiring such a decision.

---

## 9. Selected pi_1 and the historical claim boundary

Selected policy:

```text
repair02
```

Frozen policy path:

```text
JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json
```

Identity:

- actor SHA-256: `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`;
- payload SHA-256: `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`.

Historical quickcheck:

```text
Tube_0 core = 222/222
upstream    = 117/117
downstream  = 105/105
regressions = 0
boundary    = 26/260
parent groups with candidate success = 4
```

Historical quickcheck summary:

```text
JIT/runs/pi_unified_gate/pi_0_to_pi_1_repair02_quickcheck_20260903/summary.json
```

### 9.1 The three baseline-reproduction failures

The historical boundary gate contains 3 baseline-reproduction failures caused by a mismatch between the historical continuation-label PRNG hierarchy and the paired-gate re-roll hierarchy.

These are not repair02 Tube_0 core regressions.

The correct language is:

- **engineering selection:** repair02 is selected as pi_1 and is authorized to generate C^1 / Tube_2 / pi_2 evidence;
- **publication/formal historical claim:** do not claim that the old Iteration-1 strict gate formally PASSed.

Do not modify historical artifacts to remove this discrepancy.

The selection CLI supports the explicit quarantine:

```text
JIT/cli/select_iteration_policy.py --allow-baseline-reproduction-mismatch
```

The selected policy artifact must record the distinction as machine-readable state, including:

```text
engineering_selection = true
formal_acceptance_claim = false
baseline_reproduction_mismatch_quarantined = true
```

---

## 10. Retained-core replay method selected for the mainline

The selected repair02-style training contract is:

```text
outer reset mixture:
  90% Tube RSI
  10% natural

inside Tube RSI:
  75% retained source Tube_k
  25% newest expansion
```

For pi_2, **retained source Tube_k means the whole Tube_1 support of 3,119 states**, not the original 222 Tube_0 subset.

This point is critical. Cumulative structural retention and sampling retention must advance together with the iteration.

Do not hard-code the 222-state Tube_0 subset into later policy training.

The 75/25 replay ratio is the selected mainline method based on the completed Iteration-1 study. Do not automatically sweep replay ratios after every later gate result.

---

## 11. Current code-control architecture

The repository has moved from one-off experiment scripts toward stable capabilities under `JIT/src/jit_dvgc/` plus thin CLIs.

Important package boundaries:

- `jit_dvgc.training` — unified PPO, preflight, training/freeze support;
- `jit_dvgc.tube` — Soft Tube and Tube-RSI capabilities;
- `jit_dvgc.snapshots` — snapshot and handoff state representations;
- `jit_dvgc.acquisition` — real-dynamics boundary/frontier acquisition;
- `jit_dvgc.continuation` — continuation labeling/fields;
- `jit_dvgc.analysis` — bounded diagnostics/gates;
- `jit_dvgc.workflow` — resumable manifest-driven orchestration.

Iteration-generic durable implementations include:

```text
JIT/src/jit_dvgc/iterative_frontier_protocol.py
JIT/src/jit_dvgc/iterative_continuation_fields.py
JIT/src/jit_dvgc/iterative_tube.py
JIT/src/jit_dvgc/iterative_acceptance_gate.py
JIT/src/jit_dvgc/workflow/
```

Primary CLIs for the automatic loop:

```text
JIT/cli/select_iteration_policy.py
JIT/cli/prepare_iterative_envelope_workflow.py
JIT/cli/run_iteration_workflow.py
JIT/cli/run_iterative_frontier_protocol.py
JIT/cli/fit_iterative_continuation_fields.py
JIT/cli/build_iterative_tube.py
JIT/cli/audit_iterative_role_isolation.py
JIT/cli/run_iterative_acceptance_gate.py
JIT/cli/prepare_iterative_unified_training.py
JIT/cli/train_unified.py
JIT/cli/freeze_unified_policy.py
```

Do not create `pi2_*`, `tube2_*`, `retry03_*`, or seed-specific production source modules just because a new iteration/run exists. Iteration identity belongs in configs, manifests, and run metadata.

---

## 12. Automatic k -> k+1 workflow

The generic automatic path is implemented for `k >= 1`.

Workflow preparation:

```text
JIT/cli/prepare_iterative_envelope_workflow.py
```

Workflow execution:

```text
JIT/cli/run_iteration_workflow.py
```

The generated DAG is:

```text
selected pi_k + Tube_k
  -> prepare_frontier_plan
  -> frontier_train
  -> frontier_calibration
  -> frontier_acceptance
  -> fit_and_calibrate_Ck
  -> build_Tube(k+1)
  -> smoke_Tube(k+1)
  -> audit_role_isolation
  -> lock_pi_k_acceptance_baseline
  -> prepare_pi(k+1)_training
  -> train_pi(k+1)
  -> freeze_pi(k+1)
  -> gate_pi(k)_to_pi(k+1)
  -> select_pi(k+1)
```

### 12.1 Workflow runner semantics

The runner is intentionally scientifically ignorant. It sequences declared stages and verifies artifacts; it does not invent method changes.

Important behavior:

- without `--execute`, it only resolves/prints the plan;
- state is resumable;
- the workflow config SHA is immutable after state creation;
- each stage has declared prerequisites;
- each stage has a machine-readable completion artifact;
- completion assertions must pass before the runner advances;
- an existing completion artifact is revalidated rather than blindly trusted;
- completed stages are not silently rerun;
- the runner stops on scientific or engineering failure;
- it never includes final TEST/JCE/JEL stages;
- it never automatically changes thresholds, rewards, PPO hyperparameters, replay ratio, network architecture, physics, reset semantics, or acceptance criteria to force progress.

### 12.2 Frontier source behavior

The automatic frontier uses only the **newest expansion shell of Tube_k** as the parent pool.

It does not silently fall back to the full Tube if the newest shell is absent or lacks enough phase/group support.

If the shell is scientifically insufficient, stop and make a new parent-generation decision explicitly.

### 12.3 Tube construction behavior

`JIT/src/jit_dvgc/iterative_tube.py` implements:

```text
Tube_(k+1)
  = every Tube_k entry retained exactly
  + new logical-TRAIN states
      where pi_k continuation label is positive
      and C^k score is strictly above the disjoint calibration threshold
```

For the immediate round:

```text
Tube_2 must retain all 3,119 Tube_1 states.
```

CALIBRATION and ACCEPTANCE rows must not appear in Tube_2.

---

## 13. Future acceptance gate and the PRNG fix

The historical Iteration-1 gate attempted to reproduce baseline negative states after training under a different PRNG hierarchy. That produced the 3-state reproduction mismatch.

Future rounds use a different protocol implemented by:

```text
JIT/src/jit_dvgc/iterative_acceptance_gate.py
```

The future contract is:

1. evaluate and lock `pi_k` core outcomes **before** candidate training;
2. evaluate and lock ACCEPTANCE-role `pi_k` boundary negatives before candidate training;
3. preserve exact labeling PRNG identity;
4. do not re-roll baseline boundary negatives after candidate training;
5. evaluate candidate core using the same locked core seeds;
6. evaluate candidate boundary using the same `labeling_seed -> candidate_index -> tick` hierarchy;
7. require zero baseline-success -> candidate-failure core regressions;
8. require the predeclared boundary-success parent-group criterion;
9. perform no training, expert switching, validation/calibration fitting, or TEST access during the gate.

This fixes the future protocol without rewriting the historical Iteration-1 record.

---

## 14. Regression tests that protect the automatic loop

The main contract test file is:

```text
JIT/tests/test_iterative_envelope_automation.py
```

It verifies, among other things:

- frontier generation uses only the newest expansion shell;
- the frontier refuses an iteration when the newest shell lacks required two-phase support;
- iterative Tube construction retains the entire source Tube exactly;
- only logical-TRAIN expansion rows can be added;
- CALIBRATION/ACCEPTANCE/TEST rows are not embedded;
- the workflow stops before a later stage when a scientific gate fails.

When changing production code, use the existing test suite rather than adding iteration-specific duplicate modules.

At minimum after structural code changes run:

```bash
python -m compileall -q JIT/src JIT/cli
```

plus targeted tests for the changed capability.

---

## 15. Exact current position

The mainline state machine is:

```text
Tube_0
  -> pi_0 frozen
  -> C^0 complete
  -> Tube_1 complete
  -> Iteration-1 candidate/repair/A/B study complete
  -> repair02 selected as engineering pi_1
  -> CURRENT POSITION
  -> pi_1-conditioned frontier evidence
  -> C^1
  -> Tube_2
  -> pi_2
```

There is no scientific reason to reopen the A/B checkpoint sweep before this mainline produces new evidence.

The next job is not to invent another pi_1 variant. The next job is to execute the generic pi_1 -> pi_2 workflow.

---

## 16. Immediate Codex/operator startup sequence

First synchronize safely:

```bash
cd ~/DVGC
git fetch origin
git checkout agent/two-phase-soft-tube
git pull --ff-only origin agent/two-phase-soft-tube
git status --short

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python
```

Do **not** reset, clean, stash, rebase, or overwrite unrelated local work.

### 16.1 Verify/register selected repair02

Because `JIT/runs/` contains runtime artifacts and is normally not the Git source of truth, verify the local artifact first. If the selected-policy registration has not already been created, run:

```bash
$PY JIT/cli/select_iteration_policy.py \
  --frozen-policy JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json \
  --gate-summary JIT/runs/pi_unified_gate/pi_0_to_pi_1_repair02_quickcheck_20260903/summary.json \
  --output-dir JIT/runs/iteration_selection/pi_1_repair02_selected_20260903 \
  --allow-baseline-reproduction-mismatch
```

Required selected artifact:

```text
JIT/runs/iteration_selection/pi_1_repair02_selected_20260903/selected_policy.json
```

Verify that it records the engineering/formal distinction instead of silently declaring a historical formal PASS.

### 16.2 Generate the pi_1 -> pi_2 workflow

```bash
$PY JIT/cli/prepare_iterative_envelope_workflow.py \
  --selected-policy JIT/runs/iteration_selection/pi_1_repair02_selected_20260903/selected_policy.json \
  --source-tube JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901 \
  --tag 20260903 \
  --work-root JIT/runs/iteration_auto/pi_1_to_pi_2_20260903 \
  --config-out JIT/runs/iteration_auto/pi_1_to_pi_2_20260903/workflow.json
```

### 16.3 Dry-run the DAG

```bash
$PY JIT/cli/run_iteration_workflow.py \
  --config JIT/runs/iteration_auto/pi_1_to_pi_2_20260903/workflow.json
```

This must not execute scientific stages.

Inspect the resolved plan and verify the expected source/target iteration identities before execution.

### 16.4 Execute

```bash
$PY JIT/cli/run_iteration_workflow.py \
  --config JIT/runs/iteration_auto/pi_1_to_pi_2_20260903/workflow.json \
  --execute
```

If interrupted for an engineering reason, rerun the same command. The runner should revalidate and reuse completed stages.

Do not generate a new config merely to bypass the existing workflow state unless the previous scientific protocol has been explicitly abandoned and a new experiment is being predeclared.

---

## 17. What each immediate stage is supposed to produce

### Frontier plan

Outcome-blind role assignment for newest-shell parents.

Expected role classes:

```text
TRAIN
CALIBRATION
ACCEPTANCE
```

The plan must exist before frontier outcomes are observed.

### TRAIN frontier

Real-dynamics evidence used to fit C^1 and potentially enlarge Tube_2.

### CALIBRATION frontier

Disjoint evidence used only for C^1 threshold calibration.

### ACCEPTANCE frontier

Disjoint evidence that will be used to lock the pi_1 baseline before pi_2 training.

### C^1

Fit `C_up^1/C_down^1` on TRAIN only, then calibrate on CALIBRATION only.

A failed fixed calibration contract stops the workflow.

### Tube_2

Must:

- preserve all 3,119 Tube_1 entries;
- add only qualifying TRAIN expansion;
- contain no CALIBRATION or ACCEPTANCE rows;
- have positive expansion for the workflow to continue under the current contract.

### Tube_2 smoke

Confirms Tube-RSI runtime/config compatibility before expensive policy training.

### Role-isolation audit

Confirms that CALIBRATION and ACCEPTANCE have not leaked into Tube_2 and that role boundaries remain valid.

### pi_1 baseline lock

Must occur before pi_2 training.

Locks exact core outcomes and boundary-negative evidence with exact PRNG identity.

### pi_2 preparation/training

Uses the whole Tube_1 source support as retained core and the new Tube_2 expansion under the selected strong replay method.

Target formal budget remains 10,009,600 PPO training transitions under the generated mainline config unless a new scientific decision explicitly changes the method in a separate experiment.

### pi_2 freeze

Freeze the exact final checkpoint and bind provenance.

### strict pi_1 -> pi_2 gate

Requires zero core regressions and the declared boundary-gain criterion on the already locked evidence.

### pi_2 selection

Only occurs if the strict gate PASSes.

---

## 18. Failure decision tree

The automation is allowed to stop. A stop is a scientific result or an engineering blocker, not a reason to silently mutate the protocol.

### Case A — newest Tube_1 shell lacks adequate phase/parent-group support

Do:

- preserve the frontier-plan failure;
- inspect support geometry and parent availability;
- make a new parent-generation/acquisition decision explicitly.

Do not:

- silently fall back to the full Tube;
- reassign roles after seeing outcomes.

### Case B — C^1 fitting/calibration fails

Do:

- preserve TRAIN/CALIBRATION evidence;
- identify whether the fixed model/data contract was violated;
- open a new method decision if architecture/data-generation changes are scientifically required.

Do not:

- repeatedly tune thresholds against the same calibration outcomes until PASS.

### Case C — Tube_2 expansion is zero

Treat this as potential envelope saturation or insufficient frontier generation.

Do not fabricate expansion by lowering thresholds post hoc inside the same declared round.

### Case D — Tube-RSI smoke fails

This is an engineering/runtime blocker. Repair compatibility/provenance without changing the scientific data roles or task semantics.

### Case E — role-isolation audit fails

Stop. Leakage invalidates progression to policy training.

Repair the pipeline/data membership; do not continue with pi_2.

### Case F — pi_2 training fails technically

Resume only if the existing run/provenance contract supports safe resumption. Do not silently warm-start a stage declared fresh-only.

### Case G — strict pi_1 -> pi_2 gate has core regressions

Preserve the failed gate and diagnose retention/interference.

Do not automatically change replay ratio or start a hyperparameter sweep.

### Case H — core preserves but boundary gain fails

The candidate is not the next capability authority under the declared protocol.

Preserve evidence and diagnose whether the envelope is saturating, the frontier is uninformative, or policy improvement failed.

### Case I — gate PASS

Select pi_2 as the next authority and begin the same generic loop:

```text
pi_2 -> C^2 -> Tube_3 -> pi_3
```

---

## 19. What must not be done

A new agent must not:

- modify `main` without explicit authorization;
- reset/clean/stash/rebase unrelated local work;
- reopen A/B or continue the B checkpoint sweep during the current mainline;
- describe the historical repair02 gate as a strict formal PASS;
- edit historical artifacts to remove the 3 PRNG reproduction failures;
- use final TEST/JCE/JEL evidence during iteration;
- embed CALIBRATION or ACCEPTANCE rows into a Tube;
- train later continuation fields using bootstrap `V_up/V_down` as if they were C^k;
- equate PPO critic/value with C^k;
- change physics/reward/action/snapshot semantics to fix an iteration result;
- create iteration-specific production modules when an existing generic capability should be extended;
- hard-code Tube_0's 222 states as the retained core for pi_2 or later policies;
- auto-tune after a failed scientific gate;
- claim a larger Tube alone proves a larger empirical capability envelope.

---

## 20. Repository maintenance rules for Codex

Follow a modify-first policy:

1. modify/consolidate an existing production capability first;
2. add a new production Python file only for a genuinely new durable capability;
3. keep iteration/run identity in config/artifact metadata;
4. keep `JIT/cli/` thin;
5. keep reusable scientific logic under `JIT/src/jit_dvgc/`;
6. keep tests under `JIT/tests/`;
7. do not move path-bound configs/manifests merely for aesthetics because recorded paths are provenance;
8. delete superseded code only after proving no production import, API, CLI, test, loader, config, or frozen reproducibility path depends on it;
9. after deletion/structural changes, run compile/import/targeted-test gates before deleting more.

Git history is the archive. Do not create an in-tree cemetery of old experiment Python files.

---

## 21. When to update documentation

After any major state transition, update at least:

```text
JIT/docs/CURRENT_STATUS.md
PROJECT.md
AGENTS.md
JIT/AGENTS.md   (when agent-control rules or current authority change)
```

Update this handoff if the architecture, authority model, or takeover procedure materially changes.

Do not let root `AGENTS.md` or `PROJECT.md` lag behind `JIT/docs/CURRENT_STATUS.md`; stale root entry files can send a new agent back into a closed experiment.

---

## 22. Project-level stopping and completion

Do not iterate indefinitely just because the workflow can mechanically produce another round.

A project-level stopping decision can be based on predeclared evidence such as:

- negligible/new Tube growth;
- repeated inability to expand without retention loss;
- empirical frontier saturation;
- reaching the intended physical task envelope;
- resource budget;
- a scientifically justified diminishing-return criterion.

Only after the stopping rule is declared and the final policy is frozen should final TEST/JCE/JEL evaluation run.

The final result should report:

- exact final policy identity;
- exact physical/task identity;
- exact evaluation protocol and seeds;
- empirical success envelope / limit evidence;
- retained-core evidence across accepted iterations;
- expansion evidence across accepted iterations;
- failure/frontier behavior;
- limitations and claim boundary.

The final claim remains empirical and policy-conditioned.

---

## 23. First-read order for a new agent

Read in this order before changing scientific flow:

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/CODEX_HANDOFF_20260903.md`
5. `PROJECT.md`
6. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
7. `JIT/docs/CODE_ORGANIZATION.md`

Then inspect the actual runtime artifacts referenced by the current status before launching or resuming a workflow.

Do not reconstruct current truth from older Phase-U reports or obsolete experimental narratives when the current authority documents supersede them.

---

## 24. One-paragraph takeover instruction

If you are Codex taking over now: work only on `agent/two-phase-soft-tube`, preserve unrelated local work, treat repair02 as the selected engineering `pi_1` while preserving the historical non-formal Iteration-1 claim, do not reopen A/B, verify/register the selected-policy artifact, generate and dry-run the generic `pi_1 -> pi_2` workflow, then execute it. Allow the workflow to stop on scientific failure. The immediate research objective is to obtain policy-conditioned `C^1`, a Tube_2 that retains all 3,119 Tube_1 states plus evidence-backed TRAIN expansion, and a pi_2 that passes the new pre-candidate locked-baseline retention/expansion gate. If pi_2 passes, select it and repeat the same generic loop; if any gate fails, preserve the evidence and diagnose rather than automatically retuning the method.
