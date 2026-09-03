# DVGC Project

## Scientific objective

DVGC/JIT studies **Policy-Conditioned Soft-Tube Expansion for Empirical Jumping Capability Envelope Identification** on a single-track two-wheeled robot.

The goal is to iteratively identify and enlarge an empirical state-space support from which **one unified policy** can complete the jumping maneuver, while preserving previously demonstrated capability.

The project is not a two-policy deployment system and not merely a natural-reset jumping benchmark. Two phase experts bootstrap the initial support and continuation evidence; later iterations are conditioned on the exact frozen unified policy that will actually execute the maneuver.

The final scientific output is an empirical, policy-conditioned jumping capability envelope/limit (JCE/JEL). It is not a proof of safety, viability, reachability, or invariance.

---

## Method

### Bootstrap phase

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
        ↓
freeze pi_0
```

### Iterative phase

```text
selected/frozen pi_k + Tube_k
        ↓
outcome-blind newest-shell frontier plan
        ↓
TRAIN / CALIBRATION / ACCEPTANCE roles
        ↓
real-dynamics frontier acquisition under pi_k
        ↓
pi_k-conditioned continuation labels
        ↓
fit C_up^k / C_down^k on TRAIN only
        ↓
calibrate thresholds on disjoint CALIBRATION only
        ↓
Tube_(k+1)
= every Tube_k entry retained exactly
+ evidence-backed logical-TRAIN expansion
        ↓
Tube-RSI smoke + role-isolation audit
        ↓
lock pi_k core/boundary baseline before candidate training
        ↓
fresh unified pi_(k+1) training
        ↓
freeze exact final checkpoint
        ↓
strict locked-baseline core-preservation + boundary-gain gate
        ↓
PASS: select pi_(k+1) and repeat
FAIL: preserve evidence, stop, diagnose
        ↓
after a declared stopping decision only:
independent final frozen-policy empirical JCE/JEL
```

---

## Phase meaning

- `Propulsion-Ascent`: launch and rising-flight behavior needed to reach the Apex/transition region.
- `Descent-Recovery`: descent, landing, and stable recovery from that region.
- The Apex transition is a physical handoff band, not a third expert.

The final unified Actor does not perform expert switching at runtime.

---

## Continuation-field meaning

- `V_up/V_down` are bootstrap expert-conditioned continuation fields used to construct Tube_0.
- `C_up^k/C_down^k` are policy-conditioned continuation fields tied to the exact selected/frozen `pi_k` identity.
- The same state may fail under `pi_k` and succeed under `pi_(k+1)`.
- Later continuation fields must not be silently replaced by the bootstrap expert fields.
- PPO critic/value is not a JIT continuation field.

---

## Tube meaning

Every learned Tube is empirical training/curriculum support only.

For the iterative regime:

```text
Tube_(k+1)
  = all Tube_k entries retained exactly
  + qualifying logical-TRAIN expansion states
```

A qualifying expansion row must be supported by the exact selected `pi_k`, have a positive continuation label, and pass the frozen `C^k` decision threshold calibrated on disjoint CALIBRATION evidence.

CALIBRATION and ACCEPTANCE rows never enter a Tube.

Structural retention alone does not prove policy retention. A newly trained policy must still pass a direct capability gate.

A larger Tube, higher training return, or training completion does not establish capability expansion.

---

## Data-role contract

Later iterations use three predeclared development roles:

- `TRAIN`: may fit `C^k` and contribute qualifying Tube expansion;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: pre-candidate locked baseline and later candidate comparison only.

These roles must be parent-group disjoint before outcomes are observed. Seed disjointness alone is insufficient.

A fourth role, final TEST/JCE/JEL, remains untouched until the final policy and stopping decision are fixed.

---

## Completed state — 2026-09-03

### Frozen phase experts

`pi_up_star`

- 9,977,856 training transitions;
- actor SHA-256: `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`.

`pi_down_star`

- 25,600 training transitions;
- actor SHA-256: `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`.

Frozen manifest:

```text
JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json
```

### Tube_0

```text
JIT/runs/soft_tube/soft_tube_train_v1_20260828
```

Composition:

```text
222 TRAIN states
= 117 upstream
+ 105 downstream
```

### pi_0

Frozen authority:

```text
JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json
```

- 10,009,600 PPO training transitions;
- actor SHA-256: `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`;
- payload SHA-256: `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`.

### C^0 and Tube_1

`C_up^0/C_down^0` were fitted from frozen-pi_0 continuation evidence and passed fresh independent validation/calibration.

Tube_1:

```text
JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
```

Composition:

```text
retained Tube_0 = 222
expansion       = 2897
total           = 3119

upstream   = 427  = 117 core + 310 expansion
downstream = 2692 = 105 core + 2587 expansion
```

Tube_1 is a true core-retaining superset of Tube_0.

---

## Iteration-1 policy study — closed

The first Tube_1 candidate failed core preservation, which motivated a retained-core replay repair and a bounded initialization/warm-start study.

The final comparison is:

| policy/checkpoint | Tube_0 core | regressions | upstream | downstream | boundary | groups |
|---|---:|---:|---:|---:|---:|---:|
| **repair02** | **222/222** | **0** | **117/117** | **105/105** | 26/260 | 4 |
| B 1.024M | 217/222 | 5 | 112/117 | **105/105** | 33/260 | 3 |
| B 2.5088M | 206/222 | 16 | 101/117 | **105/105** | 28/260 | 4 |
| B 5.0176M | 214/222 | 8 | 109/117 | **105/105** | 25/260 | 4 |
| **B 7.5008M** | **217/222** | **5** | **112/117** | **105/105** | **42/260** | **4** |
| B 10.0096M | 212/222 | 10 | 107/117 | **105/105** | **46/260** | 4 |

No B checkpoint achieved both:

```text
Tube_0 = 222/222
and
boundary > 26/260
```

Therefore:

- **repair02 is selected as the engineering pi_1 authority**;
- warm-start A is discarded;
- warm-start B is closed as an ablation/scientific diagnostic;
- do not continue B checkpoint sweeping or reopen warm-start variants during the active mainline.

### Scientific interpretation of B

B core-regression counts were non-monotonic:

```text
5 -> 16 -> 8 -> 5 -> 10
```

Every B checkpoint preserved downstream at `105/105`; all core regressions occurred upstream. Boundary gains were also overwhelmingly upstream.

The supported interpretation is **upstream expansion/retention policy interference under naive full warm-start**, not simple monotonic overtraining and not a general descent/recovery failure.

Warm-start plus explicit retention constraints may become a later research extension, but it is not the current mainline.

---

## Selected pi_1 authority

Selected policy: **repair02**.

Frozen policy:

```text
JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json
```

Identity:

- actor SHA-256: `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`;
- payload SHA-256: `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`.

Engineering quickcheck:

```text
Tube_0 core = 222/222
upstream    = 117/117
downstream  = 105/105
regressions = 0
boundary    = 26/260
successful parent groups = 4
```

### Historical formal-claim quarantine

The historical repair02 quick gate contains 3 baseline-reproduction failures caused by the old continuation-label versus paired-gate PRNG hierarchy mismatch.

Correct claim:

- engineering continuation: **authorized**;
- historical strict publication-level Iteration-1 PASS: **not claimed**.

The mismatch is preserved as historical technical debt. Do not alter the old artifact to make it PASS.

`JIT/cli/select_iteration_policy.py --allow-baseline-reproduction-mismatch` exists to encode this distinction explicitly.

---

## Current mainline

The project is no longer diagnosing the original 21-regression candidate and is no longer choosing among A/B checkpoints.

The active state is:

```text
Tube_0
  -> pi_0
  -> C^0
  -> Tube_1
  -> repair02 selected as pi_1
  -> CURRENT
  -> pi_1-conditioned frontier evidence
  -> C^1
  -> Tube_2
  -> pi_2
```

Immediate objective:

> Execute the generic `pi_1 -> pi_2` workflow, obtain `C_up^1/C_down^1`, construct a Tube_2 that retains all 3,119 Tube_1 states plus evidence-backed TRAIN expansion, train/freeze pi_2, and subject it to the new strict locked-baseline capability gate.

---

## Selected retained-core replay contract

The current mainline policy-improvement method is:

```text
outer reset mixture:
  90% Tube RSI
  10% natural

inside Tube RSI:
  75% retained source Tube_k
  25% newest expansion
```

For pi_2, the retained source core is the **entire Tube_1 support of 3,119 states**.

Do not treat only the original 222 Tube_0 states as the retained core in later iterations.

---

## Code-control architecture

The repository is organized around reusable production capabilities rather than iteration-specific scripts.

Stable capability areas:

- `jit_dvgc.training` — unified PPO/preflight/freezing;
- `jit_dvgc.tube` — Soft Tube and Tube-RSI;
- `jit_dvgc.snapshots` — snapshot/handoff structures;
- `jit_dvgc.acquisition` — real-dynamics frontier acquisition;
- `jit_dvgc.continuation` — policy-conditioned labels/fields;
- `jit_dvgc.analysis` — bounded diagnostics/gates;
- `jit_dvgc.workflow` — resumable orchestration.

Iteration-generic implementations include:

```text
JIT/src/jit_dvgc/iterative_frontier_protocol.py
JIT/src/jit_dvgc/iterative_continuation_fields.py
JIT/src/jit_dvgc/iterative_tube.py
JIT/src/jit_dvgc/iterative_acceptance_gate.py
JIT/src/jit_dvgc/workflow/
```

The generic automatic workflow for `k >= 1` is prepared by:

```text
JIT/cli/prepare_iterative_envelope_workflow.py
```

and executed by:

```text
JIT/cli/run_iteration_workflow.py
```

The workflow is resumable, validates declared completion artifacts, and stops on failed scientific/engineering assertions. It does not auto-tune the method.

Regression coverage for the generic iteration contracts is in:

```text
JIT/tests/test_iterative_envelope_automation.py
```

---

## Future acceptance protocol

For later rounds, the baseline is locked **before** candidate training.

The new gate preserves exact pi_k core outcomes, boundary negatives, seeds, and labeling PRNG identity, and later evaluates the candidate against those locked outcomes.

This prevents the historical failure mode in which a baseline negative was re-rolled under a different PRNG hierarchy after candidate training.

Core PASS requires zero baseline-success -> candidate-failure regressions. Boundary PASS requires the predeclared successful-parent-group criterion.

---

## Immediate operator path

Use the exact procedure documented in:

```text
JIT/docs/CURRENT_STATUS.md
JIT/docs/CODEX_HANDOFF_20260903.md
```

The high-level operator sequence is:

```text
1. safely sync agent/two-phase-soft-tube
2. verify/register repair02 selected_policy.json
3. generate pi_1 -> pi_2 workflow.json
4. dry-run the resolved DAG
5. execute with --execute
6. allow scientific failures to stop the workflow
7. PASS -> select pi_2 and repeat
8. FAIL -> preserve evidence and diagnose; do not auto-retune
```

---

## Leakage and claim boundary

- TRAIN evidence may fit continuation models and contribute qualifying Tube expansion.
- CALIBRATION evidence may set the frozen continuation threshold but never enters a Tube.
- ACCEPTANCE evidence is reserved for the baseline/candidate capability gate and never enters a Tube.
- Final TEST/JCE/JEL evidence does not affect threshold selection, Tube construction, policy training, checkpoint selection, method repair, or iteration stopping.
- Natural cold-start failure remains an out-of-domain diagnostic for the present JCE scope and is not a reason to silently change reward/reset semantics.

---

## Immutable task contract

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`

Do not create a replacement XML to fix the historical filename. Envelope iteration must not silently alter physics, meshes, collision geometry, reward meaning, snapshot semantics, action semantics, or TEST isolation.

---

## Project completion

The generic loop may continue as:

```text
pi_2 -> C^2 -> Tube_3 -> pi_3 -> ...
```

but the project must not iterate merely because another round is mechanically possible.

A later project-level stopping decision should be based on declared evidence such as saturation/negligible Tube growth, repeated inability to expand without retention loss, reaching the intended physical envelope, or resource/diminishing-return limits.

Only after the stopping rule is declared and the final policy is frozen should final TEST/JCE/JEL evaluation be performed.

---

## Authoritative documentation

For takeover and current state, read:

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/CODEX_HANDOFF_20260903.md`
5. `PROJECT.md`
6. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
7. `JIT/docs/CODE_ORGANIZATION.md`

Do not reconstruct current truth from obsolete Phase-U or pre-repair reports when these authority documents supersede them.
