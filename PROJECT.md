# DVGC Project

## Scientific objective

DVGC/JIT studies **Policy-Conditioned Soft-Tube Expansion for Empirical Jumping Capability Envelope Identification** on a single-track two-wheeled robot.

The project is not a two-policy deployment system and not merely a natural-reset jumping-policy benchmark. The deployable controller is one unified Actor. Two phase experts bootstrap the state distribution and continuation labels; later envelope expansion is conditioned on the frozen unified policy that will actually execute the maneuver.

## Method

```text
Propulsion-Ascent expert pi_up
        +
Descent-Recovery expert pi_down
        ↓
freeze experts
        ↓
real handoff/continuation labels
        ↓
expert-conditioned V_up / V_down
        ↓
TRAIN-only Tube_0
        ↓
unified Tube-RSI policy pi_0
        ↓
freeze pi_k
        ↓
real-dynamics boundary acquisition
        ↓
pi_k-conditioned continuation labels
        ↓
C_up^k / C_down^k
        ↓
fresh independent validation/calibration
        ↓
Tube_(k+1) = retained core ∪ evidence-backed expansion
        ↓
unified Tube-RSI policy pi_(k+1)
        ↓
core-preservation + boundary-gain gates
        ↓
accept only when both gates pass
        ↓
repeat while the predeclared iteration protocol authorizes expansion
        ↓
independent final frozen-policy empirical JCE/JEL
```

### Phase meaning

- `Propulsion-Ascent`: launch and rising-flight behavior needed to reach the Apex transition band.
- `Descent-Recovery`: descent, landing, and stable recovery from that band.
- The Apex transition is a physical handoff band, not a third expert and not a certified safe set.

### Field meaning

- `V_up/V_down` are bootstrap expert-conditioned continuation fields used to construct Tube_0.
- `C_up^k/C_down^k` are policy-conditioned continuation fields tied to the exact frozen `pi_k` identity.
- The same state may fail under `pi_k` and succeed under `pi_(k+1)`; later continuation fields cannot be silently replaced by the bootstrap expert fields.

### Tube meaning

Every learned Tube is training guidance/curriculum support only.

`Tube_(k+1)` is **core retaining** structurally: the established source Tube support is retained and qualifying new TRAIN states are added. Structural retention alone does not prove that policy training preserves competence on that core.

A larger Tube does not establish capability expansion. Empirical expansion is recorded only after the newly trained policy passes both:

1. core preservation;
2. boundary gain.

The final empirical JCE/JEL is policy-conditioned evidence, not a proof of invariance, viability, or safety.

## Current completed state — 2026-09-01

Completed and locked in the active JIT line:

- `pi_up_star` and `pi_down_star` phase experts;
- bootstrap `V_up/V_down`;
- Tube_0: 222 TRAIN entries;
- unified `pi_0`, frozen as iteration-0 expansion authority;
- pi_0 TRAIN boundary evidence;
- frozen policy-conditioned `C_up^0/C_down^0` with fresh independent validation;
- Tube_1: 3,119 TRAIN entries = 222 retained Tube_0 core + 2,897 expansion states;
- Tube_1 mixed legacy/unified snapshot runtime gate;
- `pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`: fresh formal PPO completed exactly 10,009,600 training transitions, five TRAIN panels, no validation/TEST, no expert switching, and verified final checkpoint restore;
- exact pi_1 frozen as iteration-1 comparison authority;
- generic paired pi_0 -> pi_1 iteration gate completed under the predeclared protocol.

### Paired pi_0 -> pi_1 gate result

Core preservation **failed**:

- Tube_0 core states: 222
- pi_0 success: 222 / 222
- pi_1 success: 201 / 222
- regressions: 21 = 16 upstream + 5 downstream

Boundary gain **passed**:

- locked pi_0-negative frontier states: 56
- pi_0 failure reproduction errors: 0
- pi_1 new successes: 12
- successful parent groups: 5
- all observed boundary gains were upstream in this audit

Therefore:

- iteration accepted: false
- empirical pi_0 -> pi_1 capability-envelope expansion accepted: false

This is a scientific rejection, not an engineering failure. The result must not be converted into a PASS by changing the already-consumed gate bank, acceptance threshold, reward, or PPO settings.

## Current blocker

The project is now diagnosing **core forgetting under Tube_1 training**.

The main working hypothesis is that the 2,897 expansion states may have diluted replay of the retained 222 Tube_0 core states despite structural retention. That is not yet established. Before changing the training method, the existing frozen gate records and Tube sampling weights must be audited for:

- regression outcome classes;
- phase and parent/source concentration;
- retained-core sampling probability mass within each 50/50 phase;
- whether failed core states are unusually low-weight or concentrated near particular source groups;
- whether the failure is better explained by curriculum/replay dilution or by a deeper phase/runtime issue.

No `C^1`, Tube_2, or pi_2 stage is authorized while this blocker remains unresolved.

## Next scientific sequence

1. preserve the completed paired-gate FAIL artifact unchanged;
2. run zero-interaction regression diagnosis on the 21 core failures;
3. determine the mechanism before choosing a repair;
4. predeclare a revised policy-improvement method only after diagnosis;
5. train a repaired candidate without using TEST/final data;
6. repeat a newly predeclared paired core/boundary audit;
7. only if both gates pass may that accepted policy become the next expansion authority and generate `C^1` / Tube_2 evidence.

Final TEST/JCE/JEL remains untouched.

## Leakage and claim boundary

- TRAIN evidence may affect model fitting and Tube construction.
- Fresh validation may calibrate/gate a frozen field, but validation rows never enter TRAIN or a Tube.
- Consumed validation is not reused for later tuning.
- TEST/final evaluation does not affect threshold selection, Tube construction, policy training, checkpoint selection, or iteration stopping.
- Natural cold-start failure is retained as an out-of-domain diagnostic for the present declared JCE scope; it is not a reason to silently change reward, reset ratio, or task semantics.

## Immutable task contract

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`

Do not create a replacement XML to fix the historical `4kg` filename. Envelope iteration must not silently alter physics, meshes, collision geometry, reward meaning, snapshot semantics, action semantics, or TEST isolation.

## Engineering direction

The repository is moving from experiment-stage scripts toward reusable iteration capabilities. Iteration numbers belong in config/artifact/run metadata, not new production source modules.

The intended operator interface is one explicit resumable workflow launch. Automation may execute and verify declared stages, but a scientific gate failure must stop the workflow and surface diagnosis; automation may not retune thresholds/hyperparameters or bypass the gate.

See:

- `JIT/docs/CURRENT_STATUS.md`
- `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
- `JIT/docs/CODE_ORGANIZATION.md`
