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
- The same state may fail under `pi_k` and succeed under `pi_(k+1)`; therefore later continuation fields cannot be silently replaced by the bootstrap expert fields.

### Tube meaning

Every learned Tube is training guidance/curriculum support only.

`Tube_(k+1)` is **core retaining**: the established source Tube support is retained and qualifying new TRAIN states are added. It is not simply a new thresholded level set that may discard old support.

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
- `pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`: fresh formal PPO completed exactly 10,009,600 training transitions, five TRAIN panels, no validation/TEST, no expert switching, and verified final checkpoint restore.

The first pi_1 attempt is preserved as engineering-error provenance. It is not a scientific result and is not a warm-start source.

## Current next gate

The next scientific sequence is fixed:

1. freeze the exact completed pi_1 final checkpoint as iteration-1 authority;
2. evaluate core preservation under a predeclared comparable state bank;
3. evaluate boundary gain under a predeclared/disjoint comparable boundary bank;
4. only if both pass, record empirical `pi_0 -> pi_1` envelope expansion;
5. collect/freeze pi_1-conditioned TRAIN evidence;
6. fit/validate `C_up^1/C_down^1`;
7. construct core-retaining Tube_2;
8. train pi_2 with the same stable production capabilities;
9. continue by the same generic workflow.

Final TEST/JCE/JEL remains untouched during this iteration loop.

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

The intended operator interface is one explicit resumable workflow launch. Automation may execute and verify declared stages, but it may not retune thresholds/hyperparameters or bypass failed scientific gates.

See:

- `JIT/docs/CURRENT_STATUS.md`
- `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
- `JIT/docs/CODE_ORGANIZATION.md`
