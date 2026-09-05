# OrangeBike DVGC / JIT

DVGC/JIT is an empirical reset-curriculum and capability-measurement project for
a fixed single-track two-wheeled robot jump. The final controller target is one
unified Actor without runtime expert switching.

## Current method

The active experiment starts from a locked ground jump state at `x = 2.5 m`.
Frozen π0 supplies a real-frame centerline and proposal prefix. Bounded action
perturbations reach exact candidate states through `env.step`. Frozen π0, π1 and
π2 then continue from each state; a candidate is positive if any evaluator
reaches the first valid landing before physical failure.

The project distinguishes:

- forward arrival evidence;
- policy-family landing witnesses;
- physical-resolution cell occupancy;
- raw reset/replay Tube rows;
- one-Actor realization.

These are not interchangeable. The method does not claim formal reachability,
a viability kernel, certified safety or a complete physical jump limit.

## Current status — 2026-09-05

The first family round produced 1,230/1,258 positive TRAIN candidates and a
1,159-row Tube3 increment. π3 was trained for 10,009,600 transitions, but its
stored selection gate mixed a stable-recovery baseline with a first-landing
candidate endpoint. π3 therefore remains trained historical evidence, not valid
prospective authority for π4.

An upstream advisory predictor has been fit. A larger fresh round acquired
TRAIN/CALIBRATION/ACCEPTANCE catalogs and locked pre-outcome scores, but family
labels are incomplete after GPU allocation failures in long-lived evaluators.
The active repair is independent-process evaluator sharding. No π4 training is
authorized, and final TEST/JCE/JEL remains untouched.

See [CURRENT_STATUS](JIT/docs/CURRENT_STATUS.md) for exact counts and next steps.

## Repository layout

```text
JIT/src/jit_dvgc/   scientific and runtime logic
JIT/cli/            thin command-line entry points
JIT/tests/          contract and regression tests
JIT/configs/        declared experiment configurations
JIT/docs/           current authority, protocol, reports and handoffs
JIT/runs/           run artifacts; lightweight evidence is indexed in Git
assets/             fixed MuJoCo task assets
```

## Fixed task

- XML: `assets/orange_bike_4kg_horizontal.xml`
- payload: 2 kg
- simulation substep: 0.005 s
- control interval: 0.020 s
- action order: steer, rear-wheel drive, hip, knee
- hip/knee limits: +/-30 N m

Do not silently modify physics, task geometry, reward or actuator order.

## Paper direction

The working hypothesis is that reachability-filtered reset curricula improve a
single unified jumping policy at controlled total interaction cost. This has not
yet been demonstrated. Required evidence includes same-budget baselines,
independent training seeds, group-aware uncertainty, full interaction accounting
and a frozen independent final distribution.

## Read order

1. [AGENTS.md](AGENTS.md)
2. [JIT/AGENTS.md](JIT/AGENTS.md)
3. [Current status](JIT/docs/CURRENT_STATUS.md)
4. [Project definition](PROJECT.md)
5. [Iteration protocol](JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md)
6. [Technical handoff](JIT/docs/CODEX_HANDOFF_20260904.md)
7. [Scientific review response](JIT/docs/JIT_SCIENTIFIC_REVIEW_RESPONSE_20260905.md)
