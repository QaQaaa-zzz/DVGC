# DVGC repository authority

## Current research truth — 2026-09-05

DVGC/JIT studies an empirical, trajectory-centered reset curriculum for one
fixed single-track two-wheeled robot jump. The active experiment is conditioned
on the locked ground jump start at `root x = 2.5 m`; it does not currently claim
reachability from the natural episode reset.

The five objects below must remain separate:

```text
A  forward arrival evidence
   pi_0 + bounded lookback action perturbation reaches an exact state by env.step

E  policy-family landing witness
   from that exact state, any frozen member of {pi_0, pi_1, pi_2} reaches the
   first valid landing before physical failure

J  physical capability occupancy
   positive A intersect E states projected into predeclared physical cells

S  raw/control Soft Tube
   reset/replay snapshots, including retained historical core rows

r  single-Actor realization
   one frozen Actor's measured success on a common locked state panel
```

An `A` state is conditional on the declared jump start and proposer. An `E`
witness may concatenate a `pi_0` prefix with another family's suffix, so it is
not evidence that one Actor executes the full chain. A physical cell is an
occupancy statistic, not a continuous feasible region. A raw Tube is not a
causal reachable set. The final runtime target remains one unified Actor with no
expert switching.

JIT does not claim a formal reachability set, viability kernel, invariant or
certified safe set, complete physical jump limit, or final RA-L evidence.

## Active fixed experiment contract

- branch: `agent/two-phase-soft-tube`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- locked XML identity:
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- simulation substep: 0.005 s
- control interval: 0.020 s = 50 Hz
- hip/knee actuator range: +/-30 N m
- action order: `[steer, rear-wheel drive, hip, knee]`
- centerline: locked real-frame `pi_0` jump-start trajectory
- proposer: frozen `pi_0`
- evaluator family: frozen `{pi_0, pi_1, pi_2}`
- positive label: any evaluator reaches first valid landing before physical
  failure
- post-landing recovery: outside the active label
- final TEST/JCE/JEL: untouched

The centerline is a longitudinal exploration scaffold only. It is not an Actor
intent, tracking reference, reward target, interpolated trajectory, or proof of
reachability between samples. Its nominal range is `x = 2.5..4.2 m` at 0.1 m
spacing, ending earlier at the first valid landing. Only captured simulator
frames are allowed.

Prospective candidates must be created by authoritative `env.step` transitions
from the locked jump-start snapshot using the declared `pi_0` prefix and bounded
lookback perturbation. Restoring an RSI/Tube state cannot establish `A`.
Restoration is permitted only after arrival, to evaluate the exact state's
landing outcome.

## Data roles

- `TRAIN`: may contribute observed family-positive states to replay.
- `CALIBRATION`: threshold calibration and diagnostics only.
- `ACCEPTANCE`: locked development comparison only; once used for selection or
  method decisions it remains development data.
- final `TEST/JCE/JEL`: untouched until the method, policy and stopping rule are
  frozen.

Role assignment and removal of exact TRAIN/Tube overlaps must be outcome blind.
Report excluded counts and reasons. Adjacent states from one trajectory or
perturbation family are correlated; candidate count is not an independent
sample count.

## Current evidence boundary

The first fixed-jump-start family round is complete:

```text
family-positive TRAIN candidates                 1230 / 1258
Tube_2 -> Tube_3 raw increment                  1159 rows
new causal TRAIN root cells                      713
Tube_3 control root-cell increment               714
pi_3 training                                    completed, 10,009,600 transitions
```

`pi_3` was historically registered as an engineering selection, but its core
comparison is not a fair prospective policy-selection result: the stored gate
compares a `stable_recovery` baseline core against a `first_valid_landing`
candidate core. Preserve all artifacts and the 30 regressions/89 improvements,
but do not treat that selection as current scientific authority. A same-endpoint
reevaluation is retrospective diagnostic evidence only.

The fitted upstream landing predictor is advisory. Its old ACCEPTANCE AUC was
about 0.8925, but it accepted 6 of 9 observed negatives at the locked threshold.
It cannot establish arrival, label a state, filter Tube admission, or support a
safety claim. Downstream was single-class all-positive and was not fit.

The expanded next-round catalogs and pre-outcome predictor scores are locked,
but family labeling is incomplete after GPU allocation failures. No `pi_4`
training is authorized. Complete memory-bounded evaluator shards, audit the
fresh predictions, repair the same-endpoint comparison protocol, and decide the
controlled experiment before further policy training.

## Paper claim direction

The strongest defensible thesis to test is:

> Reachability-filtered reset curricula can improve one unified jumping policy
> at controlled total interaction cost.

This is a hypothesis, not a demonstrated conclusion. Required comparisons
include continued PPO/fixed curriculum, static successful Tube-RSI,
fixed-grid forward acquisition, an RSI-only candidate baseline, and the active
iterative method. Report all acquisition, family-labeling, training, selection
and failed-retry interactions. Use independent training seeds (three for pilot,
five preferred for the main result), group-aware intervals and a frozen final
distribution. Tube growth alone is not the primary result.

## Implementation ownership

- scientific logic belongs in `JIT/src/jit_dvgc/`;
- command-line entry points stay thin in `JIT/cli/`;
- tests belong in `JIT/tests/`;
- durable reports and authority belong in `JIT/docs/`;
- modify the existing implementation for an existing capability instead of
  creating iteration-numbered duplicate source files;
- schemas/configs drive iteration identity; do not hard-code policy numbers,
  seeds, retries or checkpoint paths into reusable scientific logic.

## Repository and Git safety

- preserve unrelated user work;
- never reset, clean, stash, rebase, force-push or silently overwrite files;
- use `/home/qy/mujoco_playground/.venv/bin/python`;
- compile and run targeted tests after structural changes;
- never rewrite an immutable historical artifact to manufacture a pass;
- do not repeatedly recalculate SHA-256 values already locked in manifests or
  authority documents; routine code should reuse recorded identities;
- retain automatic artifact provenance and self-hash checks, and calculate a
  hash manually only to diagnose a concrete identity problem.

## Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `PROJECT.md`
5. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `JIT/docs/CODE_ORGANIZATION.md`
8. `JIT/docs/JIT_SCIENTIFIC_REVIEW_RESPONSE_20260905.md`
9. `JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md` (historical
   redesign record)
