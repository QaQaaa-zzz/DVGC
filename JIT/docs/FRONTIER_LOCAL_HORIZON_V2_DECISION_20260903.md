# JIT frontier local-horizon v2 decision — 2026-09-03

## Status

This record supersedes the **frontier acquisition protocol only** for the next
`pi_1 -> C^1 -> Tube_2 -> pi_2` attempt. It does not change the selected policy,
Tube identity, continuation-label semantics, data-role isolation, Tube
construction rule, PPO method, acceptance gate, physics, or TEST isolation.

The failed v1 automatic workflow and all of its artifacts must be preserved.
Do not mutate its `frontier_plan.json`, role outputs, workflow state, or logs.

## Observed v1 failure

The first automatic Iteration-1 frontier TRAIN role used the predeclared probe
panel:

```text
strengths = 0.025 / 0.05 / 0.10
durations = 4 / 8 / 16 / 32 ticks
control rate = 50 Hz
```

The role had 12 TRAIN anchors and at most 1,152 action-probe variants. Acquisition
and continuation labeling completed sufficiently for the logical-role check to
run, but the downstream logical TRAIN support was:

```text
candidate_count = 0
positive_count = 0
negative_count = 0
parent_group_count = 0
```

Therefore the v1 frontier protocol is not usable for `C_down^1`; the workflow
correctly stopped before `C^1`, Tube_2, or pi_2 was constructed/trained.

This is **not** a repair02 policy failure, a core-regression result, a PPO failure,
or a GPU/OOM failure. It is a frontier-acquisition resolution/support failure.

## Why the original run felt unnecessarily slow

The original control logic checked two-phase TRAIN support only **after**
continuation labeling. Once acquisition had produced zero downstream candidates,
no possible 400-tick labeling result could satisfy the later downstream TRAIN
contract, yet upstream candidates were still labeled before the workflow stopped.

The production control logic is now changed so unlabeled acquisition receives a
necessary-condition preflight before continuation labeling:

- TRAIN: each phase must have at least 40 acquired candidates and at least 3
  parent groups, because the fixed later contract requires >=20 positive and
  >=20 negative labels across >=3 groups;
- CALIBRATION/ACCEPTANCE: each phase must have at least 2 candidates and at least
  1 parent group, because both label classes are required later.

If those necessary structural conditions fail, the workflow stops immediately
without spending the expensive continuation-label budget.

## Scientific revision

The v1 shortest perturbation duration was 4 ticks = 80 ms at 50 Hz. For the
late descent/recovery frontier this is too coarse to be treated as the only
local perturbation scale after v1 produced no downstream frontier candidates.

The predeclared v2 acquisition panel changes only the perturbation-duration grid:

```text
v1: 4 / 8 / 16 / 32 ticks = 80 / 160 / 320 / 640 ms
v2: 1 / 2 / 4 / 8 ticks   = 20 / 40 / 80 / 160 ms
```

The following remain unchanged:

- selected repair02 `pi_1` identity;
- Tube_1 identity and newest-shell-only parent pool;
- exact parent-group role assignment inherited from the v1 predeclaration;
- TRAIN / CALIBRATION / ACCEPTANCE parent disjointness;
- action dimensions and signs;
- perturbation strengths;
- acquisition and labeling seeds;
- deterministic frozen-policy evaluation;
- continuation-label horizon = 400 ticks;
- real-dynamics-only state generation;
- C^1 architecture/calibration contract;
- Tube_2 full Tube_1 retention rule;
- 75/25 retained-source/newest-expansion replay inside 90/10 Tube/natural reset;
- strict pre-candidate locked-baseline pi_1 -> pi_2 acceptance gate;
- final TEST/JCE/JEL isolation.

This is an explicit protocol revision after a failed v1 data-acquisition attempt,
not an automatic threshold/hyperparameter repair. No v2 outcome may be inspected
before the revised plan is written and self-hashed.

## Production entry point

The revision is created with:

```bash
python JIT/cli/run_iterative_frontier_protocol.py \
  revise-plan-local-horizon-v2 \
  --source-plan <failed-v1-frontier-plan.json> \
  --output <new-v2-work-root>/frontier_plan.json
```

The v2 plan records:

- `protocol_revision.name = local_horizon_v2`;
- the superseded v1 plan path and SHA-256;
- changed fields;
- unchanged scientific contracts;
- `revision_predeclared_before_v2_outcomes = true`;
- `automatic_repair = false`.

After the v2 plan exists, run a **new workflow root/tag**. Do not reuse or edit
the failed v1 workflow state/config.

## Decision after v2

The workflow may proceed to C^1 only if TRAIN and disjoint CALIBRATION satisfy
the existing two-phase support and calibration contracts. If v2 again produces
insufficient downstream acquisition support, stop again. Do not shorten durations,
change strengths, move parent groups across roles, fall back to full Tube, reuse
bootstrap `V_down`, or weaken C^1 support thresholds automatically. A second
failure would require a distinct parent-generation/saturation decision.
