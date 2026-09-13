# JIT v3b upstream calibration hard-negative repair — 2026-09-03

## Evidence entering this decision

The phase-specific two-axis v3 TRAIN role passed the fixed continuation-support gate:

- upstream: 821 candidates = 785 positive + 36 negative across 9 parent groups;
- downstream: 210 candidates = 182 positive + 28 negative across 3 parent groups.

The original v3 CALIBRATION role then failed only in upstream:

- upstream: 249 candidates = 249 positive + 0 negative across 3 parent groups;
- downstream: 70 candidates = 61 positive + 9 negative across 1 parent group.

Read-only role-balance diagnostics established that upstream CALIBRATION was not merely shifted to a disjoint high-score region. Two of its three parent anchors (approximately 0.934996 and 0.935256) lie inside the dense low-score range occupied by v3 TRAIN parents, yet every accepted single-axis calibration candidate remained continuation-positive. All eight one-axis directions and every 0.025/0.05/0.10 x 1/2/4/8 cell were all-positive. The strongest/longest accepted cell also contained no negative labels.

Downstream CALIBRATION already brackets continuation outcomes: the 0.30 x 2 cell contains 4 negatives and the 0.50 x 2 cell contains 5 negatives. No downstream repair is justified.

## Decision

Preserve the successful v3 TRAIN artifact and preserve the failed v3 CALIBRATION artifact unchanged. Do not move any parent group between TRAIN, CALIBRATION, or ACCEPTANCE.

Open one new predeclared calibration-only supplement, named:

`upstream_sparse_two_axis_calibration_v3b`

The supplement reuses only the three already-locked upstream CALIBRATION parent groups and applies the historical sparse-two-axis bracketing family:

- actions: steer, rear-wheel drive, hip, knee;
- active action dimensions: 2;
- all dimension pairs and all +/- sign combinations;
- strengths: 0.15, 0.30, 0.50;
- durations: 2, 4, 8 ticks;
- real dynamics only;
- frozen engineering pi_1 only;
- continuation horizon unchanged at 400 ticks;
- no TEST/final data;
- no policy training;
- no expert switching.

This is a new post-failure protocol revision. It does not retroactively repair the original v3 CALIBRATION round.

## Why the supplement is calibration-only

The v3 upstream TRAIN panel already produced both continuation classes and therefore remains scientifically usable for fitting C_up^1. Re-running or replacing TRAIN would discard valid evidence without addressing the observed failure.

The failed object is threshold calibration: the original disjoint upstream CALIBRATION rows contain no negative state from which the fixed rule

`accept iff score > max(calibration negative score)`

can define a threshold.

The new supplement therefore changes only the difficulty of the challenge applied to the already-designated calibration parents. Model weights remain fit on TRAIN only.

## Fixed v3b gate

Before any v3b outcome is observed, the repair requires the unique supplement rows to contain at least:

- 5 upstream continuation-negative states;
- negatives from at least 2 distinct upstream CALIBRATION parent groups.

The combined repaired calibration role must still contain both classes in both phases. Original downstream calibration rows are reused without relabeling.

If this fixed v3b gate fails, stop. Do not automatically increase strengths, add a third action dimension, extend durations, move TRAIN/ACCEPTANCE parents into CALIBRATION, lower the negative-support gate, change the 400-tick label definition, or touch TEST/final evidence. A further failure requires a new parent-generation / calibration-domain decision.

## Artifact semantics

The repair CLI creates new paths only:

- `calibration_repair_plan.json` — self-hashed pre-outcome v3b protocol;
- `frontier_calibration_v3b_repaired/` — new completed logical calibration role if the gate passes;
- `workflow_v3b_calibration_repair.json` — revised workflow pointing later fit/isolation stages at the repaired calibration role.

The original `frontier_calibration/` and original `workflow.json` remain immutable evidence.

The repaired role retains the original v3 `plan_sha256` for parent-role identity and records the separate v3b repair-plan SHA for acquisition provenance.

## Claim boundary

This remains empirical policy-conditioned continuation calibration under nominal dynamics. It is not a certified probability model, viability kernel, invariant set, reachability proof, JCE, or JEL result.
