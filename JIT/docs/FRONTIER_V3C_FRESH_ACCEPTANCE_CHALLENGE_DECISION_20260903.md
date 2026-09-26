# V3C Fresh Acceptance Challenge Decision — 2026-09-03

## Status

The phase-specific v3 TRAIN role passed after memory-bounded labeling. The v3b upstream calibration hard-negative supplement was prepared before new outcomes and the revised workflow progressed past calibration into the original v3 ACCEPTANCE stage. Therefore the v3b calibration repair completed sufficiently for the workflow completion gate.

The original v3 ACCEPTANCE role then completed acquisition/labeling but failed logical support in upstream:

- upstream candidate_count = 254
- upstream positive_count = 254
- upstream negative_count = 0
- upstream parent_group_count = 3
- total acceptance labeling candidates reported by the workflow = 324
- downstream candidates reported by the workflow = 70

The original v3 ACCEPTANCE role is preserved as failed evidence and must not be represented as a PASS.

## Why this blocks pi2

`iterative_acceptance_gate.lock_baseline` requires the completed acceptance role to contain baseline-negative states in both upstream and downstream. It also requires baseline negatives from at least two parent groups. These negatives are locked before candidate training and later reused with exact baseline labeling seed/candidate-index PRNG identity.

An all-positive upstream acceptance role therefore cannot be used to create the required baseline lock. pi2 training must not start from the failed bank.

## V3C decision

Before any pi2 candidate training, create one fresh acceptance-only challenge bank named:

`fresh_sparse_two_axis_acceptance_v3c`

The old failed v3 acceptance artifact remains immutable and is not merged into the new bank.

### Parent membership

Keep the already-predeclared ACCEPTANCE parent groups exactly unchanged.

- no TRAIN parent is moved
- no CALIBRATION parent is moved
- no ACCEPTANCE parent is moved into TRAIN or CALIBRATION
- no TEST/JCE/JEL/final data is touched

### Probe family

Use the historical sparse two-axis family in both phases:

- actions: steer, rear_wheel_drive, hip, knee
- active action dimensions: 2
- signs: -1/+1 on each active dimension
- strengths: 0.15, 0.30, 0.50
- durations: 2, 4, 8 ticks

This is 24 directions × 3 strengths × 3 durations = 216 variants per anchor.

Upstream is strengthened from the failed single-axis v3 panel. Downstream keeps the same two-axis family already used by v3; the v3c bank is nevertheless generated and labeled as one fresh unified logical acceptance bank so that global candidate-index PRNG identity is exact.

### Seeds and continuation definition

Reuse the original source plan ACCEPTANCE seeds, especially the original acceptance labeling seed. Do not introduce a new labeling seed because `lock_baseline` reconstructs baseline policy keys from the source frontier plan.

Keep max continuation horizon = 400 ticks.

### Unified labeling requirement

Do not independently label upstream/downstream and concatenate afterward. The v3c phase acquisitions are merged into one root catalog first, then one logical continuation-label protocol is executed over the merged candidate order. This preserves:

`candidate_key = fold_in(PRNGKey(original_acceptance_labeling_seed), global_candidate_index)`

for later immutable baseline locking and candidate comparison.

## Pre-pi2 support gate

The fresh v3c bank is accepted for baseline locking only if all of the following are true before pi2 training:

- each phase has at least 1 positive
- each phase has at least 5 negatives
- upstream negatives cover at least 2 parent groups
- downstream negatives cover at least 1 parent group
- total negative parent groups across phases >= 3

These conditions are stricter than the minimum implementation requirement in `lock_baseline` and are intended to prevent a formally lockable but pathologically thin boundary bank.

If the gate fails, STOP. Do not automatically increase strength, move parents, add 3/4-axis perturbations, extend duration, reroll seeds, lower support thresholds, reuse TRAIN/CALIBRATION negatives, or begin pi2 training.

## Claim boundary

A successful v3c bank authorizes only a fresh candidate-blind empirical acceptance baseline lock for the future pi2 candidate.

It does not retroactively make the original v3 acceptance protocol pass. It is not a certified safe set, viability kernel, invariant set, JCE/JEL result, or formal reachability proof.

## Workflow

`JIT/cli/acceptance_challenge_repair.py prepare` creates:

- a self-hashed v3c challenge plan
- a new acceptance output root
- a new workflow configuration with a fresh state directory

The revised workflow keeps completed TRAIN and repaired CALIBRATION artifacts, replaces only the failed `frontier_acceptance` stage, rewrites later acceptance consumers to the v3c root, and then proceeds to continuation-field fitting, Tube2 construction, baseline locking, and pi2 training only after the v3c support gate passes.
