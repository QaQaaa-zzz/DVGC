# JIT frontier v3: TRAIN pass, upstream calibration all-positive — 2026-09-03

## Status

The phase-specific two-axis v3 revision solved the failed downstream TRAIN frontier-identification problem.

Completed v3 TRAIN evidence:

```text
candidate_count = 1031

upstream:
  candidate_count = 821
  positive = 785
  negative = 36
  parent groups = 9

downstream:
  candidate_count = 210
  positive = 182
  negative = 28
  parent groups = 3
```

Therefore the fixed TRAIN support requirement is satisfied in both phases:

```text
positive >= 20
negative >= 20
parent groups >= 3
```

The downstream sparse two-axis revision is empirically successful as a TRAIN frontier-bracketing method for this pi_1/Tube_1 iteration. Do not reopen single-axis downstream probing for this round.

However, the disjoint CALIBRATION role failed after all 319 candidates were labeled:

```text
upstream calibration:
  candidate_count = 249
  positive = 249
  negative = 0
  parent groups = 3
```

The workflow therefore stopped before fitting/calibrating C^1. C^1, Tube_2, and pi_2 remain unauthorized.

## What must not be done

Do not:

- borrow the 36 upstream TRAIN negatives for threshold calibration;
- weaken the requirement that calibration contain both labels;
- set a threshold from positive-only calibration rows;
- move exposed TRAIN, CALIBRATION, or ACCEPTANCE parent groups between roles;
- inspect ACCEPTANCE outcomes to repair calibration;
- alter the 400-tick continuation definition;
- introduce TEST/final evidence;
- mutate the completed v3 TRAIN or failed v3 CALIBRATION artifacts in place.

The continuation-field contract explicitly calibrates an exclusive threshold above the maximum disjoint calibration-negative score. With zero upstream calibration negatives, that threshold is undefined by the declared method.

## Structural split concern

The current generic plan selection orders newest-shell parent groups by increasing `value_score`, then assigns the repeated role pattern:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE
```

This is outcome-blind and parent-disjoint, but it is not score-balanced. Within each five-parent block, CALIBRATION and ACCEPTANCE systematically receive later/higher-score parents than the first three TRAIN parents. Because the score is itself a continuation-support proxy inherited from Tube construction, this deterministic ordering can create class-support shift between roles even before any labels are observed.

The present upstream result is consistent with that concern:

- TRAIN has two classes (785/36) across 9 parents;
- CALIBRATION is all-positive (249/0) across 3 different parents;
- both roles used the same upstream v3 single-axis probe family.

This is not yet sufficient to claim the role ordering is the sole cause. The exact parent anchor-score distributions and probe-cell/direction support must be inspected first.

## Required read-only diagnostic before a new calibration protocol

Run the newly added role-balance diagnostic on the completed TRAIN and CALIBRATION roots. It performs zero new rollouts and zero new labels and works with the phase-specific v3 acquisition protocol.

Required outputs include, separately by phase and role:

- candidate/positive/negative/parent counts;
- phase-specific probe panel;
- acquisition acceptance/exclusion counts;
- support by strength/duration;
- support by direction;
- support by parent group;
- each exposed parent's original anchor `value_score`;
- role-level anchor-score min/max/mean/sorted values;
- parent-group overlap audit.

The diagnostic may inform a new predeclared calibration protocol, but it may not retroactively change v3 role membership or authorize C^1.

## Decision boundary after diagnostic

Preferred decision order:

1. If CALIBRATION upstream parents are materially shifted to higher anchor scores than TRAIN and the same upstream panel brackets failures in TRAIN but not CALIBRATION, treat this primarily as a role-parent coverage/split-design failure. Define a fresh, parent-disjoint, outcome-blind calibration supplementation/replacement protocol from previously unexposed newest-shell parents with explicit score-balanced selection. Do not strengthen the probe merely to manufacture negatives.
2. If CALIBRATION upstream anchor-score coverage overlaps TRAIN well but strong/long cells are still uniformly positive, then a calibration-specific boundary-bracketing acquisition revision may be justified, fully predeclared before new outcomes.
3. If there are no sufficient previously unexposed parent groups for a scientifically independent calibration repair, stop and escalate to a new parent-generation decision rather than reassigning exposed parents.
4. If downstream CALIBRATION also lacks two-class support, repair design must cover both phases in one new predeclared calibration protocol rather than patching phases sequentially after outcomes.

Any repaired calibration evidence must remain disjoint from TRAIN and ACCEPTANCE and must never enter Tube_2 directly.
