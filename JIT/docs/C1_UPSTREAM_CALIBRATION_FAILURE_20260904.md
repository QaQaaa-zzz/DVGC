# C1 upstream calibration failure — 2026-09-04

## Current evidence

The pi1 -> pi2 iteration has advanced through the fresh v3c ACCEPTANCE challenge.
The v3c bank completed before any pi2 candidate training and passed its predeclared
support gate:

- upstream: 516 candidates = 511 positive + 5 negative, negatives from 2 parent groups;
- downstream: 70 candidates = 61 positive + 9 negative, negatives from 1 parent group;
- total negative parent groups: 3;
- no TEST/final data and no expert switching.

C1 fitting then used the preserved v3 TRAIN role and v3b repaired CALIBRATION role.
The inherited bootstrap requirement that every TRAIN parent contain both labels was
handled by the separately documented k>=1 observed-cell weighting compatibility
rule; TRAIN rows, role membership, architecture, optimizer and calibration contract
were not changed.

The resulting upstream field was fitted and its disjoint calibration artifact was
written, but the fixed calibration contract returned `calibration_passed=false` and
stopped the workflow before downstream C1 fitting, Tube2 construction or pi2
training.

Therefore the current state is:

```text
v3 TRAIN                         PASS
v3b CALIBRATION support          PASS enough to fit/calibrate
v3c ACCEPTANCE                   PASS
C_up^1 parameter fit             completed partial artifact
C_up^1 fixed calibration         FAIL
C_down^1                         not attempted in this run
Tube2                            not constructed
pi2                              not trained
```

## Required handling

The failed `continuation_C1/upstream` directory is scientific evidence and must be
preserved.  Do not delete or overwrite it.  Do not lower AUC/recall requirements,
change the threshold rule, refit on CALIBRATION, reuse ACCEPTANCE, inspect TEST, or
reroll the same calibration bank under a different seed.

Before any method revision, run the read-only diagnostic:

`JIT/cli/analyze_iterative_calibration_failure.py`

It reports which fixed gate failed:

1. ROC AUC below 0.70 -> ranking/generalization failure;
2. AUC passes but positive recall below 0.20 -> conservative-threshold score-overlap failure;
3. global metrics pass but one or more calibration parents have zero accepted positive -> parent-local threshold-coverage failure;
4. any other contract failure is surfaced separately.

The diagnostic deterministically re-scores the already-fitted frozen partial field
on the already-labeled calibration rows only to verify the stored artifact.  It
performs no model refit, threshold change, state acquisition, continuation labeling,
or artifact mutation.

A scientific method revision, if required, must be predeclared only after this
failure mode is identified.  The original failed field/calibration remains retained
as evidence and must not be retroactively claimed as PASS.
