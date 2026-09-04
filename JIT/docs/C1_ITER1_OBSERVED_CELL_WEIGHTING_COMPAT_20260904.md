# Iteration-1 C1 observed-parent-label weighting compatibility repair

## Status

The fresh v3c ACCEPTANCE challenge bank passed its predeclared support gate before
any pi2 candidate training:

- upstream: 516 candidates = 511 positive + 5 negative, 3 parent groups, negatives in 2 groups;
- downstream: 70 candidates = 61 positive + 9 negative, 1 parent group;
- total negative parent groups: 3;
- no TEST/final data and no expert switching.

The workflow then stopped at `fit_and_calibrate_Ck` before any C1 model parameters
were fit. The error was:

```text
ValueError: TRAIN weighting requires both labels in every parent group
```

## Root cause

The generic k>=1 frontier TRAIN contract and the inherited bootstrap refit helper
had incompatible support assumptions.

The generic automatic frontier accepts a phase when it has, at minimum, both
classes in aggregate and sufficient parent-group diversity. It does not require
every individual parent group to contain both labels.

The bootstrap shared-field refit was built from a curated Iteration-0 TRAIN bank
in which every parent group did contain both labels. Its shared helper
`_cell_balanced_weights` therefore rejected any outcome-pure parent group. The
k>=1 iterative fitter reused that helper directly without first imposing the
bootstrap-only per-parent two-class validator.

This is an implementation-contract mismatch, not new evidence that the v3 TRAIN
role is invalid.

## Repair

For k>=1 iterative fitting only, use equal loss mass for every **observed**
`(parent_group_id, label)` cell.

Consequences:

- when every parent contains both labels, weights are bit-for-bit identical to the historical rule;
- an outcome-pure parent contributes its one observed cell instead of aborting the fit;
- no missing label cell is fabricated;
- no TRAIN row is dropped, duplicated, resampled, relabeled, or moved across roles;
- CALIBRATION and ACCEPTANCE rows remain excluded from parameter fitting;
- the 76->8 tanh->1 architecture, optimizer, steps, regularization, seeds, and calibration rule are unchanged;
- the bootstrap C0 strict per-parent two-class validator remains unchanged;
- TEST/JCE/JEL remain untouched.

The compatibility implementation is:

- `JIT/src/jit_dvgc/iterative_weighting_compat.py`
- `JIT/cli/fit_iterative_continuation_fields.py`

The wrapper writes a self-hashed sibling weighting-contract artifact containing
per-parent TRAIN class counts before fitting.

## Preservation of the failed attempt

The observed failure occurred before `_fit_phase` initialized or wrote model
parameters: the top-level `continuation_C1` directory was created, then the
weight calculation raised.

On retry, the compatibility wrapper may archive **only an empty** failed output
directory to:

```text
continuation_C1_failed_prefit_parent_cell_contract
```

and writes an `engineering_failure.json` marker. Any nonempty partial output is
refused and must be preserved for explicit diagnosis.

## Claim boundary

This repair does not claim that v3/v3b/v3c was predeclared with sparse parent
cells. It records a post-failure engineering compatibility correction made
before any C1 model fit and before any pi2 training.

It does not authorize Tube2 by itself. C1 must still fit and the unchanged,
disjoint CALIBRATION contract must pass before Tube2 construction is authorized.
