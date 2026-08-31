# Iteration-0 TRAIN Evidence Freeze Design

## Purpose

Freeze the completed `pi_0`-conditioned upstream and downstream transition-band
labels before any expansion-validation interaction or continuation-field fit.
The frozen artifact is TRAIN-only input evidence for `C_up^0` and `C_down^0`;
it is not a continuation model, Tube, policy, final-evaluation bank, or safety
claim.

## Artifact contract

One stable builder validates the terminal refinement summary and accumulated
label file against a predeclared config. It rejects any non-TRAIN row, policy or
iteration drift, duplicate physical state, invalid/non-finite observation,
readiness mismatch, missing class/parent-group support, or claim-boundary drift.

The output contains:

- an exact copy of the accumulated TRAIN labels;
- a manifest binding the source file hashes, frozen `pi_0`, XML, `Tube_0`, and
  terminal search protocol;
- recomputed phase/class/parent-group counts and the TRAIN parent-group denylist
  that later validation must exclude;
- zero new environment interactions and zero training transitions.

The copied labels and manifest are self-hashed and revalidated by the loader.

## Leakage boundary

Expansion validation must use parent groups absent from the frozen TRAIN
denylist and must also perform physical-state and near-duplicate audits. TEST and
final-envelope evaluation remain untouched. `C^0` fitting/calibration remains
blocked until a separately predeclared group-disjoint validation protocol is
implemented and completed.
