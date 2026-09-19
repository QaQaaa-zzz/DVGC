# C1 upstream calibration ranking-generalization failure — 2026-09-04

## Status

The pi_1-conditioned frontier pipeline has progressed through a completed fresh
v3c acceptance bank.  The first C_up^1 fit then failed its independent
calibration contract after the k>=1 observed-parent-label-cell weighting
compatibility repair.

This is a scientific model/calibration failure, not a GPU, workflow, labeling,
or acceptance failure.

## Frozen evidence

Selected engineering pi_1 remains repair02:

- actor SHA-256: `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`
- payload SHA-256: `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`

Original v3 TRAIN upstream:

- 821 candidates
- 785 positive
- 36 negative
- 9 parent groups
- 5 outcome-pure positive parent groups

Negative support by checkpoint-domain prefix is highly imbalanced:

- `transition_4988928`: 33 negatives across its three TRAIN parents
- `transition_7987200`: 0 negatives across its three TRAIN parents
- `transition_9977856`: 3 negatives across its three TRAIN parents

The repaired v3b upstream CALIBRATION artifact contains:

- 739 candidates
- 733 positive
- 6 negative
- 3 parent groups

Its negative support is distributed in the opposite way:

- `transition_4988928__1000001`: 164 positive / 0 negative
- `transition_7987200__1000002`: 274 positive / 3 negative
- `transition_9977856__1000003`: 295 positive / 3 negative

The v3c acceptance bank passed its predeclared support gate:

- upstream: 516 candidates = 511 positive / 5 negative, negative support across 2 parents
- downstream: 70 candidates = 61 positive / 9 negative, negative support across 1 parent
- total negative parent groups: 3

No pi_2 candidate has been trained.

## Failed C_up^1 calibration

The fitted 76->8 tanh->1 C_up^1 field produced:

- calibration ROC AUC: `0.6634834015461574`
- required minimum ROC AUC: `0.70`
- positive recall at the conservative max-negative threshold: `0.23465211459754434`
- required minimum recall: `0.20`
- threshold: `0.9968386901320423`
- accepted negative count: `0`

Therefore recall and zero-accepted-negative constraints pass, while ranking AUC
fails.

Parent-local threshold coverage also fails:

- `transition_4988928__1000001` has 164/164 positive labels but zero accepted
  positives; its maximum positive score is `0.9753890125845645`, below the
  global threshold.
- the other two calibration parents each contain three negatives, including
  negative scores near 0.996, which set the conservative threshold.

The diagnostic category is:

`ranking_generalization_failure`

with a simultaneous `accepted_positive_in_every_parent` failure.

## Scientific interpretation

The active failure cannot be repaired by simply lowering the threshold.  The
current threshold is exactly the maximum calibration-negative score; lowering
it would violate the fixed zero-accepted-negative contract.

The failure also should not immediately be interpreted as evidence that the
frozen 625-parameter architecture is intrinsically too small.

There is a structural support mismatch:

1. original v3 upstream TRAIN used the weak single-axis panel;
2. v3b was introduced only after the original upstream calibration bank was
   all-positive;
3. the v3b upstream hard-negative supplement uses the historical strong sparse
   two-axis panel;
4. the six calibration negatives are therefore drawn from a challenge family
   that is absent from the original upstream TRAIN acquisition;
5. TRAIN negative support is concentrated in the `transition_4988928` domain,
   while repaired CALIBRATION negatives occur only in the other two domains.

This creates both acquisition-family and checkpoint-domain label-support shift.
A more expressive model may still fail if trained on the same support mismatch.

## Required next decision

Do not:

- lower the AUC or recall thresholds;
- lower the conservative max-negative threshold;
- reuse current calibration rows to choose a new model and then report them as
  independent calibration;
- move existing CALIBRATION or ACCEPTANCE parents into TRAIN;
- use ACCEPTANCE outcomes for model selection;
- touch TEST/JCE/JEL;
- train pi_2.

First determine whether the Tube_1 newest shell contains unused parent groups
outside the original 15-parent cap.  Use:

`JIT/cli/analyze_frontier_reserve_parents.py`

This diagnostic reads the source Tube and existing plan only.  It reads no
continuation outcomes and performs no environment interaction.

### If sufficient fresh reserve parents exist

A new method revision may be predeclared with all of the following safeguards:

1. preserve the failed C_up^1 artifact and consumed v3b calibration bank;
2. add an upstream TRAIN hard-negative acquisition on the already-TRAIN parent
   groups using the strong sparse two-axis family so parameter fitting sees the
   same type of hard neighborhood that exposed the failure;
3. keep all original TRAIN rows; do not move role membership;
4. select fresh, previously unused newest-shell parent groups for a new
   calibration bank before inspecting any new outcomes;
5. prefer checkpoint-domain coverage across the three upstream transition
   prefixes if reserve support allows it;
6. keep the 400-tick continuation definition and real dynamics;
7. use a new fixed acquisition/label seed pair and immutable self-hashed repair
   plan;
8. do not reuse the consumed v3b upstream calibration for pass/fail of the
   revised field;
9. keep the existing v3c ACCEPTANCE bank locked and untouched, provided the new
   TRAIN/CALIBRATION parents remain disjoint from it;
10. only after a fresh calibration PASS may Tube_2 construction resume.

### If sufficient fresh reserve parents do not exist

Stop for a distinct real-dynamics parent-generation decision.  Do not recycle
old calibration/acceptance parents or weaken role isolation.  The next protocol
must create new parent trajectories/snapshots before any revised C_up^1 can be
validated independently.

## Claim boundary

Current state remains:

- v3 TRAIN: PASS
- v3b calibration support: completed/consumed
- v3c acceptance: PASS and locked as candidate-blind evidence
- first C_up^1: fitted but calibration FAIL
- C_down^1: not yet fit in the failed run
- Tube_2: not constructed
- pi_2: not trained
- TEST/JCE/JEL: untouched

No formal safe-set, viability-kernel, reachability, or certified-probability
claim is authorized.
