# C1 standard-MLP same-data architecture comparison — 2026-09-04

## Decision

After the first `C_up^1` 76->8 tanh->1 field produced calibration ROC AUC
`0.6634834015461574`, the user authorized a controlled architecture comparison
that intentionally reuses the exact same Iteration-1 TRAIN and repaired v3b
CALIBRATION evidence.

The purpose is to isolate model capacity from data acquisition: change the
network architecture and nothing else.

## Models

Historical profile:

- profile: `legacy_tiny_tanh`
- architecture: `76 -> 8 tanh -> 1`
- parameters: 625

Comparison profile:

- profile: `standard_mlp_64x64_tanh`
- architecture: `76 -> 64 tanh -> 64 tanh -> 1`
- parameters: 9,153

## Fixed comparison contract

The comparison profile keeps unchanged:

- selected frozen engineering `pi_1 = repair02` identity;
- original v3 TRAIN role membership and labels;
- repaired v3b CALIBRATION role membership and labels;
- k>=1 observed `(parent_group_id, label)` cell balancing;
- TRAIN-only z-score normalization and +/-10 clipping;
- Adam full-batch optimizer;
- learning rate `0.01`;
- 4,000 optimizer steps;
- L2 weight `0.01`;
- phase-specific seed derivation;
- calibration threshold rule: score strictly greater than maximum calibration-negative score;
- minimum ROC AUC `0.70`;
- minimum positive recall `0.20`;
- zero accepted calibration negatives;
- accepted-positive support in every calibration parent;
- 400-tick continuation definition and all underlying real-dynamics labels;
- TEST/JCE/JEL exclusion;
- v3c ACCEPTANCE isolation.

No new environment interaction, acquisition, continuation labeling, policy
training, physics change, threshold relaxation, or role movement is part of
this architecture comparison.

## Data used

Upstream TRAIN remains the original v3 bank:

- 821 candidates
- 785 positive
- 36 negative
- 9 parent groups

Upstream CALIBRATION remains the repaired v3b bank:

- 739 candidates
- 733 positive
- 6 negative
- 3 parent groups

The comparison therefore answers the engineering question: on the exact same
available evidence, does a standard 64x64 MLP rank continuation outcomes better
than the historical 8-unit tiny MLP?

## Interpretation boundary

Because the v3b calibration outcomes were already inspected before this
architecture revision, this run is explicitly a same-data post-failure model
comparison. It must not be described as a fresh independent architecture
selection experiment.

The repository records this provenance in the field/calibration/summary
artifacts. The user has nevertheless chosen this same-data protocol for the
current engineering C1 train-and-test decision. If the unchanged C1 calibration
gate passes, the engineering pipeline may continue under that declared scope;
no publication-level claim of fresh architecture-selection independence is
implied.

TEST/final JCE/JEL remains untouched.
