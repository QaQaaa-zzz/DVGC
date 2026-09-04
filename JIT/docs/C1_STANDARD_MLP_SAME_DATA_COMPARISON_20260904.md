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
- observed upstream calibration ROC AUC: `0.6634834015461574`
- observed upstream positive recall: `0.23465211459754434`
- accepted calibration negatives: 0
- accepted-positive support in every parent: FAIL

First comparison profile:

- profile: `standard_mlp_64x64_tanh`
- architecture: `76 -> 64 tanh -> 64 tanh -> 1`
- parameters: 9,153
- observed upstream calibration ROC AUC: `0.6903137789904502`
- observed upstream positive recall: `0.5934515688949522`
- accepted calibration negatives: 0
- accepted-positive support in every parent: PASS
- calibration result: FAIL only because ROC AUC remains below 0.70

Final capacity-escalation profile:

- profile: `standard_mlp_128x128_tanh`
- architecture: `76 -> 128 tanh -> 128 tanh -> 1`
- parameters: 26,497
- status: authorized final same-data capacity comparison; outcome not yet observed

The 128x128 run is the stopping point for architecture-size escalation on this
same evidence. If it still fails the unchanged calibration gate, do not continue
increasing network width as the automatic next action.

## Fixed comparison contract

All comparison profiles keep unchanged:

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
available evidence, how much of the continuation-ranking failure is attributable
to the historical tiny network capacity?

## 64x64 interpretation

The 64x64 comparison produced a material improvement without changing the data:

- ROC AUC increased from `0.66348` to `0.69031`;
- positive recall increased from `0.23465` to `0.59345`;
- zero accepted negatives remained satisfied;
- the previously failing parent-local accepted-positive condition became PASS.

Therefore model capacity is demonstrably one contributor to the original C1
failure. However the fixed AUC >= 0.70 requirement remains unmet, so the 64x64
field is not authorized for Tube2 construction.

## Interpretation boundary

Because the v3b calibration outcomes were already inspected before these
architecture revisions, these runs are explicitly same-data post-failure model
comparisons. They must not be described as fresh independent architecture
selection experiments.

The repository records this provenance in the field/calibration/summary
artifacts. The user has chosen this same-data protocol for the current
engineering C1 train-and-test decision. If the unchanged C1 calibration gate
passes, the engineering pipeline may continue under that declared scope; no
publication-level claim of fresh architecture-selection independence is
implied.

TEST/final JCE/JEL remains untouched.
