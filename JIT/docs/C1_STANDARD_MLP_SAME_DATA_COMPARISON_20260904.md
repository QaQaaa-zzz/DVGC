# C1 standard-MLP same-data architecture comparison — 2026-09-04

## Final engineering decision

The Iteration-1 upstream continuation field is now fixed for the engineering
mainline as:

- profile: `standard_mlp_64x64_tanh`
- architecture: `76 -> 64 tanh -> 64 tanh -> 1`
- parameters: 9,153
- exact field SHA-256: `94528aed8bfb4e6db5c01a2bd4231297a0cd3252f198a889c929bca4ee8aac07`
- exact field manifest SHA-256: `f010f1cafd17dc7e981f1c0c0d62f55dbeda75ea8d65185568041ac8f199955d`
- exact calibration SHA-256: `3b1de2557eba250fdaa1df6e4b6f05082e2f808e9979a3c10ce843d53469cdf3`

This selection is an explicit user-authorized engineering override.  The
original formal calibration result is preserved and is **not** rewritten as a
PASS: ROC AUC is `0.6903137789904502`, below the previously fixed `0.70`
criterion.

The rationale is sample-limited calibration support plus the controlled
same-data capacity study: only six upstream calibration negatives are present;
the 64x64 model materially improves every useful non-AUC diagnostic; and a
larger 128x128 network sharply degrades, so continued width escalation is not
justified.

This is an engineering-mainline decision, not a publication-level claim that
the original formal AUC gate passed.  TEST/JCE/JEL remains untouched.

## Same-data model comparison

All three models used the same upstream TRAIN and repaired-v3b CALIBRATION
roles, labels, weighting, optimizer, learning rate, number of optimizer steps,
L2 penalty, phase seed rule, threshold rule, and calibration gates.

### Historical tiny MLP

- profile: `legacy_tiny_tanh`
- architecture: `76 -> 8 tanh -> 1`
- parameters: 625
- ROC AUC: `0.6634834015461574`
- positive recall: `0.23465211459754434`
- accepted calibration negatives: 0
- accepted-positive support in every parent: FAIL

### 64x64 MLP — selected

- profile: `standard_mlp_64x64_tanh`
- architecture: `76 -> 64 tanh -> 64 tanh -> 1`
- parameters: 9,153
- ROC AUC: `0.6903137789904502`
- positive recall: `0.5934515688949522`
- threshold: `0.9835533512239714`
- accepted calibration negatives: 0
- accepted-positive support in every parent: PASS
- parent accepted-positive counts:
  - `transition_4988928__1000001`: 1 / 164 positives
  - `transition_7987200__1000002`: 211 / 274 positives
  - `transition_9977856__1000003`: 223 / 295 positives
- formal calibration result: FAIL only because ROC AUC remains below 0.70

### 128x128 MLP — rejected

- profile: `standard_mlp_128x128_tanh`
- architecture: `76 -> 128 tanh -> 128 tanh -> 1`
- parameters: 26,497
- ROC AUC: `0.5295588904047295`
- positive recall: `0.05593451568894952`
- threshold: `0.9999508335045384`
- accepted calibration negatives: 0
- positive mean score: `0.8065869471586649`
- negative mean score: `0.8508407340215104`
- score gap: `-0.044253786862845534`
- accepted-positive support in `transition_4988928__1000001`: 0 / 164
- calibration result: FAIL

The 128x128 result shows that simply increasing width is no longer a sensible
repair path on this evidence.  It produces strong score saturation and worse
ranking/generalization than the 64x64 model.

## Fixed data used

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

No new environment interaction, acquisition, continuation labeling, policy
training, physics change, threshold relaxation, or role movement was introduced
for the architecture comparison.

## Engineering continuation rule

For the engineering mainline only:

1. preserve the exact 64x64 upstream field and its original
   `calibration_passed=false` artifact;
2. do not lower the AUC criterion or the max-negative threshold;
3. write a separate self-hashed engineering-override artifact selecting that
   exact field;
4. use the same 64x64 architecture for `C_down^1`;
5. require downstream calibration to pass normally — the upstream override does
   not automatically waive downstream evidence;
6. only after downstream passes may the explicit engineering Tube2 builder run;
7. Tube2 must record that its upstream continuation selection used an
   engineering override and that the formal all-phase calibration gate did not
   pass;
8. keep v3c ACCEPTANCE locked and TEST/JCE/JEL untouched.

## Claim boundary

Authorized:

- engineering `C_up^1` selection = exact 64x64 field above;
- proceed to fit/calibrate `C_down^1` with the same architecture;
- if downstream passes, construct engineering Tube2 and continue toward pi2.

Not authorized:

- claim that upstream ROC AUC >= 0.70;
- claim fresh independent architecture selection;
- claim a certified safe set, viability kernel, reachability set, or certified
  continuation probability;
- touch TEST/JCE/JEL before the designated final evaluation stage.
