# JIT frontier v2 downstream all-positive result — 2026-09-03

## Status

The `local_horizon_v2` TRAIN attempt is a **failed frontier-identification round** and must be preserved unchanged.
It resolved the v1 acquisition-support failure, but it did not provide two-class downstream continuation support for fitting `C_down^1`.

Observed TRAIN evidence from the operator log:

```text
acquired candidates: 932
upstream candidates: 821
downstream candidates: 111

downstream continuation labels:
  positive = 111
  negative = 0
  parent groups = 3
```

Therefore `C_down^1`, Tube_2, and pi_2 are **not authorized** from this v2 round.
Do not weaken the fixed TRAIN requirement (`>=20` positive, `>=20` negative, `>=3` parent groups) after seeing this outcome.

## Interpretation boundary

What is established:

- the v2 1/2/4/8-tick local acquisition grid fixed the earlier zero-candidate downstream problem;
- the selected frozen pi_1 successfully continued from every acquired downstream TRAIN candidate in this round;
- the current downstream v2 sample therefore does not contain an empirical pi_1 continuation failure.

What is **not yet established** from the aggregate log alone:

- whether the strongest/longest downstream cell (`strength=0.10`, `duration=8`) was also all-positive;
- whether failures are absent uniformly across action dimensions/signs or only missing in particular directions;
- whether one-axis perturbation magnitude, perturbation dimensionality, or parent-state location is the limiting factor.

Do not choose a v3 panel until the completed v2 acquisition and label artifacts are stratified by probe metadata.

## Required read-only diagnostic

Use:

```bash
python JIT/cli/analyze_frontier_support.py \
  --role-root JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_localhorizon_v2/frontier_train \
  --output JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_localhorizon_v2/frontier_train/support_diagnostics.json
```

This command performs zero new environment interactions and zero training transitions. It joins completed candidates and labels by exact `candidate_id` and reports, separately for upstream/downstream:

- total positive/negative/group support;
- outcome classes;
- positive/negative support by `(strength, duration)`;
- support by action direction/sign;
- support by parent group;
- the strongest/longest predeclared cell;
- whether that strongest/longest cell is itself all-positive.

The v2 artifacts remain immutable and this diagnostic cannot retroactively repair v2.

## Decision logic after the diagnostic

A new v3 round is allowed only after its complete probe rule is written before any v3 outcome is observed.

If the downstream strongest/longest v2 cell is all-positive across useful parent/direction coverage, the evidence supports opening a new **boundary-bracketing** acquisition method that reaches farther from the parent state (for example larger magnitude and/or predeclared sparse multi-axis perturbations), while preserving real-dynamics-only state generation and the fixed continuation-label definition.

If negatives already appear in particular strong cells/directions but aggregate downstream support remained all-positive because those cells produced no accepted candidates, investigate acquisition exclusion mechanisms and directional support before increasing perturbation globally.

The next round must not automatically:

- lower continuation-label support requirements;
- use v2 TRAIN rows as CALIBRATION or ACCEPTANCE;
- move parent groups between logical roles after outcomes;
- fall back to full Tube_1 parent sampling;
- reuse bootstrap expert-conditioned `V_down` as pi_1 continuation authority;
- change the 400-tick continuation success definition;
- introduce TEST/final evidence;
- alter physics/domain randomization as part of this same frontier repair.

The research question remains nominal-dynamics pi_1-conditioned state-space frontier identification. Domain randomization is a separate robustness extension.
