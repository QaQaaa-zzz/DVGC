# Causal residual warm-start pilot — 2026-09-11

Outcome: data export and a bounded supervised fit completed. The learned artifact is NOT approved for exploration: development-condition imitation is worse than zero residual. No simulation, PPO, final TEST, replay or automatic refit was performed.

## Evidence and cost

Source comparison: checkpoint_discovery_20260911 (original pi6 and32k/64k/128k). From809 saved candidates,32 lack a stored next action and31 lack a later novel clean witness. The746 remaining sparse action pairs contain90 nonzero residual targets. Every pair binds pre-action snapshot/tick/Actor observation to the next recorded action and a strictly later same-trajectory witnessed root cell, with exact nested prefixes and revalidated bank/subset receipts. Saved Actor observation semantics match historical observation.py/constants.py hashes.

Fit:543 examples,66 nonzero,512 target cells; development:203 examples,24 nonzero. Six/two perturbation conditions are held together across all policies; ancestors are shared. This is TRAIN development, not independent repetitions or final holdout. Fixed action-schedule goals are exogenous, not outcome features or learned novelty goals.

One CPU fit,500 optimizer updates,seed9901101,learning_rate0.001. Initial requested-residual loss0.0006114898 →0.000001379094. Fit stage6.082s; full process9.085s. New environment interactions0; new PPO transitions0. Code at fit:e1c88b5. Full source hashes and budget in fit_reservation.json. The initial dataset/ export failed contract validation before fitting; it is preserved and NOT used. validated_dataset/ is the actual input.

## Development failure and decision

Teacher-forced effective-action RMSE, weighted by witnessed target cell:

|Partition|Zero residual|Fitted residual|
|---|---:|---:|
|Fit, all543|0.0247283|0.00100172|
|Development, all203|0.0238125|0.130104|
|Fit, nonzero66|0.0747714|0.00245431|
|Development, nonzero24|0.0750000|0.149841|

The predeclared split holds both steering signs out of fitting. direction_steer is zero in every fit row, so empirical std flooring at0.001 maps development±1 to about±1000. direction_coverage.csv and feature_shift.csv show the complete diagnosis. This explains an input-distribution mismatch; a controlled ablation would be required to separate normalization from missing-channel effects. Neither action imitation nor its error is a measured discovery/recovery result.

A new opt-in actor_fit_std_goal_units_v1 normalization mode keeps observed Actor normalization fit-only and represents known base-action/goal units with mean0,scale1. It is behavior-tested and independently reviewed, but was NOT applied to this historical fit or refitted on this result. Existing default contracts remain unchanged. It removes direction amplification; it cannot supply missing steering examples.

Next: a separately declared fully recorded multi-condition dataset covering all four action channels in each fit/development partition, grouping whole condition trajectories and preserving ancestors; use fixed-unit goal scaling. Keep the old failure intact. Then bounded fitting and frozen-explorer versus fixed-perturbation real acquisition with shared-bank witnesses and full total costs. No additional PPO-length sweeps; no diffusion, automatic promotion or current residual deployment.

## Complete artifacts

- validated_dataset/: both full datasets, records, source hashes, partitions and exclusions.
- explorer.json: immutable learned parameters and full fitting provenance; server-side.
- all_predictions.csv: all746 predictions, targets, origins, target identities and weights.
- imitation_metrics.csv and imitation_quality.png/pdf/svg: all before/after fit/development views.
- direction_coverage.csv and feature_shift.csv: every condition and feature, not selected failures only.
- train.py/analyze.py, training.log/analysis.log, fit_status.json, fit_reservation.json, summary.json.
- results_full.zip: local detailed review bundle; referenced original snapshots/checkpoints remain at source paths, not a standalone replay archive.

Verification:76 related CPU tests passed in5.97s; independent code review found no remaining blocker. Exact model reload and both dataset partitions were revalidated after the opt-in normalization change. No GPU exploration claim.
