# Local multi-checkpoint PPO — completed 2026-09-11

User authorized this separate development training run. From frozen pi6 Actor/normalizer, fresh critic/optimizer, immutable round_001 TRAIN support;20% fixed jump start /80% witnessed snapshot reset.128 environments; seed9871101. No all_proposers_v1 rerun, new acquisition, automatic extension, strategy-bank promotion or final TEST.

## Result

| Checkpoint PPO steps | Same fixed TRAIN panel | Physical failures | Evaluation interactions |
| ---: | ---: | ---: | ---: |
| 32,000 | 4/4 | 0 | 76 |
| 64,000 | 4/4 | 0 | 76 |
| 128,000 | 4/4 | 0 | 73 |

128,000 actual PPO transitions +225 panel interactions =128,225 total, below132,800 predeclared ceiling. One attempt, exit0,58.835s full child/process supervision wall time including startup, compilation, training and panels. This is not Brax's narrower training/walltime. No retries or automatic extension.

This validates multi-checkpoint training execution. Four correlated TRAIN support states cannot establish convergence, improvement over pi6, or wider capability coverage; no per-checkpoint exploration or final holdout was run. These inference checkpoints do not restore optimizer state for resumed training.

## Complete evidence

- `reservation.json`, `command.json`, `launch.py`: budget, locked inputs/sources, exact command and bounded one-attempt supervisor.
- Tracked config:`/home/qy/DVGC/JIT/configs/pi6_multicheckpoint_train_20260911.json`; config commit3e20ed2, implementationf22039b.
- `process.log`, `exit.json`, `status.json`, `summary.json`: full process log, UTC start/end, outcome and costs.
- `training/pi6_multicheckpoint_train_20260911/formal_report.json`: trainer completion, schedule and accounting.
- `training/pi6_multicheckpoint_train_20260911/checkpoints/transition_{0,32000,64000,128000}/`: identity.json and payload.pkl; all8 files verified and hashed in `checkpoint_manifest.json`. Models remain local and are not committed to Git or included in review bundles.
- `training/pi6_multicheckpoint_train_20260911/train_panels/`: all three panel reports, reservation/completion receipts, trajectories and xz PNGs.
- `checkpoint_panels.csv`, `all_panel_records.csv`, `training_metrics.csv`: all panel results and training metric records.
- `checkpoint_training.png`, `.pdf`, `.svg`: replot data above; PNG visually inspected, labels/limits readable. `analyze.py` reproduces this analysis without simulation.
- `analysis_summary.json`: derived full outcome and panel identities.

Before launch:30 relevant CPU behavior tests passed in3.07s; configuration loading and Git diff checks passed. Actual GPU completion above is separate evidence.
