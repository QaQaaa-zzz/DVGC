# JIT current status — 2026-09-11

## Latest verified production result

`all_proposers_v1`: **round_limit_reached**, two rounds completed, pi_5/pi_6 frozen, each128,000 PPO transitions. No new failed attempts reported. Final TEST unused; physical boundary unproven.

| Stage | New candidates | New root cells | Cumulative cells | Cumulative new interactions |
| --- | ---: | ---: | ---: | ---: |
| Inherited knee/lower/paired support | — | — | 1,689 | 0 |
| Round0 including pi_5 | 3,147 | 2,940 | 4,629 | 483,320 |
| Round1 including pi_6 | 5,323 | 4,667 | 9,296 | 1,275,465 |

Interactions: PPO256,000; final training panels142; acquisition15,289; suffix labels1,004,034. Inherited recorded195,551; combined1,471,016 excludes some earlier bootstrap/seed costs. 11 new acquisition batches,352 trajectories,49,346 evaluator-candidate jobs,413 label subprocesses. No complete wall-clock decomposition yet.

Second-round proposer gains: pi0=596, pi1=732, pi2=629, pi3=860, pi4=715, pi5=406, pi6=729. This is cumulative novelty, not controlled policy ranking. Latest lowest unconflicted successful peak root z≈0.502206m (pi0); not the exact physical lower boundary.

Four forward receipts contain both landing and physical_failure; audit priority remains open. This is separate from the accepted numerical replay differences. Production images/data are saved server-side; text reports/CSV retrieved. Do not assert every image has been visually reviewed.

## Current next action

Do not rerun the completed campaign. Follow [the roadmap](JIT_TRAINING_ROADMAP.md): event audit, existing-data geometry, timing and evaluator early stopping, checkpoint evaluation, lightweight residual exploration. Diffusion remains optional and unimplemented. Current128k does not establish sufficient convergence.

## Evidence

[Production summary](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/summary.json), [proposer metrics](https://github.com/QaQaaa-zzz/DVGC/blob/agent/jit-run-reports/reports/all_proposers_v1-e36bfec926/round_001/figures/proposer_metrics.csv), [full successor handoff](CODEX_HANDOFF_20260911.md), [machine-readable local summary](../review_evidence/all_proposers_result_20260911.json).

Code before this documentation update:eb9584d. Earlier29 CPU tests supported implementation; production report now establishes two-round execution. This documentation turn did not rerun simulation or those tests. Historical guidance is archived in history/handoff_before_20260911, not current instructions.
