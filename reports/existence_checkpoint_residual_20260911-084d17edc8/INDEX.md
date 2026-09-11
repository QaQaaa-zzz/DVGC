# First-success / checkpoint / residual engineering evidence

Scope: implement the approved DVGC continuation roadmap without rerunning all_proposers_v1, training pi7, or using final TEST.

## Existing-data analysis

- `analyze.py`: reproducible source-validated offline scheduling analysis; invocation takes original discovery directory and this output directory.
- `offline_summary.json`: complete input hashes and totals. Existing pi6 TRAIN panel:738 candidates,5166 full-bank calls,98093 useful steps; fixed pi0→pi6 first-success replay:844 calls,15388 useful steps.84.31% is an offline useful-step reduction, excluding process time, padding and retries.
- `candidate_schedule.csv`, `offline_witness_index.json`: all738 candidates and explicit per-evaluator states, including untested.
- `offline_cost.png`, `.pdf`, `.svg`: replot-ready cost comparison; PNG visually inspected for labels and limits.
-64 individual evaluator labels contain contact/failure conflicts. New quarantine aggregation yields735 witnessed,3 unknown, matching full-bank aggregation under the same new rule. Historical738-positive union remains untouched. This is distinct from the four forward-trajectory receipt conflicts.

## Bounded real-run declaration

`engineering_indices.json` selects16 existing TRAIN states: first8 plus first8 where pi0 was not a clean success. This is deliberately an engineering regression panel, not independent held-out evidence.

Declared evaluator order pi0→pi6, original deterministic seed9852501, horizon400, serial backend, batch1, max16 candidates/process,44,800 total simulator-step budget and600s per worker. No acquisition/PPO/final TEST. CPU analysis costs zero simulator interactions. Failed/interrupted attempts reserve their maximum. Original `plan.json` was superseded before launch by a source-review correction; zero interactions under that plan. Actual run will bind `reviewed_plan.json`.

## Implemented boundaries

Multi-checkpoint mode saves and evaluates aligned milestones inside the same live trainer, with a fixed hash-bound TRAIN panel and reserved costs. No optimizer restart, adaptive extension, or equal-budget intermediate-checkpoint exploration is implemented. No PPO was rerun in this task.

Residual module is a zero-initialized64→32→4 supervised warm-start component with explicit observation/history/base-action/goal normalization, action amplitude/slew constraints, frozen identities and source hashes. Tests use synthetic, clearly declared examples. Production observation/action training export and explorer state in causal snapshots are not yet available; no real learned exploration, independent witness certification or matched-budget superiority claim.

## Verified execution and checks

The reviewed16-state serial GPU run completed with40 evaluator calls across pi0/pi1/pi2/pi3; pi4/pi5/pi6 were untested because every candidate already had a clean witness. Actual charge1,633 interactions,104.596s complete scheduler wall time, zero retries. All16 existence outcomes agree with the verified complete bank under the new quarantine rule. `gpu/` retains full per-worker commands, reservations, results, logs, start/end times and cost ledger; `gpu_comparison.csv` retains every comparison. No paired full-bank wall-time run was made.

`compare.py` regenerates the engineering summary. `tests.log`:126 CPU tests passed in19.57s; source and CLIs also passed py_compile and git diff --check. Independent reviewers fixed failed-resume budgeting, subset identity drift, source-change scheduling, checkpoint receipt joins and clipped-loss gradients before the GPU plan was locked.
