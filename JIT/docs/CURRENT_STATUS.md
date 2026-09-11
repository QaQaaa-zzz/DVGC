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


## Parallel continuation optimization — 2026-09-11

Implemented an opt-in `vectorized` backend using genuine Warp multi-world stepping. Shared contact buffers remain unbatched and scale with actual batch size; all physics substeps accumulate contact/CCD/constraint saturation flags. Any saturation/nonfinite/switching aborts label publication. Candidate keys, complete snapshot validation and first-landing semantics remain unchanged. Inactive simulator slots are charged. Snapshot restoration is no longer duplicated for device batches, compilation is cached per batch shape within a worker, and timing separates restore/compile/execute/full process.

Existing CLI `JIT/cli/label_policy_family_first_landing.py` accepts `--execution-backend vectorized --batch-size N` in shard mode. Set shard count according to the existing catalog; do not duplicate rows for scientific evaluation. `JIT/cli/benchmark_continuation.py` provides bounded serial comparisons and `--capacity --batch-sizes 4096 8192 16384` measurements on replicated existing TRAIN snapshots. Its explicit catalog/acquisition/evaluator/seed/budget arguments bind each run; timeout and memory guard preserve charged attempts.

4096/8192/16384-world initial capacity tests completed. Final checked16384 result:19.495s complete worker wall,10.593s initialization,2.907s compilation,1.908s execution,14,522MiB observed device memory,157,227 useful simulator steps/s.24576 reached22,542MiB and was stopped by the22,000MiB protection threshold before a valid result;no claim of hardware OOM. This short16-source replicated panel is engineering capacity evidence. It does not establish long-run memory stability or full-campaign speedup.

The16-candidate comparison preserves endpoint labels but has one38→39 control-step difference in the final checked path. Automatic scientific backend promotion remains disabled;old campaign serial requests remain unchanged. The final derived comparison verifies a relative/absolute catalog path alias before comparing protocol-identical content;raw records are untouched. Total task charge3,384,811 includes inactive work, interrupted attempts and the memory-stopped reservation. Zero PPO,new acquisition or final TEST.

The four historical landing/failure receipts were preserved in a derived audit. Source review confirms contact detection can coincide with roll/pitch/backward termination;training masks physical failure while acquisition/continuation prioritizes first contact. Historical substep ordering remains unresolved;no labels were rewritten.

Full evidence/plots/CSV/cost index:`JIT/runs/engineering/continuation_parallel_20260911/INDEX.md`. Compact tracked summary:`JIT/review_evidence/parallel_capacity_20260911.json`. Independent review findings on resource saturation and failed-attempt accounting were fixed and regression-tested.

Validation:76 related CPU behavior/regression tests passed in the production interpreter. Real GPU checks are the bounded runs above;no full campaign was rerun.

## Versioned witnesses and training prerequisites — 2026-09-11

`probe_bank.py prepare-existence` / `run-existence` now provide opt-in first-success existence evaluation. A plan locks TRAIN catalog, complete bank/order, exact global candidate indices/keys, seed, horizon, sources and budget. Completed subsets have a distinct schema and cannot be merged as full matrices. Untested/error/incomplete outcomes remain explicit unknowns; no-witness requires all evaluators to fail. Contact plus physical-failure evidence is quarantined under a named new aggregation rule, leaving historical labels unchanged.

Resume audits and charges every prior reservation before deciding retries, reuses valid later witnesses, and retains changed subsets in separate chunks. Child timeouts/interruption preserve worst-case charges. Source drift after a worker stops further scheduling. Cache validation checks requested policy/context/catalog/endpoint identity, subset execution schedule, label hash and useful/padded costs.

Existing pi6 TRAIN analysis verified all seven completed evaluator outputs against their original contracts.738 candidates: full bank5166 calls/98,093 useful steps; fixed pi0→pi6 offline first-success844 calls/15,388 useful steps.84.31% is hypothetical useful-step reduction, not measured wall-clock speedup.64 evaluator-level contact/failure conflicts yield735 clean witnessed states and3 unknowns under the new rule, exactly matching full-bank aggregation under that rule. This differs from legacy738 positives and is separate from the four forward receipt conflicts. Historical event substep ordering remains unresolved.

Multi-checkpoint TRAIN panels are opt-in, aligned to3200-step PPO blocks (e.g.32k/64k/128k), end at the declared maximum, and use the same live trainer/critic/optimizer. Exact fixed panel entries, horizon, RNG and reset context are hash-bound. Panel reservations and completed receipts are identity-checked and included in campaign costs. Legacy default requests remain compatible. No optimizer restart, automatic extension, or matched-budget exploration per intermediate checkpoint is implemented. No new PPO run occurred in this task.

`residual_exploration.py` and `fit_residual_explorer.py` implement a zero-initialized64→32→4 conditional supervised warm start, with explicit observations/history/base action/goal, normalization, action and internal residual slew bounds, and hash-bound artifacts. Supervised loss learns requested residuals before runtime slew clipping, avoiding a zero-gradient dead zone. Synthetic CPU fixtures validate logic only. Production action-aligned observation export, explorer context in causal snapshots, RL discovery and equal-budget comparisons remain pending. File/witness assertions in an export are checked but not independently certified by this fitter. Diffusion remains unimplemented.

Validation:126 related CPU tests passed in19.57s using the production interpreter, including retry/source-drift/unknown handling, subset global keys, panel schedule/accounting, and residual fitting/state bounds. Independent review corrected resume over-budget risk, cache identity gaps, mismatched panel receipts and residual loss gradients. Full analysis, figures, CSV, scripts, run declaration and costs: `JIT/runs/engineering/existence_checkpoint_residual_20260911/INDEX.md`.

Bounded serial GPU validation completed on16 deliberately selected existing TRAIN states.40 evaluator-candidate calls across pi0→pi3,1,633 simulator interactions,104.596s scheduler wall time, no retries; all16 existence outcomes match the verified full-bank view under the new quarantine rule. Remaining evaluators are explicitly untested. No paired wall-time baseline was run. Original unused pre-review plan remains archived with zero charge. `gpu_comparison.csv` and per-process receipts preserve complete evidence.

## Authorized local multi-checkpoint PPO — completed 2026-09-11

User selected multi-checkpoint PPO after implementation. Separate run `multicheckpoint_pi6_20260911` starts from pi6 Actor/normalizer with fresh critic/optimizer, reuses completed round_001 TRAIN support,128 environments,seed9871101, and preserves20% fixed start/80% complete snapshot resets. Declared128k PPO + maximum4,800 panel steps, one attempt,30-minute limit. Config committed as3e20ed2; no all_proposers_v1 rerun or bank promotion.

Observed:128,000 PPO +225 evaluation =128,225 interactions,58.835s full process wall,exit0. Checkpoints0/32k/64k/128k saved and all8 identity/payload files verified. Fixed4-state TRAIN panel success4/4 at each evaluated checkpoint; panel costs76/76/73,zero physical failures. No final TEST, new acquisition or retries. Small correlated development panel cannot establish convergence or improved exploration.30 related CPU tests passed before launch; no new implementation changes were required.

Full logs, metrics, all panel trajectories, PNG/PDF/SVG, CSV, manifests and cost:`JIT/runs/experiments/multicheckpoint_pi6_20260911/INDEX.md`. Compact tracked record:`JIT/review_evidence/multicheckpoint_train_20260911.json`. Next capability work remains intermediate-checkpoint exploration identity/equal-budget evaluation and actual residual data/export integration; do not automatically extend PPO.

## Diagnostic checkpoint exploration comparison — 2026-09-11

Implemented explicit `jit_frozen_development_checkpoint_v1` identities for declared checkpoints from a completed training run. Original config/report/sidecar/payload/model identities remain locked; formal final-only freezing is unchanged. Diagnostic names cannot impersonate pi_N and causal acquisition permits them only in explicit TRAIN probe-bank mode. No formal envelope/promotion authority is granted.

`compare_checkpoints.py` now predeclares and runs a bounded common-schedule comparison, verifies all completed subset receipts and unknown aggregation, preserves full failure reservations and terminates subprocess groups on timeout. Generic offline analysis produces complete phase-separated geometry, cost curves, all-row CSV and matched-budget tables.100 related CPU tests passed in4.59s before source locking; independent review corrected timeout and receipt-accounting issues.

The user-approved local comparison uses originalpi6 and32k/64k/128k as proposers with common pi0–pi6 evaluators;8 identical four-channel signed perturbation trajectories each. Full actual schedules and conservative catalog-order equal-cost replay are distinct; incremental training costs are added separately. The9,296-cell historical campaign baseline conservatively excludes old witnessed cells without retroactively changing old label semantics. Single TRAIN seed and correlated frames remain limitations. Full evidence:`JIT/runs/experiments/checkpoint_discovery_20260911/INDEX.md`. Implementation/config commit5a8cc03.

Actual comparison completed:809 candidates across32 trajectories,778 witnessed,25 no-bank-witness,6 unknown.20,733 interactions,1127.341s,no new PPO or engineering retries. Full-schedule novel cells pi6/32k/64k/128k=146/186/182/205 at4,225/4,058/6,502/5,948 exploration steps. Common exploration budget4,058 gives132/186/132/165 novel cells in conservative catalog-order replay. Training-inclusive common budget4,225 cannot cover any new checkpoint's training surcharge; no amortized benefit shown.

Clean forward landing observations8/8,7/8,6/8,7/8. This pilot favors32k for incremental novelty per small exploration budget, but favors originalpi6 for clean forward landings and lowest clean trajectory peak. Higher witnessed z values exist in trained checkpoints; most geometry overlaps, and failed low trajectories are not a lower boundary. Six conflicts remain unknown. Recommend no further PPO by default; prioritize pi6/32k complementary evaluation and independent repeats. No bank promotion or final TEST was performed. Full tables/phase5cm CSV/geometry/cost PNG-PDF-SVG/source copies and all receipts are indexed in the local report; compact record:`JIT/review_evidence/checkpoint_discovery_20260911.json`.
