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

256k PPO completed with392 validated panel interactions (256,392 total),84.061s,exit0; all five4-state TRAIN panels4/4,zero physical failures. These panels do not establish convergence or exploration improvement. Late pi6/128k/192k/256k matched exploration is running under a separate6M ceiling.63 relevant CPU tests passed in5.75s. Parallel zero-interaction residual data audit found809 saved states,777 potential observation/next-action pairs and90 nonzero pairs with later witnesses; sparse saved-state inputs reconstruct from FIFO/events, but explicit action-time/downstream-witness credit and trajectory-group split are required before fitting. Audit:`JIT/runs/engineering/residual_data_readiness_20260911/INDEX.md`.

## 2026-09-11: length pilot closed; causal residual export ready

256k late comparison completed: pi6/128k/192k/256k common-budget novelty134/122/103/161 at4,135 exploration interactions; full novelty134/177/176/237. Full new cost281,308 (256,392 training/panels +24,916 exploration). Provisional128k–256k range with early snapshots, no more length sweeps. Report publication recovered after HTTP408: https://github.com/QaQaaa-zzz/DVGC/tree/agent/jit-run-reports/reports/multicheckpoint_pi6_256k_20260911-71a144003c .

`residual_dataset.py` and `export_residual_dataset.py` add v2 causal sparse exports, explicit pre-action origin and strictly later witnessed target, complete prefix/index/cost checks, historical observation source hashes and grouped fit/development partitions. Source is the completed earlier checkpoint_discovery comparison; revalidation is offline evidence inspection, not old-plan resume authority. Validated export543 fit/203 development rows,66/24 nonzero residuals;32 terminal saved states and31 rows without later novel witnesses excluded. Six/two trajectory conditions are disjoint across policy arms but share ancestors, so they are not independent runs or final TEST. Original invalid export (missing contract fields) is retained and never fitted.75 related CPU tests passed in6.01s; independent review's observation-semantic drift issue fixed. One predeclared500-update CPU supervised fit follows; actual learned exploration remains pending.

Residual warm-start completed:500 supervised updates,543 fit rows,zero interactions/PPO,9.085s process wall. Fit RMSE0.02473→0.00100 but development RMSE0.02381→0.13010 (203 rows). No promotion: declared split holds both steering directions out of fit; constant direction feature/std floor maps development values to±1000. Added independently reviewed opt-in actor_fit_std_goal_units_v1, retaining legacy contracts; no refit or gain claimed. Next dataset must cover all4channels in both partitions under different whole trajectory conditions, with complete pre-action recording and fixed-unit goal scaling.76 related CPU tests pass; all results, failed first export, pictures/predictions and costs retained at `JIT/runs/experiments/residual_warmstart_20260911/INDEX.md`.

## Per-policy privileged coverage PPO redesign

User replaced supervised residual imitation with106-input256×3 full-action PPO explorer, novelty relative to frozen currentpi only. First bounded TRAIN pilot completed; initial32/32→final0/32 clean landings, no promotion.1,576 cumulative new cells includes1,403 from the pre-update stochastic actor. Excessive first-update displacement observed. Cost291,200 including failed preflight reservation; pilot15,917 valid training steps versus256,000 scheduled training slots. Full evidence in `../runs/experiments/per_policy_coverage_ppo_20260911/INDEX.md`; source0917c65. Existing historical warm-start results remain valid for their original scope.

## Corrected residual explorer (latest)

User clarified explorer perturbs frozenpi, not replaces it. Canonical trainer now requires jit_frozen_policy_residual_ppo_v1, composes frozenpi+bounded4delta, and pays own-pi novelty at candidate action ticks only after same-context frozen-bank suffix success; parent trajectory landing is not required.184 CPU tests and real one-update GPU chain passed,2,564 interactions including padding/suffix. Full-action pilot above is off-target historical engineering evidence. No long corrected training or performance claim yet. See `../runs/engineering/residual_suffix_ppo_20260911/INDEX.md`.

## Delayed exploration loop implementation — latest

User-approved implementation now separates provisional per-pi arrival reward from verified envelope membership. Pending real snapshots may train the next jumping policy under a new explicit candidate-support schema; later frozen-policy suffix results append to history and refresh training support. Current-bank no-witness is not infeasibility, unknown remains unknown, and old PPO trajectories are not replayed. Original witnessed-only support/evaluation paths remain available. Critic values are recorded without reward penalties.

A finite CPU orchestrator couples matched-forward learned/random residual exploration, current-bank labeling, candidate-reset warm-start PPO, diagnostic checkpoint freeze and new-member delayed evaluation for at most two rounds. Original witnessed fixed TRAIN panels remain separate. The random arm shares the evolving pi trained on learned-arm pending data, so this is a conditional exploration comparison. Per-round full evidence, replot data, pictures and actual/reserved costs are retained.

Execution waits for the STTW priority_survival pipeline's terminal `complete`/`completed` status AND no matching live Python process under the declared project owner. Every GPU child rechecks the gate. No STTW files/processes are modified. Source drift, failed/unfinished engineering, absent pending candidates or any failed child stops progression. Queue templates declare352,000 engineering +3,664,000 pilot maximum interactions, one attempt per stage,128k per new policy only after the short complete chain. This section describes implemented code, not a successful GPU run or capability improvement.

Verification and actual launch state:230 relevant CPU tests passed in7.51s; no new GPU simulation/training was used for this implementation. Source/input/budget-locked queue successfully prepared, then CPU-only supervisor PID1907686 started. Its actual status is `waiting`, reserved_interactions0, stages[]; it detects STTW pipeline1761653/trainer1761715 still running. Live record: `../runs/experiments/delayed_exploration_20260911/queue/execution/status.json`; complete entry: `../runs/experiments/delayed_exploration_20260911/INDEX.md`. This is a waiting launch, not completed engineering or a training result. Compact validation record: `../review_evidence/delayed_exploration_20260911.json`.

## 2026-09-12 recovery and pilot

Previous queue failed before new-policy PPO because canonical preflight omitted jit_iterative_candidate_training_v1. Fixed and verified with232CPU tests plus the original1040-row config. Real recovery reused saved support:3200PPO+77panel+25suffix=3302interactions, fixedpanel4/4, but no pending candidate recovery (learned0/2,random0/1). Complete evidence:`../runs/engineering/delayed_recovery_20260912/INDEX.md`. Old failed records unchanged. The new comparison locks one common per-pi baseline artifact for both arms.

Fresh two-round128k pilot launched under3,664,000maximum, background supervisor3959291. Current status:`../runs/experiments/delayed_exploration_20260912/queue/execution/status.json`; do not infer completion from launch. Full iteration output will be under queue/stage_0_result. Compact recovery evidence:`../review_evidence/delayed_recovery_20260912.json`.

Latest queue path supersedes the preceding launch: `../runs/experiments/delayed_exploration_reuse_20260912/queue/execution/status.json`. The prior pilot completed19,200forward slots but was blocked before labeling by an STTW analysis process. Inner stages now wait finitely; first learned arrivals are explicitly reused with locked artifacts and matching sampling contract.233CPU tests passed. All stopped outputs remain immutable; inherited interactions are separate from newly dispatched cost.

## Latest user instruction: JIT paused

Let STTW finish first; JIT requires a fresh explicit user start. Cancelled JIT wait runner487670 with zero child dispatch. No automatic restart. First-round128k and second-round exploration/labels are complete; second-round PPO has not started. Its prior timeout was a dependency-watcher deadlock, not a numerical failure. The opt-in passive-watcher gate fix passed43CPU tests; real STTW training workers remain blocking. Prepared continuation:`../runs/experiments/delayed_completion_20260912/`, currently cancelled by user instruction. Original failed pilot statuses remain historical evidence.

Latest user instruction supersedes pause: explicitly restart remaining second-round128k training. New background supervisor914277, root `../runs/experiments/delayed_completion_start_20260912/`; follow pipeline_status.json and training_execution/status.json there. Existing second-round config passed offline preflight and inputs remain locked. STTW fixed-roll was still evaluating at launch, so the runner waits for completion and worker exit before GPU dispatch, then trains and evaluates the21saved pending candidates. Total remaining ceiling138000; no reacquisition or automatic retry. Old pause record is historical, unchanged.
