# DVGC / JIT — empirical jumping envelope discovery

Updated2026-09-11. The project studies fixed-condition jumping locomotion of a single-track two-wheeled bicycle–pendulum robot. The research product is a growing empirical capability envelope supported by real arrivals and successful landing continuations, not a single policy required to master every collected state.

## Current outcome

The two-round all-policy production campaign completed. pi_5 and pi_6 each trained128,000 transitions and were frozen. Coverage:1,689 inherited →4,629 →9,296 root cells; new candidates8,470; new interactions1,275,465. Stop reason was the declared two-round cap. This is development evidence, not convergence or a global physical boundary.

Old policies still discover substantial new space. The earlier pi_2/pi_4 retrospective control showed pi_4 exploration novelty benefits but not yet a training-inclusive cost advantage at its smaller matched budget. Multi-policy gains cannot all be assigned to newest training.

[Full handoff, results, risks and next tasks](JIT/docs/CODEX_HANDOFF_20260911.md) is the primary successor brief; [current status](JIT/docs/CURRENT_STATUS.md) is the concise record; [roadmap](JIT/docs/JIT_TRAINING_ROADMAP.md) is the implementation sequence.

## Method

Bootstrap up/down support → successful unified pi_0 and captured centerline → frozen-policy bank plus bounded legal action perturbations → real arrivals → successful suffix witnesses → deduplicated empirical Tube → witnessed-support PPO → new frozen policy → repeat within budget.

All four action channels are allowed. Fixed x2.5 near-ground start and accepted replay limits remain. Every candidate keeps provenance and full context. Phase-separated5cm slices support visualization; the existing physical quantization remains unchanged. No artificial state lifting/lowering, no interpolated reachable hull, no whole-Tube Actor gate.

## Implemented versus proposed

| Capability | State |
| --- | --- |
| Growing frozen bank, all-proposer fixed perturbations, two-round PPO loop | Production completed |
| Per-proposer/evaluator figures, replot CSV, phase slices, receipts, cost and automatic text report publication | Implemented; production artifacts reported; not all images visually inspected remotely |
| First-success evaluator stopping | Implemented as an opt-in versioned witness index with explicit unknowns and bounded subset workers; legacy full matrices preserved |
| Multi-checkpoint capability evaluation and adaptive training length | Fixed TRAIN-panel checkpoints implemented; adaptive extension and per-checkpoint exploration remain unimplemented |
| Gain-based proposer budget allocation | Proposed; current equal32-trajectory calls |
| Learned residual exploration network | Bounded supervised warm-start module implemented; actual exploration/data export/campaign integration still pending |
| Conditional diffusion perturbation sequences | Optional later comparison; neither implemented nor established as beneficial |

## Next development

Audit four conflicting forward landing/failure receipts, produce geometry/sensitivity views, measure stage wall time, implement versioned existence-only labeling alongside full comparison panels, then checkpoint evaluation and a lightweight residual exploration baseline. Preserve the completed all_proposers_v1; do not start pi_7 or rerun that command by default.

Residual exploration should generate bounded actions conditioned on the frozen policy/state/goal and be rewarded primarily for new witnessed cumulative support. A diffusion model could generate temporally structured perturbations, but cannot certify reachability or automatically seek boundaries. Its extra cost must be justified against the lightweight baseline.

## Paper scope

Submission target discussed: RAL. Still needed: fixed-bank versus training-iteration total-cost comparison, independent repetitions, phase geometry/resolution evidence, full lifecycle costs, locked final evaluation and bootstrap ablations if their benefit is claimed. Do not invent a completion percentage or promise acceptance after a fixed number of rounds.

TRAIN is adaptive support; CALIBRATION serves optional predictors; used ACCEPTANCE is development. Final TEST/JCE/JEL remains unopened. Tube0 was weighted support (222 rows,42 historical negatives); preserve actual pi_0 lineage and the invalid historical pi_3 mixed-endpoint gate.

## Operations

Working branch agent/two-phase-soft-tube; report branch agent/jit-run-reports in QaQaaa-zzz/DVGC. Full artifacts on /home/qy/DVGC, compact reports on GitHub. Routine project pushes and report publication already authorized. Raw experimental runs immutable; new protocols need new outputs. Follow [AGENTS](AGENTS.md).

## Active optimization — 2026-09-11

User authorized GPU parallelization and bounded engineering measurements. Implement in place on the target branch; preserve unrelated untracked files and completed campaigns. Design: a separate `vectorized` execution backend with genuine multi-world Warp stepping, backend-aware shared contact buffers, device-side endpoint masks, unchanged candidate keys/labels and charged inactive work. Retain serial/device paths. Expose batch/shard execution settings without changing old requests. Measure initialization, batch compilation/execution and process wall time; archive JSON/CSV/plots and failures. Select settings by measured throughput and label equality, not occupied VRAM.

Execution checklist (writing-plans / inline execution):
- [x] Add failing behavior tests for vectorized termination, keys, Warp shared buffers and label/cost equivalence.
- [x] Extend continuation/device_rollout.py, unified_continuation_shards.py and policy_family_landing.py; expose configurable execution through existing CLIs.
- [x] Run CPU regression tests; audit landing/failure priority without rewriting historical outcomes.
- [x] Run a predeclared bounded TRAIN execution comparison on existing candidates, increasing batch size with time/memory limits, no PPO/acquisition/final TEST; preserve attempts, cost, plotting data and figures.
- [x] Record measured limits and recommended execution settings and review diff; commit/push delivery follows verified76-test run.

GPU initially shared with a STTW_CONTROL training process; never stop unrelated jobs. The engineering comparison does not authorize new scientific labels when outcomes differ, nor reopen accepted snapshot replay questions.

Outcome: genuine vectorized Warp continuation and explicit batch CLI are implemented.4096/8192/16384 capacity measurements completed;the final16384 run checked aggregate contact/CCD/constraint capacity at every physics substep, peak14,522MiB and157,227 useful simulator steps/s.24576 stopped at the22,000MiB protection threshold (observed22,542MiB), retaining full failure charge. Existing16-candidate endpoints agree, but one final checked continuation changes38→39 ticks;do not automatically replace locked serial scientific campaigns. Larger capacity is not a full-campaign speedup claim. This task charged3,384,811 interactions including inactive slots and failures, zero PPO/new scientific candidates/final TEST. Full artifacts:`JIT/runs/engineering/continuation_parallel_20260911/INDEX.md`;compact evidence:`JIT/review_evidence/parallel_capacity_20260911.json`.

## Roadmap implementation — 2026-09-11

The new `probe_bank.py prepare-existence` / `run-existence` path locks bank, TRAIN catalog, exact candidate indices, evaluator order, horizon, seed, sources and budget. It stops each candidate after a nonconflicting witness; skipped evaluators stay untested, and incomplete/error/conflicting outcomes stay unknown. Every declared evaluator must complete with failure before a no-witness result. Resume audits all historical reservations before allocating retries. Completed old campaigns remain unchanged.

Existing pi6 full-matrix scheduling analysis:98,093→15,388 useful continuation steps (84.31% hypothetical reduction); not measured wall-clock savings.64 evaluator-level contact/failure conflicts produce735 witnessed and3 unknown under the new quarantine view; historical738-positive union is not rewritten. Full artifacts and cost: `JIT/runs/engineering/existence_checkpoint_residual_20260911/INDEX.md`.

The opt-in campaign `--checkpoint-steps 32000 64000 128000 --ppo-steps 128000` declares fixed TRAIN panel identity and costs inside one live trainer. This is an option description, not a command executed here. Intermediate equal-budget exploration needs a separate diagnostic checkpoint identity; automatic extension and optimizer restart remain unsupported.

`residual_exploration.py` / `fit_residual_explorer.py` implement a bounded conditional supervised warm start. Historical snapshots do not supply matched per-action observations; actual production fitting requires an explicit matched TRAIN export. Frozen explorer state and identity must enter causal prefixes before acquisition integration. No learned exploration experiment, diffusion model, pi7 or production retraining was launched.

## Authorized multi-checkpoint training — 2026-09-11

User requested the corresponding local training after implementation. Prepared a separate development run `multicheckpoint_pi6_20260911`, initialized from frozen pi6 Actor/normalizer with fresh critic/optimizer, reusing immutable completed round_001 TRAIN support.128 environments,128,000 PPO transitions, seed9871101; fixed4-state TRAIN panels at32k/64k/128k, horizon400. Total worst-case132,800 interactions; one attempt,30-minute process limit; stop on numerical/device/checkpoint/evaluation failure, no automatic retry/extension/bank promotion or new acquisition/final TEST. This measures the new checkpoint path; it does not reconstruct old checkpoints or establish equal-budget exploration gains. Configuration:`JIT/configs/pi6_multicheckpoint_train_20260911.json`; declaration/results:`JIT/runs/experiments/multicheckpoint_pi6_20260911/`.

Outcome: this run completed128,000 PPO transitions and225 panel steps (128,225 total), exit0,58.835s process wall.32k/64k/128k each scored4/4 on the same4-state TRAIN panel, with no physical failures. Four inference checkpoints0/32k/64k/128k and complete trajectories/plots/CSV/cost are saved. This validates checkpoint execution; it does not establish improvement, convergence or exploration gains. No automatic continuation or bank promotion. See `JIT/runs/experiments/multicheckpoint_pi6_20260911/INDEX.md` and `JIT/review_evidence/multicheckpoint_train_20260911.json`.

## Authorized checkpoint exploration comparison — 2026-09-11

User approved comparing original pi6 with32k/64k/128k under common exploration conditions and total costs. Add explicit TRAIN-only diagnostic checkpoint identities (no formal pi_N/envelope promotion), then use one common pi0–pi6 evaluator bank. Each proposer gets8 real forward trajectories: all4 action channels,±0.15,anchor x2.9,lookback0.15m,same acquisition seed9881101 and label seed9881601. Fixed complete x2.5 start,5cm real-frame samples,max64 candidates/trajectory,400 tick horizon; no new PPO/final TEST. Per-arm ceiling1,500,000 interactions (6,000,000 maximum overall), single attempt;600s acquisition/evaluator worker limits. Serial scientific continuation remains selected. All failures/padding charged.

Full matched-schedule results and conservative catalog-order common-budget replay are distinct views. Training-inclusive costs add32,076/64,152/128,225 to the corresponding checkpoints, with pi6's inherited training shared baseline. Historical campaign witnessed root-cell CSV is a frozen conservative novelty-exclusion baseline; it does not silently adopt new conflict semantics. Preserve phase geometry gaps, all rows/trajectories/unknowns, plots and costs. Config:`JIT/configs/checkpoint_discovery_pi6_20260911.json`; outputs:`JIT/runs/experiments/checkpoint_discovery_20260911/`. Implementations and source locks are verified before launch; no automatic promotion or extra training follows the result.

Completed comparison:20,733 new exploration interactions,1127.341s,no retries/PPO/final TEST. At common exploration budget4,058,novel cells pi6/32k/64k/128k=132/186/132/165. Full schedules yield146/186/182/205 novel cells at4,225/4,058/6,502/5,948 interactions. Clean forward landings8/8,7/8,6/8,7/8;64k has1 unknown and128k5 unknowns, preserved.32k is the best incremental novelty result at this small budget, while originalpi6 remains the strongest clean landing observation and lowest clean peak in this schedule. Most geometry overlaps; extra cells are not continuous volume. Training cost is not amortized by this pilot; pause further PPO and prioritize pi6/32k complementary verification, with independent repeats still needed. No automatic bank promotion. Full evidence:`JIT/runs/experiments/checkpoint_discovery_20260911/INDEX.md`;compact record:`JIT/review_evidence/checkpoint_discovery_20260911.json`.

## Bounded length pilot closure — 2026-09-11

User authorized a fresh pi6-initialized 256,000-transition run with unchanged seed9871101/support/recipe, checkpoints32k/64k/128k/192k/256k,128 environments. One attempt,1800s,8,000 maximum TRAIN-panel interactions,264,000 total ceiling. This supersedes the earlier pause recommendation; no optimizer restart represented as continuous training. Following one late-checkpoint exploration comparison, select a provisional working range and move to residual data/integration instead of further length sweeps. Prepared pi6/32k perturbation-condition plans remain unrun. Artifacts: `JIT/runs/experiments/multicheckpoint_pi6_256k_20260911/`.

256k PPO completed with392 validated panel interactions (256,392 total),84.061s,exit0; all five4-state TRAIN panels4/4,zero physical failures. These panels do not establish convergence or exploration improvement. Late pi6/128k/192k/256k matched exploration is running under a separate6M ceiling.63 relevant CPU tests passed in5.75s. Parallel zero-interaction residual data audit found809 saved states,777 potential observation/next-action pairs and90 nonzero pairs with later witnesses; sparse saved-state inputs reconstruct from FIFO/events, but explicit action-time/downstream-witness credit and trajectory-group split are required before fitting. Audit:`JIT/runs/engineering/residual_data_readiness_20260911/INDEX.md`.

Residual next-stage design: preserve a separate pre-action origin (state/context/tick, actual Actor observation and base/applied actions) and downstream witnessed target (state/context/root cell, endpoint protocol and bank identity), joined within one locked acquisition protocol/trajectory with verified prefix nesting and strictly increasing ticks. Freeze goal and baseline before fitting; no outcome-derived feature leakage. Compare sparse existing-pair warm start with a small fully recorded fixed-perturbation export if needed; group by trajectory/ancestor, no frame random split. Start bounded supervised fitting only after these provenance checks, then frozen explorer acquisition with declared additional history/slew state and equal-cost fixed-perturbation control. No diffusion or automatic length sweep.

## Residual export and warm-start implementation — 2026-09-11

Length pilot closed: late comparison completed24,916 exploration interactions in1232.074s; new training+panels+exploration281,308. At common4,135 exploration budget pi6/128k/192k/256k novelty134/122/103/161; full schedules134/177/176/237 and clean forward8/8,6/8,7/8,8/8. Use provisional128k–256k training ceilings with early snapshots retained, no further length sweep. Single-lineage developmental evidence, no universal optimality/convergence or amortized training advantage.

Implementation checklist (writing-plans and test-driven-development; established branch, no unrelated changes):
- [x] Add causal action-pair tests: post-action/off-by-one rejection, altered prefix, unknown/old-cell exclusion, trajectory-group separation, file tampering.
- [x] Add canonical residual dataset exporter with separately bound pre-action origin and strictly later same-trajectory witness; retain complete input hashes and exclusion counts. Reconstruct sparse original actor observations from stored FIFO/events without physics replay. Revalidate completed artifact identities, without resuming old runtime protocols or requiring old source files to equal current code.
- [x] Extend the existing warm-start loader to validate versioned causal exports while retaining legacy explicit-export compatibility. Freeze exogenous perturbation goals, training-only normalization and trajectory-group split; development rows remain TRAIN development, not final TEST or independent trials.
- [x] Export existing completed checkpoint comparison data, then execute one CPU supervised fit with seed9901101, learning_rate0.001,500 optimizer updates maximum,zero new interactions/PPO,one attempt. Stop on invalid provenance/nonfinite data or fit; no automatic retries or exploration performance claim.
- [x] Preserve model, full records, prediction CSV, figures, wall time and cost; verify tests/diff; commit and push. Frozen explorer integration and real equal-cost exploration remain the subsequent bounded stage.

Residual warm-start completed:500 supervised updates,543 fit rows,zero interactions/PPO,9.085s process wall. Fit RMSE0.02473→0.00100 but development RMSE0.02381→0.13010 (203 rows). No promotion: declared split holds both steering directions out of fit; constant direction feature/std floor maps development values to±1000. Added independently reviewed opt-in actor_fit_std_goal_units_v1, retaining legacy contracts; no refit or gain claimed. Next dataset must cover all4channels in both partitions under different whole trajectory conditions, with complete pre-action recording and fixed-unit goal scaling.76 related CPU tests pass; all results, failed first export, pictures/predictions and costs retained at `JIT/runs/experiments/residual_warmstart_20260911/INDEX.md`.
