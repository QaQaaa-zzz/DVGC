Latest action (2026-09-10): all frozen policies jointly explore before each new PPO; reuse completed pi_2/pi_4, initially add pi_0/pi_1/pi_3, then train pi_5 onward and retain old policies. Run the bounded two-round command in JIT/docs/JIT_ALL_PROPOSER_CAMPAIGN_20260910.md (repository-root path). Every round saves per-proposer PNG/PDF/SVG, exact plotting CSVs, 5 cm slices, receipts and provenance, before and after training. Fixed equal proposer allocation; adaptive allocation is not implemented. Novelty uses the cumulative inherited union; full bootstrap cost remains separate. CPU verified, production multi-round outcome pending. This supersedes older next-run notes.

# JIT agent instructions — active empirical-envelope direction

Next action (2026-09-09): recovery succeeded (442 seed + 734 novel = 1,176 campaign cells; not all historical Tube). Run `JIT/cli/compare_pi2_pi4_discovery.py --gpu 0`: reuse verified pi_4 evidence, run only pi_2 under the same 32-trajectory profile and identical five-evaluator bank, no new PPO. Report equal-cost discovery plus marginal training/known failure cost views and exclusive proposer contributions. This is a retrospective TRAIN control, not independent repetitions. CPU launchers no longer actively hide CUDA devices while selecting CPU. See JIT/docs/JIT_PI2_PI4_PAIRED_DISCOVERY_20260909.md (repository-root path).


Current action (2026-09-09): callback-fixed smoke completed pi_4 PPO (25,600 steps), freeze, 753 arrivals and all five evaluator panels. Final plotting failed with KeyError pi_4 because the old panel has only four policies. Use JIT/cli/analyze_envelope_campaign.py --source JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1 to recover reports with zero new simulation/PPO. It verifies source artifacts and labels, writes a separate sibling analysis directory, and preserves the original failure. Untested old-policy coverage is null, not zero. New panel union = 743 witnessed contexts / 734 root cells; historical novelty still needs this analysis. Do not retrain this smoke or run the old campaign command after source changes.


Production fix (2026-09-09): the first campaign smoke failed inside PPO because `JAX_PLATFORMS=cuda` hid the CPU device required by `jax.debug.callback`. Training children now use `cuda,cpu`: GPU stays the default, CPU is available for callbacks. No completed pi_4 or discovery round was reported. The 27,200 charged interactions are the failed attempt reservation, not a measured completed-transition count. Preserve the original directory and retry under `JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1`; code-identity locks intentionally reject in-place continuation after source changes. Both runs' costs must be included in total experiment accounting. See JIT/docs/JIT_AUTONOMOUS_ENVELOPE_20260909.md.


Latest authorized action (2026-09-09): run a bounded autonomous exploration/learning campaign via `JIT/cli/run_envelope_campaign.py`. This supersedes historical no-PPO/stop-scan next-run notes below. New full-context TRAIN support adapter uses 20% fixed x=2.5 starts and 80% witnessed snapshots, Actor+normalizer warm start, fresh critic/optimizer, first-valid-landing termination and growing frozen probe banks. Keep historical results immutable. Stop on budget, round cap, or consecutive low novel-cell gains; never label finite search stagnation as a proven physical boundary. CPU tests do not establish GPU training success. First production command and complete method are in JIT/docs/JIT_AUTONOMOUS_ENVELOPE_20260909.md (path from repository root). Compact reports auto-publish to agent/jit-run-reports; no more manual ZIP upload unless publication fails.


User-authorized result publication (2026-09-09): top-level compact bundles under JIT/runs automatically publish JSON/CSV/Markdown/log evidence to the isolated agent/jit-run-reports branch using origin and existing server credentials. No checkpoint/snapshot/image upload. Publish failure preserves local results; retry via JIT/cli/publish_results.py --output-dir <run>. Read this results branch when the user reports completion instead of requiring ZIP upload. JIT_AUTO_PUBLISH=0 disables automatic publication.

Current user-directed next run (2026-09-09): support validation completed (195 witnessed, 95 unwitnessed; pi_2 initializer). Run JIT/cli/explore_lower_boundary.py --gpu 0 to explore lower successful support through real signed hip/knee action perturbations. Sixteen pi_2 trajectories; four evaluators; no PPO or state-height injection. Report observed 5 cm phase-separated slice minima and the lowest control-step peak among complete successful forward trajectories separately. Neither is a global physical minimum; slice minima need not form one trajectory. See JIT/docs/JIT_LOWER_BOUNDARY_20260909.md. This user request supersedes the earlier pause on further scanning.


Current result/action (2026-09-08): corrected local boundary completed nine trajectories, 290 arrivals, 195 bank witnesses, 95 no-witness states and 27,954 interactions. All three x=2.85 window endpoints landed; all six x=2.90/2.95 ended with physical_failure. This is one development repetition, not nine independent repeats. Stop broad/local resweeps now. Run JIT/cli/prepare_complementary_support.py (CPU only) to validate exact source snapshots/labels and emit witnessed support plus separate unwitnessed targets. pi_2 leads this TRAIN panel (172 successes). Support construction is not a runnable PPO recipe: reset adapter, mixture, budget/control and training smoke remain explicit work. See JIT/docs/JIT_PAPER_EXPERIMENT_GAPS_20260908.md.


Budget fix (2026-09-08): first knee-boundary attempt failed before rollout because the collector requires a conservative 6,000-step acquisition reservation. The profile now matches that bound, with the same nine trajectories and 2,000,000 total budget. Default output is JIT/runs/discovery/knee_boundary_v1_budgetfix; keep the failed knee_boundary_v1 directory and its cost record.

Current next run (2026-09-08): JIT/cli/refine_jump_boundary.py --gpu 0 runs nine pi_1 positive-knee trajectories (offsets 0.15/0.175/0.20; window endpoints x=2.85/2.90/2.95 m), with all four frozen continuation evaluators. Source TRAIN frontier evidence is hash-locked. This is local gap refinement, not a global action restriction, independent-seed study, or PPO. Default budget 2,000,000 interactions including failures/retries; first-attempt ceiling 1,849,200. Return compact results_to_send.zip. See JIT/docs/JIT_BOUNDARY_REFINEMENT_20260908.md from the repository root. Older next-run notes below are historical.


Latest production result (2026-09-08): landing_frontier_v1 completed 48 trajectories, 1,141 arrivals, 1,129 bank witnesses, +1,107 root cells (cumulative 3,463), with 92,058 interactions and no PPO. Forty trajectories landed; eight terminated before landing; none hit sampling limits. Twelve no-witness states come from one pi_1 positive-knee 0.2 trajectory, not twelve independent failures. Next: targeted TRAIN boundary refinement and a locked complementary training recipe, not another complete broad scan or automatic PPO. Review bundles are now compact by default; use JIT/cli/package_results.py --output-dir <existing-run> to repackage without simulation. See JIT/docs/JIT_FRONTIER_RESULT_AND_PACKAGING_20260908.md (relative to repository root). Earlier next-run notes below are historical.


Current action after four-proposer production results (2026-09-08): 1,116 arrivals, 1,115 bank witnesses and +904 root cells; cumulative baseline 2,356. All formal labels used serial execution. Run [landing/frontier exploration](docs/JIT_LANDING_FRONTIER_20260908.md) via JIT/cli/explore_frontier.py --gpu 0. Use a locked TRAIN-informed 48-trajectory allocation, stronger legal action offsets, and an explicitly versioned sampling guard to x=8 m while retaining the original pi_0 reference. Report landing/failure/truncation and separate old-corridor novelty from newly observed extended-domain support. Skip repeated device benchmarks; preserve serial identity checks and cost accounting. No PPO, no extra snapshot replay checks, no automatic training admission. Older “next” actions below remain historical.


Latest action (2026-09-08): the 5 cm pi_0 pilot completed with 242 arrivals, 240 bank witnesses and +227 root cells (cumulative 1,452 versus the prior TRAIN panel). All four Warp device benchmarks failed at contact__dim; serial fallback completed labels. The user authorized continuing with the single-world device-map fix and four-proposer discovery comparison. Follow [the current run guide](docs/JIT_MULTI_PROPOSER_DISCOVERY_20260908.md) and run JIT/cli/compare_discovery.py --gpu 0. Preserve prior labels, 5 cm real-frame sampling, the old physical grid and accepted replay limits. New discovery counts use the old shared TRAIN union plus the completed dense pilot. No PPO, no extra snapshot replay investigation; one exploration seed remains development evidence. Earlier “next run” instructions below are historical.


Latest action (2026-09-07): the four-policy comparison completed in production (48 shards, 197,604 new interactions). The user authorized 5 cm real-frame multi-state acquisition and measured label acceleration, with all four legal action channels allowed. Run [the bounded dense Tube pilot](docs/JIT_DENSE_TUBE_PILOT_20260907.md) next; it automatically compares serial/device execution and falls back to serial if needed. Preserve the original centerline and physical-grid resolution, all old evidence, and the accepted replay limitation. No new PPO or additional snapshot replay investigation. This is a 16-trajectory TRAIN pilot, not a completed matched-budget multi-proposer experiment. Earlier run instructions below are historical.


Current user decision (2026-09-07): no additional numerical replay validation. About 3.1 cm initial wheel clearance is accepted as the declared near-ground reset. Run [the four-policy comparison](docs/JIT_POLICY_ENVELOPE_COMPARISON_20260907.md), preserve the original three-member family, and plot each policy plus union on identical reached-state panels. This is continuation support conditioned on pi_0 arrivals, not each policy's own forward exploration envelope.

The root [AGENTS.md](../AGENTS.md) and user-confirmed [paper outline](docs/JIT_PAPER_OUTLINE.md) govern this work. Audited baseline: `bfc22f2`. The first implementation update is documented in [implementation status](docs/JIT_PROBE_BANK_IMPLEMENTATION_20260905.md); remaining requirements are not runtime success claims.

## Objective and objects

Bootstrap with up/down phase support and a successful unified seed, then discover empirical jumping support using complementary frozen probes. Do not require one Actor to retain/realize the entire cumulative Tube.

Keep forward arrivals, exact landing witnesses, empirical support, projected cells, training Tube and individual-Actor realization distinct. Historical Tube0 is value-weighted (222 rows, including 42 negative phase labels), not an all-positive capability set. Original unified training completion and later Round1 pi_0 identity are distinct records.

## Arrival and continuation

- Begin at the complete declared near-ground jump-start state, x=2.5 m. Do not demand or claim the earlier natural approach.
- Keep pi_0 centerline coordinates fixed; all centerline frames are captured, not interpolated.
- New protocols may use multiple frozen proposers and evaluators. Lock membership before each round and record bank versions.
- Each candidate needs exact prefix/suffix context, action history, events, start/remaining time and source identities. qpos/qvel identity alone is insufficient for policy-conditioned continuation.
- RSI may restore a verified ancestor or candidate, but cannot invent a prefix. Current source only restarts from the fixed start; general ancestor reuse needs implementation and replay checks.
- A first-valid-landing witness is sufficient. Negative means no success under the declared evaluators/horizon, not physical infeasibility.
- Do not discard evidence just because physical coordinates already occur in the training Tube. Deduplicate coverage separately.
- Keep incomplete/error/untested outcomes separate from completed failed rollouts.

## Legacy execution versus new method

The already locked wide/expanded scans retain pi_0 proposer and pi_0/pi_1/pi_2 evaluators. Resume under their exact catalog, seed, horizon and endpoint. Do not rename or retrofit those runs as multi-probe discovery.

Use `cli/probe_bank.py` for new bank pilots. The six-state production smoke has completed; the user accepted observed numerical differences and declined more replay validation. It emits observations with replay/physical-envelope claims explicitly unverified; it does not authorize automatic PPO or certify physical cells. The old `prepare_iterative_envelope_workflow.py` selects one successor and requires coverage eligibility. It is legacy automation, not the new discovery scheduler. Do not launch a new experiment through that chain without changing and validating its contracts.

pi_3 remains a frozen trained candidate. Its mixed-endpoint gate is quarantined as comparison evidence, not a blanket rejection of every pi_3 witness. Assess technical probe eligibility and complementary witnesses under a new protocol. Do not automatically promote the old selected manifest.

## Immediate correctness gates

Historical 2026-09-06 diagnostic entry: `cli/validate_jump_evidence.py --gpu 0`, using the production interpreter and `PYTHONPATH=JIT/src` from repository root. Follow the [run/return guide](docs/JIT_GPU_EVIDENCE_VALIDATION_20260906.md). It locks source and policy identities, captures a real ground prefix, compares live/preserved/fresh-counter continuations, and runs serial plus fresh-process shards. Preserve the generated ZIP even on failure. Its legacy `split=train` catalog marker is accompanied by `logical_role=engineering_validation` and no training/admission authorization. Never use this panel for adaptive training or silently promote replay-verification flags.

See issue IDs and evidence in [review](docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md).

1. Preserve the implemented evaluator/cache identity checks, per-row endpoint checks and staged merge publishing; validate them against production artifacts.
2. Preserve endpoint validation in both legacy analysis and selector; mixed/missing endpoint evidence now refuses.
3. Separate training-support membership, exact witness identity and physical novelty.
4. Validate public warm-start routing; preserve Actor/normalizer only, with fresh critic/optimizer unless explicitly declared.
5. Preserve the completed six-state serial/shard production check. Numerical replay differences are accepted for continuing work; do not schedule additional replay validation without a new user request.
6. Extend the first versioned bank/observation index and suffix-attempt ledger to validated cumulative physical support and full end-to-end cost before formal discovery training.

Shards must preserve catalog/global index/seed/horizon/endpoint/full policy identity. The historical operational suggestion is at most 600 candidates per fresh GPU process; it is not an enforced code limit or measured universal safe capacity. Start small, end the process between shards, and avoid concurrent evaluators without measured capacity. Completed cache entries need full requested-contract verification; preserve failures.

## Predictor

Optional, advisory, never proof of arrival or a Tube admission label. Preserve repaired score self-hash, target-family/catalog binding and tie-invariant AP. Historical locks lacking target identity must remain explicitly marked. Do not use ACCEPTANCE labels to decide whether TRAIN fitting is authorized. Unsupported/single-class outcomes need explicit metrics availability, not fabricated negatives. A predictor-free discovery experiment does not wait for predictor quality.

## Training and evidence roles

- Predeclare training support, frozen initializer, normalizer handling, reset distribution, reward, steps, seed, probe admission rule and stopping rule.
- Existing configs say `natural_reset_probability` and `existing_phase_u_natural_reset`; these are historical training semantics, not automatically the x=2.5 task reset. Lock a deliberate training mixture and adapt code before using a new reset meaning.
- TRAIN only supplies adaptive acquisition/training support. Preserve CALIBRATION/ACCEPTANCE isolation across bank versions and ancestor chains.
- Final TEST/JCE/JEL stays unopened. Old bootstrap `test` labels are already historical development/evaluation evidence and cannot be advertised as untouched final tests.
- Keep phase-specific successes/failures and individual-probe contributions, not only family totals. A weak Actor may still add unique witnesses; no full-Tube retention gate for witness validity.
- Account all bootstrap, acquisition, labels, retries, PPO and development evaluation; independent seeds are not checkpoints from one seed.

## Physical metrics

Reuse `analysis/capability_tube.py` resolution contracts. Root geometry includes position/velocity **and** orientation/angular rate; full physical adds joints/rates and wheel tangential velocity. Existing bins are 0.10 m, 0.10 m/s, 0.50 degrees and 2 degrees/s as applicable, with discrete phase. Report resolution sensitivity and physical support separately from all-state replay projections.

## Runtime and maintenance

Use `/home/qy/mujoco_playground/.venv/bin/python` for production. A temporary CPU test stack was installed for this code change. Its tests do not replace production GPU smoke, actual checkpoint restore, PPO or serial/sharded rollout equivalence.

Keep durable logic in `src/jit_dvgc`, CLIs thin, tests in `tests`, and guidance in `docs`. Preserve history and unrelated work; never reset/clean/stash/rebase/force-push. Do not remove identity checks to make a run pass. No large training is triggered by editing documentation.

## Read order

1. Root AGENTS and PROJECT.
2. [CURRENT_STATUS](docs/CURRENT_STATUS.md).
3. [Paper outline](docs/JIT_PAPER_OUTLINE.md) and [review](docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md).
4. [Protocol](docs/ENVELOPE_ITERATION_PROTOCOL.md) and [training roadmap](docs/JIT_TRAINING_ROADMAP.md).
5. [Handoff](docs/CODEX_HANDOFF_20260904.md), [code organization](docs/CODE_ORGANIZATION.md), and relevant immutable run records.
