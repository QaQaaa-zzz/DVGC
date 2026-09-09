# DVGC repository authority — empirical jumping envelope

Current action (2026-09-09): callback-fixed smoke completed pi_4 PPO (25,600 steps), freeze, 753 arrivals and all five evaluator panels. Final plotting failed with KeyError pi_4 because the old panel has only four policies. Use JIT/cli/analyze_envelope_campaign.py --source JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1 to recover reports with zero new simulation/PPO. It verifies source artifacts and labels, writes a separate sibling analysis directory, and preserves the original failure. Untested old-policy coverage is null, not zero. New panel union = 743 witnessed contexts / 734 root cells; historical novelty still needs this analysis. Do not retrain this smoke or run the old campaign command after source changes.


Production fix (2026-09-09): the first campaign smoke failed inside PPO because `JAX_PLATFORMS=cuda` hid the CPU device required by `jax.debug.callback`. Training children now use `cuda,cpu`: GPU stays the default, CPU is available for callbacks. No completed pi_4 or discovery round was reported. The 27,200 charged interactions are the failed attempt reservation, not a measured completed-transition count. Preserve the original directory and retry under `JIT/runs/campaign/empirical_envelope_smoke_callbackfix_v1`; code-identity locks intentionally reject in-place continuation after source changes. Both runs' costs must be included in total experiment accounting. See JIT/docs/JIT_AUTONOMOUS_ENVELOPE_20260909.md.


Latest authorized action (2026-09-09): run a bounded autonomous exploration/learning campaign via `JIT/cli/run_envelope_campaign.py`. This supersedes historical no-PPO/stop-scan next-run notes below. New full-context TRAIN support adapter uses 20% fixed x=2.5 starts and 80% witnessed snapshots, Actor+normalizer warm start, fresh critic/optimizer, first-valid-landing termination and growing frozen probe banks. Keep historical results immutable. Stop on budget, round cap, or consecutive low novel-cell gains; never label finite search stagnation as a proven physical boundary. CPU tests do not establish GPU training success. First production command and complete method are in JIT/docs/JIT_AUTONOMOUS_ENVELOPE_20260909.md (path from repository root). Compact reports auto-publish to agent/jit-run-reports; no more manual ZIP upload unless publication fails.


User-authorized result publication (2026-09-09): top-level compact bundles under JIT/runs automatically publish JSON/CSV/Markdown/log evidence to the isolated agent/jit-run-reports branch using origin and existing server credentials. No checkpoint/snapshot/image upload. Publish failure preserves local results; retry via JIT/cli/publish_results.py --output-dir <run>. Read this results branch when the user reports completion instead of requiring ZIP upload. JIT_AUTO_PUBLISH=0 disables automatic publication.

Current user-directed next run (2026-09-09): support validation completed (195 witnessed, 95 unwitnessed; pi_2 initializer). Run JIT/cli/explore_lower_boundary.py --gpu 0 to explore lower successful support through real signed hip/knee action perturbations. Sixteen pi_2 trajectories; four evaluators; no PPO or state-height injection. Report observed 5 cm phase-separated slice minima and the lowest control-step peak among complete successful forward trajectories separately. Neither is a global physical minimum; slice minima need not form one trajectory. See JIT/docs/JIT_LOWER_BOUNDARY_20260909.md. This user request supersedes the earlier pause on further scanning.


Current result/action (2026-09-08): corrected local boundary completed nine trajectories, 290 arrivals, 195 bank witnesses, 95 no-witness states and 27,954 interactions. All three x=2.85 window endpoints landed; all six x=2.90/2.95 ended with physical_failure. This is one development repetition, not nine independent repeats. Stop broad/local resweeps now. Run JIT/cli/prepare_complementary_support.py (CPU only) to validate exact source snapshots/labels and emit witnessed support plus separate unwitnessed targets. pi_2 leads this TRAIN panel (172 successes). Support construction is not a runnable PPO recipe: reset adapter, mixture, budget/control and training smoke remain explicit work. See JIT/docs/JIT_PAPER_EXPERIMENT_GAPS_20260908.md.


Budget fix (2026-09-08): first knee-boundary attempt failed before rollout because the collector requires a conservative 6,000-step acquisition reservation. The profile now matches that bound, with the same nine trajectories and 2,000,000 total budget. Default output is JIT/runs/discovery/knee_boundary_v1_budgetfix; keep the failed knee_boundary_v1 directory and its cost record.

Current next run (2026-09-08): JIT/cli/refine_jump_boundary.py --gpu 0 runs nine pi_1 positive-knee trajectories (offsets 0.15/0.175/0.20; window endpoints x=2.85/2.90/2.95 m), with all four frozen continuation evaluators. Source TRAIN frontier evidence is hash-locked. This is local gap refinement, not a global action restriction, independent-seed study, or PPO. Default budget 2,000,000 interactions including failures/retries; first-attempt ceiling 1,849,200. Return compact results_to_send.zip. See JIT/docs/JIT_BOUNDARY_REFINEMENT_20260908.md from the repository root. Older next-run notes below are historical.


Latest production result (2026-09-08): landing_frontier_v1 completed 48 trajectories, 1,141 arrivals, 1,129 bank witnesses, +1,107 root cells (cumulative 3,463), with 92,058 interactions and no PPO. Forty trajectories landed; eight terminated before landing; none hit sampling limits. Twelve no-witness states come from one pi_1 positive-knee 0.2 trajectory, not twelve independent failures. Next: targeted TRAIN boundary refinement and a locked complementary training recipe, not another complete broad scan or automatic PPO. Review bundles are now compact by default; use JIT/cli/package_results.py --output-dir <existing-run> to repackage without simulation. See JIT/docs/JIT_FRONTIER_RESULT_AND_PACKAGING_20260908.md (relative to repository root). Earlier next-run notes below are historical.


Current action after four-proposer production results (2026-09-08): 1,116 arrivals, 1,115 bank witnesses and +904 root cells; cumulative baseline 2,356. All formal labels used serial execution. Run [landing/frontier exploration](JIT/docs/JIT_LANDING_FRONTIER_20260908.md) via JIT/cli/explore_frontier.py --gpu 0. Use a locked TRAIN-informed 48-trajectory allocation, stronger legal action offsets, and an explicitly versioned sampling guard to x=8 m while retaining the original pi_0 reference. Report landing/failure/truncation and separate old-corridor novelty from newly observed extended-domain support. Skip repeated device benchmarks; preserve serial identity checks and cost accounting. No PPO, no extra snapshot replay checks, no automatic training admission. Older “next” actions below remain historical.


Latest action (2026-09-08): the 5 cm pi_0 pilot completed with 242 arrivals, 240 bank witnesses and +227 root cells (cumulative 1,452 versus the prior TRAIN panel). All four Warp device benchmarks failed at contact__dim; serial fallback completed labels. The user authorized continuing with the single-world device-map fix and four-proposer discovery comparison. Follow [the current run guide](JIT/docs/JIT_MULTI_PROPOSER_DISCOVERY_20260908.md) and run JIT/cli/compare_discovery.py --gpu 0. Preserve prior labels, 5 cm real-frame sampling, the old physical grid and accepted replay limits. New discovery counts use the old shared TRAIN union plus the completed dense pilot. No PPO, no extra snapshot replay investigation; one exploration seed remains development evidence. Earlier “next run” instructions below are historical.


Latest action (2026-09-07): the four-policy comparison completed in production (48 shards, 197,604 new interactions). The user authorized 5 cm real-frame multi-state acquisition and measured label acceleration, with all four legal action channels allowed. Run [the bounded dense Tube pilot](JIT/docs/JIT_DENSE_TUBE_PILOT_20260907.md) next; it automatically compares serial/device execution and falls back to serial if needed. Preserve the original centerline and physical-grid resolution, all old evidence, and the accepted replay limitation. No new PPO or additional snapshot replay investigation. This is a 16-trajectory TRAIN pilot, not a completed matched-budget multi-proposer experiment. Earlier run instructions below are historical.


Updated 2026-09-05 after the user's paper-outline decisions. This is the active research direction. The original audit baseline was `bfc22f2e32cb78cb269b0e522c3bdd7c6e7a8d42`. Correctness fixes and a first probe-bank path now exist; [implementation status](JIT/docs/JIT_PROBE_BANK_IMPLEMENTATION_20260905.md) distinguishes CPU verification from pending production gates.

## Research objective

JIT studies **bootstrapping and budget-controlled discovery of an empirical jumping capability envelope** for the fixed bicycle–pendulum robot and operating condition. Phase-specific up/down learning supplies initial reset support; a successful frozen unified policy supplies a seed trajectory; complementary frozen probes discover additional forward-arrived states with successful landing continuations.

The primary result is valid new physical support versus total environment interactions. One Actor's realization is a separate diagnostic/application result. **Do not require a new Actor to cover the whole cumulative Tube or replace all previous policies.** A regression by a later policy does not erase a valid historical witness.

## User-confirmed scientific contract

- Task begins at the declared complete near-ground jump-start state at `x = 2.5 m`, including pose, velocities, controller/event history and time semantics. Earlier natural-reset approach is outside scope.
- A state enters empirical support only with real forward dynamics from that start (or a fully verified ancestor chain) and a successful continuation from the **same exact state and required context**.
- Success is `first_valid_landing` before declared failure/horizon; recovery is not required. One observed success is a witness, not a calibrated success probability or safety guarantee.
- Multiple frozen policies may provide forward proposals and continuation evaluations. Membership and roles are versioned; one forward rollout uses its declared frozen proposer plus bounded perturbations. Different prefix/suffix policies are allowed as offline witnesses, not claimed as one-Actor execution.
- Keep the real-frame pi_0 centerline fixed as longitudinal coordinates. It is not an Actor command, reward target, tracking trajectory or interpolated reachable corridor.
- All tried policies failing means `no_success_witness_under_declared_bank`, not physical infeasibility. Untested and incomplete engineering attempts are separate states.
- Never infer formal reachability, viability, safe invariance, continuous feasible volume, the complete physical limit, or universal Actor impossibility.

## Objects that must stay separate

| Object | Meaning |
| --- | --- |
| `R_hat` / arrival evidence | Exact states reached with auditable prefix provenance |
| landing witness | A declared frozen evaluator succeeded from that exact state |
| `T_hat` / empirical support | Exact arrival states with at least one valid landing witness |
| physical cells | Declared projection of witnessed states; a cell does not certify every state inside |
| `S` / training Tube | Reset/replay support, including historical rows without current arrival/landing evidence |
| `Pi` / probe bank | Immutable policy records with explicit proposer/evaluator roles and version |
| Actor realization | A single policy's results on a declared common panel |

Deduplicate physical coverage separately from storing witnesses. A row already in `S` may still need its first arrival or continuation witness. Equal qpos/qvel does not establish equal FIFO, event state, controller context or remaining time.

## Fixed runtime

Current action (2026-09-07): run `JIT/cli/compare_policy_envelopes.py --gpu 0` following the [comparison guide](JIT/docs/JIT_POLICY_ENVELOPE_COMPARISON_20260907.md). The user accepted approximately 0.031 m initial wheel clearance and the observed numerical replay differences, and explicitly declined further replay validation. The production six-state panel completed with consistent landing outcomes and serial/shard labels; exact trajectory equivalence did not pass. Preserve that record, but do not block the authorized labeling/plotting work on additional replay gates. Do not change reset physics or falsely mark exact replay verified.

- Repository target branch: `agent/two-phase-soft-tube`; isolated review branches/worktrees may be used to preserve concurrent work.
- XML: `assets/orange_bike_4kg_horizontal.xml`.
- Recorded XML identity: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`.
- Payload: 2 kg; simulation step: 0.005 s; control interval: 0.020 s.
- Actions: `[steer, rear-wheel drive, hip, knee]`; hip/knee limits: +/-30 N m.
- Production Python: `/home/qy/mujoco_playground/.venv/bin/python`.

Do not change physics, reward, endpoint, reset semantics or action order silently. If the production runtime is unavailable, report the limit; source/fixture checks are not GPU rollout validation.

## Historical evidence and migration boundary

The locked 2026-09-04/05 scans used pi_0 as proposer and exactly pi_0/pi_1/pi_2 as evaluators. Finish their missing labels under those identities; do not retrofit a larger bank or new seed into their outputs.

Legacy family and selected-policy workflows retain their original contracts. New `JIT/cli/probe_bank.py` supports versioned proposer/evaluator banks, separate causal catalogs, bounded fresh-process suffix jobs, attempt accounting and an observation index. CPU fixture tests pass; GPU equivalence, cumulative physical-cell accounting, cross-version isolation, probe admission and complementary training are still pending.

The historical pi_3 mixed-endpoint gate remains invalid as a fair comparison. Keep the trained checkpoint and valid underlying outcomes. pi_3 may be assessed as a prospective probe under a new identity/endpoint contract without requiring old full-Tube retention, but never reuse its old selected manifest as automatic authority.

New formal probe training is not ready: correctness guards and the first bank path are implemented, but a full cost/coverage ledger and the training recipe/budget remain to close. The user has accepted the numerical replay limitations; further replay validation is not a prerequisite for the authorized comparisons. Existing frozen-probe pilots should precede additional large PPO runs. A predictor is optional and must not block a predictor-free discovery experiment once essential gates pass.

## Data roles and evidence integrity

- TRAIN may guide exploration, train probes and supply reset support.
- CALIBRATION calibrates optional predictors; ACCEPTANCE is development evidence when used for decisions.
- Final TEST/JCE/JEL remains unopened for this work. Historical bootstrap files have their own splits named `test`; do not present those already-used splits as untouched final tests.
- Re-audit role isolation across every proposer, bank version, ancestor and training Tube. Shared ancestors/proposal groups imply correlated samples.
- Lock task, complete start/context, catalog, proposer/evaluator identities, endpoint, seed, horizon, remaining-time rule, role, cell resolution and budget before execution.
- A hash proves content identity, not when it was frozen. Preserve pre-outcome/pre-training records and history.
- Cache reuse and merges must verify the full requested contract before publishing; engineering failures never become physical negative labels.
- Keep raw runs immutable. Add new derived views and machine-readable eligibility records; never relabel historical results to manufacture a pass.

## Paper evidence and cost

Report raw candidates, witnessed exact states, novel physical cells, failed attempts, individual-probe contributions and single-Actor realization separately. Distinguish new arrival discovery from a new suffix witness on an old arrival.

Count expert/bootstrap training, proposal prefixes, all evaluator rollouts, unsuccessful/excluded acquisitions, PPO, development evaluations and failed retries. Shared bootstrap can be reported separately but belongs in the end-to-end total. Use matched-budget comparisons and independent seeds/group-aware uncertainty.

The existing Tube0 is value-weighted training support, not an all-success capability set: 222 rows include 42 historical negative labels. The successful pi_0 identity used by current scans is the later Round1 artifact; do not equate it with the first completed unified PPO run.

## Repository work

The user explicitly authorized subsequent routine commits and pushes to `agent/two-phase-soft-tube` within this project scope; do not repeatedly request permission. Preserve unrelated changes; never reset/clean/stash/rebase/force-push. Keep durable logic in `JIT/src/jit_dvgc/`, thin CLIs in `JIT/cli/`, tests in `JIT/tests/`, and research guidance in `JIT/docs/`. Extend existing capabilities rather than adding iteration-specific duplicate modules. Retain provenance checks; do not repeatedly recalculate locked hashes without a concrete identity question.

## Read order

1. This `AGENTS.md` and [JIT/AGENTS.md](JIT/AGENTS.md).
2. [PROJECT.md](PROJECT.md) and [CURRENT_STATUS.md](JIT/docs/CURRENT_STATUS.md).
3. [Paper outline](JIT/docs/JIT_PAPER_OUTLINE.md).
4. [Code/evidence review](JIT/docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md).
5. [Iteration protocol](JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md).
6. [Training roadmap](JIT/docs/JIT_TRAINING_ROADMAP.md).
7. [Handoff](JIT/docs/CODEX_HANDOFF_20260904.md) and [code organization](JIT/docs/CODE_ORGANIZATION.md).

Dated older reports and run manifests remain historical evidence, not overrides of this user-confirmed direction.
