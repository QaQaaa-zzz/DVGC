Latest action (2026-09-10): all frozen policies jointly explore before each new PPO; reuse completed pi_2/pi_4, initially add pi_0/pi_1/pi_3, then train pi_5 onward and retain old policies. Run the bounded two-round command in JIT/docs/JIT_ALL_PROPOSER_CAMPAIGN_20260910.md (repository-root path). Every round saves per-proposer PNG/PDF/SVG, exact plotting CSVs, 5 cm slices, receipts and provenance, before and after training. Fixed equal proposer allocation; adaptive allocation is not implemented. Novelty uses the cumulative inherited union; full bootstrap cost remains separate. CPU verified, production multi-round outcome pending. This supersedes older next-run notes.

# Current JIT status — empirical-envelope implementation, 2026-09-07

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


Current action after four-proposer production results (2026-09-08): 1,116 arrivals, 1,115 bank witnesses and +904 root cells; cumulative baseline 2,356. All formal labels used serial execution. Run [landing/frontier exploration](JIT_LANDING_FRONTIER_20260908.md) via JIT/cli/explore_frontier.py --gpu 0. Use a locked TRAIN-informed 48-trajectory allocation, stronger legal action offsets, and an explicitly versioned sampling guard to x=8 m while retaining the original pi_0 reference. Report landing/failure/truncation and separate old-corridor novelty from newly observed extended-domain support. Skip repeated device benchmarks; preserve serial identity checks and cost accounting. No PPO, no extra snapshot replay checks, no automatic training admission. Older “next” actions below remain historical.


Latest action (2026-09-08): the 5 cm pi_0 pilot completed with 242 arrivals, 240 bank witnesses and +227 root cells (cumulative 1,452 versus the prior TRAIN panel). All four Warp device benchmarks failed at contact__dim; serial fallback completed labels. The user authorized continuing with the single-world device-map fix and four-proposer discovery comparison. Follow [the current run guide](JIT_MULTI_PROPOSER_DISCOVERY_20260908.md) and run JIT/cli/compare_discovery.py --gpu 0. Preserve prior labels, 5 cm real-frame sampling, the old physical grid and accepted replay limits. New discovery counts use the old shared TRAIN union plus the completed dense pilot. No PPO, no extra snapshot replay investigation; one exploration seed remains development evidence. Earlier “next run” instructions below are historical.


Latest action (2026-09-07): the four-policy comparison completed in production (48 shards, 197,604 new interactions). The user authorized 5 cm real-frame multi-state acquisition and measured label acceleration, with all four legal action channels allowed. Run [the bounded dense Tube pilot](JIT_DENSE_TUBE_PILOT_20260907.md) next; it automatically compares serial/device execution and falls back to serial if needed. Preserve the original centerline and physical-grid resolution, all old evidence, and the accepted replay limitation. No new PPO or additional snapshot replay investigation. This is a 16-trajectory TRAIN pilot, not a completed matched-budget multi-proposer experiment. Earlier run instructions below are historical.


Latest action (2026-09-07): run [the four-policy envelope comparison](JIT_POLICY_ENVELOPE_COMPARISON_20260907.md). The production diagnostic completed in 792 interactions: six states, all four continuation arms landed with matched per-state landing steps; serial/shard labels matched. Exact prefix replay passed 1/6, preserved/fresh restore 0/6, counter behavior 5/6. The user accepted these numerical differences and the 3.1 cm initial wheel clearance, and explicitly declined further replay validation. [Original result and user decision](verification/jump_evidence_user_run_20260907.json) preserve the failed numerical gates. They do not block the authorized shared-panel labeling and plots. The new CLI reuses validated old labels, completes missing pi_0/pi_1/pi_2 labels, evaluates pi_3 under the same endpoint on a separate comparison plan, and exports role-separated figures and physical contributions. No new PPO or replay validation is scheduled.

## Historical 2026-09-06 preparation: GPU evidence validation

The current comparison implementation passed **46 focused CPU tests** for the supervisor, preflight, common-panel metrics/plots, family labels and shards ([verification record](verification/policy_comparison_cpu_20260907.json)). Actual production comparison results are pending the user's server run.

`JIT/cli/validate_jump_evidence.py` is ready for the production host. See the [exact command and ZIP return guide](JIT_GPU_EVIDENCE_VALIDATION_20260906.md). Default: unique Round1 pi_0, one real x=2.5 ground prefix, up to six reached states, four continuation controls per state, serial versus fresh-process shards, and at most 17,200 env.step calls for horizon 400. No PPO, old-run mutation, Tube admission or final TEST is involved.

**59 focused CPU tests passed** for the new diagnostic plus continuation labels/shards, policy-family integrity and probe-bank regressions ([verification record](verification/jump_evidence_cpu_20260906.json)). Compilation, CLI preflight failure packaging and changed-document links were checked. These are different test scopes from the earlier 72-test implementation record; neither is production GPU evidence.

These checks were pending when the diagnostic was prepared. The 2026-09-07 user result and decision above now govern execution: no additional replay checks, proceed to common-panel comparison; failed numerical checks remain recorded.

## Current research and implementation boundary

The user has confirmed: fixed x=2.5 m ground start; observed first-landing witnesses; cumulative empirical support; complementary frozen policies as both proposers and evaluators; no requirement for one Actor to realize the whole Tube.

The original audited source was **bfc22f2e32cb78cb269b0e522c3bdd7c6e7a8d42**. The follow-up implementation fixes evidence guards and adds a separate complementary-probe path while retaining legacy protocols. The focused CPU suite passes; production physical equivalence and new training remain unverified. See [implementation guide](JIT_PROBE_BANK_IMPLEMENTATION_20260905.md).

## Status labels

- **Recorded complete:** a committed manifest/report records completion; not a rerun in this review.
- **Locally checked:** source, JSON or isolated function behavior checked here.
- **Incomplete / invalid:** required output missing or comparison not valid.
- **Not established:** new capability or scientific benefit still lacks implementation/controlled evidence.

## Evidence matrix

| Item | Status | Evidence and scope |
| --- | --- | --- |
| Frozen up/down experts | Recorded complete | up: 9,977,856 transitions; down: 25,600; frozen identity manifest exists |
| Handoff bank | Recorded complete | `handoff_bank_9977856_jit8`: 56 snapshots, 1,876 recorded interactions |
| Initial training Tube0 | Recorded complete + locally checked | 222 rows: up 117 (99/18 positive/negative), down 105 (81/24); value-weighted, not all-success |
| Earliest unified formal run | Recorded complete | 10,009,600 transitions in the 2026-08-28 retry; this is not proof it is the current pi_0 |
| Current pi_0 seed identity | Referenced; runtime chain incomplete here | Later Round1 frozen pi_0 appears in causal reports; its full freeze/checkpoint/centerline trace is not all committed |
| Fixed-start causal acquisition | Source + recorded completed rounds | Real env.step paths and captured snapshots; only one proposer per role |
| Wide family landing round | Recorded complete | 1,230/1,258 TRAIN positives; up 714/742, down 516/516 |
| Physical support | Recorded complete | 713 reported new causal TRAIN root cells; control Tube increment is separately +714 root/+897 full |
| Tube3 | Recorded complete | 4,803 rows; +1,159 over 3,644-row Tube2 |
| Role isolation on that round | Recorded PASS | Committed strict audit reports zero exact/near overlap; not a global new-bank isolation result |
| pi_3 training | Recorded complete | 10,009,600 transitions, frozen manifest |
| pi_3 historical policy comparison | **Invalid comparison** | Baseline core stable_recovery; candidate first_valid_landing |
| pi_3 support realization | Recorded diagnostic | 1,130/1,258 source states; 1,061/1,159 increment states; neither proves nor refutes complementary discovery value |
| New baseline endpoint helper | Locally checked | Rejects missing/mismatched landing endpoint; accepts matching first_valid_landing |
| Mixed-endpoint selector quarantine | **Fixed; CPU regression passed** | Analysis and selector reject missing/mixed core/boundary endpoints; historical pi3 report is a refusal fixture |
| Independent-process family shards | Integrity fixes and CPU fixtures passed | Requested-contract checks, per-row validation, staged publishing; real serial-vs-sharded equivalence pending |
| Expanded audit arrivals | Recorded complete | TRAIN 1,754; CALIBRATION 583; ACCEPTANCE 574 |
| Expanded audit family labels | **Incomplete** | CAL/ACCEPT pi_0/pi_1 closed, pi_2 failed; TRAIN pi_0 failed at 1,409/1,754 |
| Predictor | Historical advisory result | Old AUC 0.89249; 6/9 failures accepted (FPR 66.7%); downstream all-positive |
| Forward score locking | Integrity fixes and CPU fixtures passed | Saved self-hash and catalog/protocol checked; new locks bind target family; legacy missing target identity stays explicit |
| Growing multi-probe workflow | **First implementation; CPU fixtures passed** | New bank CLI, multiple proposers/evaluators, observation index, suffix retries/cost; production and cumulative physical registry pending |
| Ancestor branching | **Not implemented as a general replay path** | Current acquisition restarts from fixed ground start |
| Complementary-probe training recipe | **Not locked/validated** | Support choice, exploration allocation, admission and stop conditions still to define |
| Same-budget discovery superiority | **Not established** | No controlled multi-seed discovery comparison or complete cumulative cost ledger |
| Final TEST/JCE/JEL | Not run in this review; keep isolated | Do not confuse with bootstrap files already containing old `test` splits |

## Numbers requiring careful interpretation

On the same 1,258 source TRAIN candidates, reported first-landings are pi_0=1,130, pi_1=1,184, pi_2=1,222, pi_3=1,130; fixed family=1,230. Thus the old family adds only 8 successes over pi_2 on this sample. This motivates a marginal-contribution/cost analysis, not blanket rejection of families. A pi_3 unique-witness count cannot be inferred from aggregate success counts.

The old mixed core reports 3,539 versus 3,598, with 89 changes called improvements and 30 called regressions. Preserve those records, but do not call them a fair policy improvement or use them for a new probe admission decision.

## Historical audit checks at bfc22f2

- 47 JSON files from the latest code/evidence commit parse; 8 changed Python files compile as source.
- Five isolated-function counterexamples reproduce: old mixed gate accepted, changed locked scores accepted, unverified shard cache reused, mixed Actor/endpoint rows merged, and AP changing with tie order.
- Endpoint helper positive/refusal cases behave as intended.
- Tube0 class counts and weight implementation were checked against committed diagnostics and source.
- The three committed forward-score self-hashes independently match (1,038 TRAIN / 342 CALIBRATION / 333 ACCEPTANCE scores). This checks current saved content, not chronology or the missing runtime audit enforcement.
- The original audit had no production runtime and did not rerun the historical `51 passed`. Subsequent implementation installed a temporary CPU stack; see the new verification record below. No production GPU or PPO was run.

See [detailed review](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md) for locations and acceptance criteria.

## Exact next actions

1. The four-policy comparison has completed. Run `run_dense_tube.py --gpu 0` for the bounded 5 cm real-frame pilot and measured label execution choice. No additional snapshot replay validation is requested.
2. Preserve the completed derived old-family labels, original failed attempts, and this comparison cost record; do not repeat the old full comparison.
3. Inspect dense per-policy/union plots and old-versus-new physical occupancy, with unchanged physical grid and explicit benchmark/acquisition/padding/retry costs.
4. Use the new bank path for a fixed-budget pilot with existing frozen policies after compatibility checks. Extend observation indexes to cumulative physical coverage, cross-role isolation and end-to-end costs; retain the accepted numerical-replay limitation.
5. Lock a complementary-probe training recipe and controlled comparison before another large PPO run. Do not wait for full-Tube Actor mastery or predictor performance.

The [training roadmap](JIT_TRAINING_ROADMAP.md) defines phase exits and deliverables. The [historical index](JIT_JUMP_START_TRAJECTORY_INDEX_20260904.md) locates older reports; its old selection wording is not current authority.


## Current implementation verification

The focused CPU suite covers probe banks/supervisor, family/cache, endpoint analysis/selection, predictor, public warm-start, shards and causal contracts. **72 passed** in the final run recorded for this change ([machine-readable record](verification/empirical_envelope_cpu_20260905.json)). Process tests simulate GPU label generation and exercise the real planning, retry-budget and merge code. Public training tests stop before PPO; no production rollout, checkpoint or full training was executed.

New bank acquisition preserves reached context and does not exclude states merely because they occur in training support. Suffix evaluation still explicitly uses fresh-continuation counter resets. `snapshot_replay_equivalence_verified=false` and `formal_envelope_claim_authorized=false` remain in new outputs. A snapshot representation hash cannot close this physical-equivalence gate.

The new index reports per-probe observed/unique contexts, not certified cells. Cumulative physical coverage, cross-bank/role isolation, all acquisition/PPO/bootstrap costs, ancestor branching and complementary training remain unfinished. Detailed commands, scope and next actions: [implementation guide](JIT_PROBE_BANK_IMPLEMENTATION_20260905.md).
