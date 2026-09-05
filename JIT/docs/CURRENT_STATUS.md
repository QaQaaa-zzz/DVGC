# Current JIT status — empirical-envelope review, 2026-09-05

## Current research and implementation boundary

The user has confirmed: fixed x=2.5 m ground start; observed first-landing witnesses; cumulative empirical support; complementary frozen policies as both proposers and evaluators; no requirement for one Actor to realize the whole Tube.

The latest reviewed remote source is **bfc22f2e32cb78cb269b0e522c3bdd7c6e7a8d42**. That code still implements the historical fixed-family, single-successor workflow. This documentation revision aligns direction and records blockers; it does not claim those source defects are fixed or that new training has run.

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
| Full selector quarantine | **Known defect reproduced** | Old mixed-endpoint gate still yields authority/prospective flags true |
| Independent-process family shards | Implemented interface; validation incomplete | No real serial-vs-sharded equivalence rerun here; cache/row merge gaps remain |
| Expanded audit arrivals | Recorded complete | TRAIN 1,754; CALIBRATION 583; ACCEPTANCE 574 |
| Expanded audit family labels | **Incomplete** | CAL/ACCEPT pi_0/pi_1 closed, pi_2 failed; TRAIN pi_0 failed at 1,409/1,754 |
| Predictor | Historical advisory result | Old AUC 0.89249; 6/9 failures accepted (FPR 66.7%); downstream all-positive |
| Forward score locking | Files present; implementation integrity incomplete | Audit does not verify saved scores_sha256; full target-bank binding absent |
| Growing multi-probe workflow | **Not implemented as a validated loop** | Three-member family assertion, one selected authority, no full bank/witness lifecycle |
| Ancestor branching | **Not implemented as a general replay path** | Current acquisition restarts from fixed ground start |
| Complementary-probe training recipe | **Not locked/validated** | Support choice, exploration allocation, admission and stop conditions still to define |
| Same-budget discovery superiority | **Not established** | No controlled multi-seed discovery comparison or complete cumulative cost ledger |
| Final TEST/JCE/JEL | Not run in this review; keep isolated | Do not confuse with bootstrap files already containing old `test` splits |

## Numbers requiring careful interpretation

On the same 1,258 source TRAIN candidates, reported first-landings are pi_0=1,130, pi_1=1,184, pi_2=1,222, pi_3=1,130; fixed family=1,230. Thus the old family adds only 8 successes over pi_2 on this sample. This motivates a marginal-contribution/cost analysis, not blanket rejection of families. A pi_3 unique-witness count cannot be inferred from aggregate success counts.

The old mixed core reports 3,539 versus 3,598, with 89 changes called improvements and 30 called regressions. Preserve those records, but do not call them a fair policy improvement or use them for a new probe admission decision.

## What was actually checked in this review

- 47 JSON files from the latest code/evidence commit parse; 8 changed Python files compile as source.
- Five isolated-function counterexamples reproduce: old mixed gate accepted, changed locked scores accepted, unverified shard cache reused, mixed Actor/endpoint rows merged, and AP changing with tie order.
- Endpoint helper positive/refusal cases behave as intended.
- Tube0 class counts and weight implementation were checked against committed diagnostics and source.
- The three committed forward-score self-hashes independently match (1,038 TRAIN / 342 CALIBRATION / 333 ACCEPTANCE scores). This checks current saved content, not chronology or the missing runtime audit enforcement.
- Production Python, JAX, MuJoCo, Brax and pytest are absent here. No full test-suite, simulator, GPU equivalence, checkpoint restoration or new training was run. The prior report's `51 passed` is historical, not a new test result.

See [detailed review](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md) for locations and acceptance criteria.

## Exact next actions

1. Repair the correctness blockers before relying on label merges or automated decisions.
2. Close the old expanded audit under its already locked pi_0 / pi_0-pi_1-pi_2 protocol using independently bounded evaluator processes. Preserve attempts and count their costs.
3. Materialize the seed/probe chain, complete start and centerline, and a small replayable catalog; validate serial/sharded and prefix/suffix equality.
4. Implement the new bank/registry/coverage-cost contract and run a fixed-budget pilot with existing frozen policies, including pi_3 only after compatibility and fresh witness checks.
5. Lock a complementary-probe training recipe and controlled comparison before another large PPO run. Do not wait for full-Tube Actor mastery or predictor performance.

The [training roadmap](JIT_TRAINING_ROADMAP.md) defines phase exits and deliverables. The [historical index](JIT_JUMP_START_TRAJECTORY_INDEX_20260904.md) locates older reports; its old selection wording is not current authority.
