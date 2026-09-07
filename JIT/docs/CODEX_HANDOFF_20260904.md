# DVGC/JIT handoff — updated for empirical envelope, 2026-09-06

Latest handoff (2026-09-07): the user has returned the GPU diagnostic and accepted the numerical trajectory differences and near-ground reset. Do not ask for more replay validation. The next executable task is [four-policy envelope comparison](JIT_POLICY_ENVELOPE_COMPARISON_20260907.md); return its ZIP. The previous diagnostic entry below is historical, not a remaining mandatory gate.

## Historical 2026-09-06 handoff: prepared diagnostic

The [GPU evidence validation entry](JIT_GPU_EVIDENCE_VALIDATION_20260906.md) is implemented. Ask for its `results_to_send.zip` after the user runs the documented command; analyze per-gate differences and logs before scheduling training. Default maximum is 17,200 env.step calls, six real reached states, one frozen Round1 pi_0. Missing assets and runtime errors produce a diagnostic bundle, not negative physics labels. This change passed 59 focused CPU tests ([record](verification/jump_evidence_cpu_20260906.json)); real GPU/checkpoint results remain pending. Continue preserving the original locked pi_0/pi_1/pi_2 scan and unverified bank flags.

## Start here

Read [root AGENTS](../../AGENTS.md), [PROJECT](../../PROJECT.md), [CURRENT_STATUS](CURRENT_STATUS.md), [paper outline](JIT_PAPER_OUTLINE.md), [review](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md), [protocol](ENVELOPE_ITERATION_PROTOCOL.md), and [training roadmap](JIT_TRAINING_ROADMAP.md).

The user has selected fixed x=2.5 ground start, one successful landing witness for admission, cumulative empirical support, and a growing frozen bank used for both proposals and suffix evaluations. Single-Actor full-Tube mastery is not a paper or iteration requirement.

## What has actually changed

Correctness fixes and a first versioned bank path now implement part of the new direction. Read [implementation status](JIT_PROBE_BANK_IMPLEMENTATION_20260905.md) for commands and verification boundaries. Preserve historical reports and locked experiments; new outputs do not claim GPU replay equivalence.

## Preserve and reuse

- Frozen up/down identities, handoff snapshots and value models.
- Initial 222-row value-weighted training Tube (includes 42 negative phase labels).
- Unified development lineage; the current causal seed references later Round1 pi_0, not merely the first completed unified run.
- Fixed real-frame centerline, exact snapshot/continuation primitives, causal acquisition and physical metric modules.
- Old pi_0/pi_1/pi_2 family records and completed evaluator caches after identity validation.
- Trained pi_3 and its valid raw outcomes, while excluding the mixed-endpoint comparison from fair eligibility evidence.

## First executable work

1. Preserve the tested identity/endpoint/cache/predictor/warm-start guards and the accepted numerical-replay limitation; do not mark exact equivalence verified.
2. Run `compare_policy_envelopes.py --gpu 0` on the production host with the existing manifests/checkpoints/trace/catalogs. Do not request further replay validation.
3. Complete the old expanded round's missing labels with its original proposer/family/seed/horizon/endpoint. Keep failed attempts and total cost.
4. Use the new bank/observation/suffix-cost path; extend cumulative physical cells, end-to-end costs and cross-role/version isolation before controlled claims.
5. Run existing-policy discovery comparisons. Evaluate pi_3 complementary contribution under a new valid contract rather than the old selected manifest.
6. Only then freeze and run a new complementary-probe PPO recipe.

A predictor is optional. Do not block discovery on its class balance or on full-Tube Actor retention.

## Existing expanded round

Root: `JIT/runs/iteration_auto/pi_3_to_pi_4_pi0_centerline_family_landing_predictor_audit_20260905`.

Recorded arrivals: TRAIN 1,754; CALIBRATION 583; ACCEPTANCE 574. CAL/ACCEPT pi_0/pi_1 completed, pi_2 failed. TRAIN pi_0 failed at 1,409/1,754. Scores are saved and self-hash/catalog checks are repaired; old locks without target-family identity must be explicitly marked. Directory naming is not training authorization or bank membership.

Use `JIT/cli/label_policy_family_first_landing.py` for existing evaluator shard/merge primitives only after the integrity repairs and small-bank equivalence check. Historical recommendation: at most 600 candidates per fresh GPU process; actual safe capacity must be measured. Preserve complete outputs and partial attempts.

## Remaining implementation traps

- The legacy family remains exactly pi_0/pi_1/pi_2; new bank runs use the separate entry point.
- The old selected-policy DAG still enforces Actor coverage. Do not use it as the new discovery scheduler.
- New snapshot hashes preserve stored FIFO/event/clock identity, but suffix evaluation still resets administrative counters. Actual replay equivalence is unverified.
- Cumulative bank-version physical coverage, global role isolation and bootstrap/acquisition/PPO/retry accounting are incomplete.
- `natural10` means historical natural reset, not automatically fixed jump start. New complementary training/reset recipes are not implemented.
- A self-hash is not proof of prospective chronology. Preserve original pre-outcome/pre-training artifacts.

## Verification boundary

The original audit checked 47 JSON and 8 Python files, isolated defects and bootstrap counts. The follow-up focused CPU suite now passes 72 tests, including simulated-process planning/merging and public warm-start routing. The production runtime is unavailable; no real GPU, checkpoint smoke or PPO occurred. Final TEST/JCE/JEL stays unopened. Bootstrap files already containing old `test` outcomes are separate historical evidence.
