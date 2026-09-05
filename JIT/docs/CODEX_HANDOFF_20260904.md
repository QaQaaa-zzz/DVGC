# DVGC/JIT handoff — updated for empirical envelope, 2026-09-05

## Start here

Read [root AGENTS](../../AGENTS.md), [PROJECT](../../PROJECT.md), [CURRENT_STATUS](CURRENT_STATUS.md), [paper outline](JIT_PAPER_OUTLINE.md), [review](JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md), [protocol](ENVELOPE_ITERATION_PROTOCOL.md), and [training roadmap](JIT_TRAINING_ROADMAP.md).

The user has selected fixed x=2.5 ground start, one successful landing witness for admission, cumulative empirical support, and a growing frozen bank used for both proposals and suffix evaluations. Single-Actor full-Tube mastery is not a paper or iteration requirement.

## What has actually changed

This handoff replaces the previous single-successor objective at documentation level. Reviewed remote code remains `bfc22f2`; no runtime implementation is fixed by changing these instructions. Preserve historical reports and pending locked experiments.

## Preserve and reuse

- Frozen up/down identities, handoff snapshots and value models.
- Initial 222-row value-weighted training Tube (includes 42 negative phase labels).
- Unified development lineage; the current causal seed references later Round1 pi_0, not merely the first completed unified run.
- Fixed real-frame centerline, exact snapshot/continuation primitives, causal acquisition and physical metric modules.
- Old pi_0/pi_1/pi_2 family records and completed evaluator caches after identity validation.
- Trained pi_3 and its valid raw outcomes, while excluding the mixed-endpoint comparison from fair eligibility evidence.

## First executable work

1. Fix F01/F02/F04 and the context contracts relevant to the next path. Fix F03/F10 before predictor audit and F07 before public warm-start training.
2. On the production host, materialize missing manifests/checkpoints/trace/catalogs and validate small serial-versus-sharded and prefix/suffix replay.
3. Complete the old expanded round's missing labels with its original proposer/family/seed/horizon/endpoint. Keep failed attempts and total cost.
4. Introduce versioned bank/witness/cost manifests and multiple per-proposer catalogs, preserving legacy protocol readers.
5. Run existing-policy discovery comparisons. Evaluate pi_3 complementary contribution under a new valid contract rather than the old selected manifest.
6. Only then freeze and run a new complementary-probe PPO recipe.

A predictor is optional. Do not block discovery on its class balance or on full-Tube Actor retention.

## Existing expanded round

Root: `JIT/runs/iteration_auto/pi_3_to_pi_4_pi0_centerline_family_landing_predictor_audit_20260905`.

Recorded arrivals: TRAIN 1,754; CALIBRATION 583; ACCEPTANCE 574. CAL/ACCEPT pi_0/pi_1 completed, pi_2 failed. TRAIN pi_0 failed at 1,409/1,754. Scores are saved, but audit identity checks need repair. Directory naming is not training authorization or bank membership.

Use `JIT/cli/label_policy_family_first_landing.py` for existing evaluator shard/merge primitives only after the integrity repairs and small-bank equivalence check. Historical recommendation: at most 600 candidates per fresh GPU process; actual safe capacity must be measured. Preserve complete outputs and partial attempts.

## Known implementation traps

- Hard-coded family exactly pi_0/pi_1/pi_2.
- Single selected-policy DAG still requires Actor coverage eligibility.
- Shard cache returns before validating inputs; some merge checks run after publication.
- Family OR does not verify every row's Actor/payload/endpoint.
- Existing training Tube states are excluded from new arrival evidence.
- Physical state hashes do not include full controller/time context.
- Public warm-start routing still relies on historical CLI monkey-patching.
- `natural10` training means existing phase-u natural reset, not automatically fixed jump start.
- Predictor audit ignores scores self-hash; AP mishandles tied values.

See the review for precise source links and acceptance criteria.

## Verification boundary

This review checked 47 JSON and 8 Python files from the latest commit, isolated defect reproductions, and bootstrap counts. The production runtime is not available in the review workspace; no GPU run or full 51-test rerun occurred. Final TEST/JCE/JEL stays unopened. Bootstrap files already containing old `test` outcomes are separate historical evidence.
