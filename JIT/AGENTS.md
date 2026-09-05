# JIT agent instructions — active empirical-envelope direction

The root [AGENTS.md](../AGENTS.md) and user-confirmed [paper outline](docs/JIT_PAPER_OUTLINE.md) govern this work. Audited baseline: `bfc22f2`. The first implementation update is documented in [implementation status](docs/JIT_PROBE_BANK_IMPLEMENTATION_20260905.md); remaining requirements are not runtime success claims.

## Objective and objects

Bootstrap with up/down phase support and a successful unified seed, then discover empirical jumping support using complementary frozen probes. Do not require one Actor to retain/realize the entire cumulative Tube.

Keep forward arrivals, exact landing witnesses, empirical support, projected cells, training Tube and individual-Actor realization distinct. Historical Tube0 is value-weighted (222 rows, including 42 negative phase labels), not an all-positive capability set. Original unified training completion and later Round1 pi_0 identity are distinct records.

## Arrival and continuation

- Begin at the complete declared ground jump-start state, x=2.5 m. Do not demand or claim the earlier natural approach.
- Keep pi_0 centerline coordinates fixed; all centerline frames are captured, not interpolated.
- New protocols may use multiple frozen proposers and evaluators. Lock membership before each round and record bank versions.
- Each candidate needs exact prefix/suffix context, action history, events, start/remaining time and source identities. qpos/qvel identity alone is insufficient for policy-conditioned continuation.
- RSI may restore a verified ancestor or candidate, but cannot invent a prefix. Current source only restarts from the fixed start; general ancestor reuse needs implementation and replay checks.
- A first-valid-landing witness is sufficient. Negative means no success under the declared evaluators/horizon, not physical infeasibility.
- Do not discard evidence just because physical coordinates already occur in the training Tube. Deduplicate coverage separately.
- Keep incomplete/error/untested outcomes separate from completed failed rollouts.

## Legacy execution versus new method

The already locked wide/expanded scans retain pi_0 proposer and pi_0/pi_1/pi_2 evaluators. Resume under their exact catalog, seed, horizon and endpoint. Do not rename or retrofit those runs as multi-probe discovery.

Use `cli/probe_bank.py` for new bank pilots after production smoke. It emits observations with replay/physical-envelope claims explicitly unverified; it does not authorize automatic PPO or certify physical cells. The old `prepare_iterative_envelope_workflow.py` selects one successor and requires coverage eligibility. It is legacy automation, not the new discovery scheduler. Do not launch a new experiment through that chain without changing and validating its contracts.

pi_3 remains a frozen trained candidate. Its mixed-endpoint gate is quarantined as comparison evidence, not a blanket rejection of every pi_3 witness. Assess technical probe eligibility and complementary witnesses under a new protocol. Do not automatically promote the old selected manifest.

## Immediate correctness gates

See issue IDs and evidence in [review](docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md).

1. Preserve the implemented evaluator/cache identity checks, per-row endpoint checks and staged merge publishing; validate them against production artifacts.
2. Preserve endpoint validation in both legacy analysis and selector; mixed/missing endpoint evidence now refuses.
3. Separate training-support membership, exact witness identity and physical novelty.
4. Validate public warm-start routing; preserve Actor/normalizer only, with fresh critic/optimizer unless explicitly declared.
5. Validate serial versus sharded labels on a small identical catalog using the production runtime.
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
