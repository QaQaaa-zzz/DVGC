# JIT — empirical jumping-envelope discovery

The user-confirmed objective is to discover cumulative, forward-arrived and landing-witnessed support at controlled cost, using complementary frozen policies. The final single-Actor application is a separate diagnostic/goal; it is not required to cover all accumulated support.

## Method and implementation status

Phase up/down data and value-weighted initial support bootstrap unified seed development. A fixed real pi_0 centerline organizes forward exploration from the declared x=2.5 ground state. The target new workflow allows multiple proposers and evaluators, preserves exact witnesses, and trains new probes under a predeclared complementary-discovery recipe.

At reviewed `bfc22f2`, multi-probe lifecycle and bank scheduling are not complete. The existing family runner enforces pi_0/pi_1/pi_2 and the old workflow still selects one coverage-eligible successor. Do not use that DAG as the new method without migration. The [review](docs/JIT_EMPIRICAL_ENVELOPE_REVIEW_20260905.md) documents reproducible integrity bugs and source gaps.

## Read before execution

- [Root authority](../AGENTS.md), [JIT instructions](AGENTS.md), and [project](../PROJECT.md).
- [Current status](docs/CURRENT_STATUS.md) and [paper outline](docs/JIT_PAPER_OUTLINE.md).
- [Protocol](docs/ENVELOPE_ITERATION_PROTOCOL.md), [training roadmap](docs/JIT_TRAINING_ROADMAP.md), [handoff](docs/CODEX_HANDOFF_20260904.md).
- [Code ownership](docs/CODE_ORGANIZATION.md).

## Validation on the production host

Production interpreter: `/home/qy/mujoco_playground/.venv/bin/python`. From the repository root, after implementing fixes:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_policy_family_landing.py JIT/tests/test_family_landing_predictor.py JIT/tests/test_capability_progression.py JIT/tests/test_causal_jump_reachability_contract.py JIT/tests/test_unified_continuation_shards.py -q
```

Existing tests need new refusal cases from the review; their prior passing results do not establish those missing contracts. A real small-catalog serial/sharded comparison and checkpoint/prefix/suffix restoration smoke are separate required runtime checks. This documentation review did not run them.

## Maintenance and evidence

Keep source under `src/jit_dvgc`, thin entry points under `cli`, tests under `tests`, and guidance under `docs`. Preserve raw run identities, partial GPU failures and bootstrap history. Current old expanded scans must close under their original proposer/family/seed/endpoint; start new multi-probe experiments with new identities.

TRAIN may guide learning/discovery. Development holdouts cannot enter TRAIN. Final TEST/JCE/JEL remains isolated; historical bootstrap splits named test are already-used evidence and are not the untouched final distribution.
