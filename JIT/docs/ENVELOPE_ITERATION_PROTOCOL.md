# JIT fixed-jump-start capability iteration protocol

Version: 2026-09-05. This protocol supersedes natural-start and stable-recovery
requirements in older JIT iteration notes for the active experiment.

## Claim boundary

The method constructs empirical, proposer-conditioned arrival and landing
evidence for one fixed robot task. It does not estimate a formal reachable set,
viability kernel, invariant set, certified safe set or complete physical limit.
The final deployable object is one unified Actor.

## Locked identities

Before acquisition, lock:

- task XML/physics/control identity;
- jump-start snapshot at `x = 2.5 m`;
- real-frame π0 centerline and 0.1 m slice targets;
- π0 proposal Actor and action perturbation family;
- frozen π0/π1/π2 evaluator family;
- TRAIN/CALIBRATION/ACCEPTANCE role assignment;
- first-valid-landing endpoint and horizon;
- physical cell resolution;
- candidate and labeling seeds;
- acquisition and total-interaction ceilings;
- optional predictor model, normalization and threshold;
- policy selection endpoint, panel, margins and stopping rule.

An artifact hash locks content identity but does not by itself prove creation
time. Preserve the pre-training/pre-outcome protocol artifact and its repository
history.

## Stage 1: centerline

Use the locked π0 trajectory. Keep only real simulator frames from the jump
start through first valid landing or `x = 4.2 m`. Do not interpolate qpos/qvel.
The centerline defines exploration slices only; it does not define desired
actions, rewards or continuous reachability.

## Stage 2: causal forward acquisition

For each declared slice and role family:

1. restore the one locked jump-start snapshot;
2. execute the deterministic frozen π0 prefix;
3. begin a bounded predeclared action perturbation inside the lookback window;
4. advance only through authoritative `env.step`;
5. capture the exact state that physically enters the target slice;
6. store qpos/qvel, Actor FIFO, last action, control and phase event context.

Required arrival flags include:

```text
jump_start_connected = true
generated_by_env_step_only = true
rsi_used_to_establish_reachability = false
qpos_qvel_injection_used = false
proposal_anchor_used_as_reset = false
```

`natural_start_connected` is false for the active experiment and must not be
silently promoted to true.

## Stage 3: optional pre-outcome predictor lock

If auditing the advisory predictor, load only the frozen candidate catalog and
snapshot Actor observations. Before reading any new evaluator outcomes, write:

- exact candidate ordering and state identity;
- model/normalization/threshold identity;
- score and threshold decision per upstream candidate;
- an explicit statement that outcome labels were not read;
- no-score/no-model status for unsupported single-class phases.

All candidates still receive real evaluator rollouts. The predictor does not
filter arrival evidence, create positive labels or admit Tube rows.

## Stage 4: policy-family first-landing labels

For each exact candidate, independently restore its full snapshot under frozen
π0, π1 and π2. Each evaluator starts with the same candidate physics and context
and stops at first valid landing, physical failure or the declared horizon.

```text
family_positive(z) = landing(pi_0,z) OR landing(pi_1,z) OR landing(pi_2,z)
```

Post-landing recovery is not required. Preserve each member's binary label,
outcome class, interactions and Actor/payload identity. The OR merge must verify
candidate order and identity row by row.

### Memory-bounded execution

Large evaluators run as independent contiguous candidate shards. Each shard is a
fresh process so CUDA/Warp/JAX allocations are released. Sharding changes only
process lifetime and must preserve:

- original catalog and global candidate index;
- global candidate-index PRNG folding;
- policy, seed, horizon and endpoint;
- complete non-overlapping coverage;
- exact catalog order after merge.

Current maximum is 600 candidates per evaluator process. Run different
evaluators serially on one GPU unless a measured resource budget authorizes
concurrency. Preserve failed partial attempts; never relabel them as complete.

The canonical CLI supports one evaluator shard and strict shard merge through
`JIT/cli/label_policy_family_first_landing.py` using `--shard-index`,
`--shard-count` or repeated `--merge-shard-dir`.

Before relying on a new shard path, compare serial and sharded labels on the same
small catalog, checking every candidate index, state identity, seed and label.

## Stage 5: logical roles and isolation

- TRAIN rows may fit models and supply observed positives to replay.
- CALIBRATION selects thresholds only.
- ACCEPTANCE supports locked development comparisons only.
- final TEST/JCE/JEL remains untouched.

Remove exact cross-role states and holdout states already present in the target
training Tube using outcome-blind identity rules. Preserve raw roles and write
derived views with excluded counts/reasons. Require exact and declared-near
TRAIN/holdout isolation. Parent IDs are not sufficient evidence of physical
independence; report proposal groups and trajectory correlation.

## Stage 6: capability and Tube accounting

Report separately:

1. raw family-positive candidates before deduplication;
2. duplicates against the source Tube;
3. raw Tube increment and total rows;
4. new causal TRAIN root cells;
5. all-state control root/full cells;
6. semantic jump-corridor cells with recovery excluded.

Only observed TRAIN positives can expand replay. CALIBRATION and ACCEPTANCE never
enter TRAIN support. Historical core rows retain their historical provenance and
must not be rewritten as newly causal.

## Stage 7: predictor audit

After fresh family labels close, join them to the locked scores by exact
candidate and state identity. Report at minimum:

- positive/negative candidates and independent proposal groups;
- ROC-AUC and PR-AUC when both classes exist;
- recall and false-positive rate at the locked threshold;
- accepted negative count;
- probability calibration diagnostics;
- group-aware bootstrap interval when supporting a scientific claim.

Do not refit before producing the locked forward audit. A predictor/no-predictor
same-budget experiment is required before claiming sample-efficiency benefit.

## Stage 8: policy training

Training requires all previous stages to pass and a predeclared comparison.
Actor-only warm start imports Actor and observation normalizer; critic and
optimizer are fresh. Record exact PPO transitions and all environment
interactions outside PPO.

No candidate policy is authorized solely by positive Tube growth. For the next
main experiment, use at least three independent pilot seeds and preferably five
main-result seeds.

## Stage 9: endpoint-identical evaluation

Before candidate training, lock one common panel and one success endpoint for
both baseline and candidate. The lock must explicitly contain:

```text
core_success_criterion = first_valid_landing
boundary_success_criterion = first_valid_landing
same horizon and remaining-time convention
same state identities and role provenance
selection margins and stopping rule
```

Never compare a stable-recovery baseline count to a first-landing candidate
count as policy improvement. If a mismatch is discovered after training,
preserve the artifact and mark any corrected rerun retrospective.

Report:

- common core success and coverage;
- old-success regressions and old-failure improvements;
- upstream/downstream results;
- new-boundary success and independent groups;
- TRAIN-support realization;
- forward-task performance on the declared distribution;
- total interaction cost.

Capability evidence and one-Actor realization remain separate decisions.

## Stage 10: selection or stop

A prospective selection may occur only under the pre-training endpoint-identical
contract. ACCEPTANCE is then consumed as development data. If the candidate does
not improve the declared primary outcome at controlled cost, stop or change the
method under a new protocol; do not hide the result with family OR or Tube size.

No π4 training is currently authorized.

## Required controlled comparisons

- continued PPO/fixed curriculum;
- static successful Tube-RSI;
- uniform fixed-grid forward acquisition;
- RSI-only candidates with matched label/training budget;
- reachable-filtered iterative TRAIN replay;
- predictor removed if predictor-guided allocation is enabled.

## Cost accounting

Count proposal-prefix steps, excluded candidates, all evaluator interactions,
failed attempts/retries, PPO transitions and development evaluation. Report
shared bootstrap cost separately and also include it in an end-to-end total.
Wall time and hardware are additional operational measurements, not substitutes
for environment interactions.

## Current resumption point

The expanded catalogs and pre-outcome scores are complete. Resume at Stage 4
with independent evaluator shards. Do not reacquire candidates, change seeds,
refit the predictor or start π4.
