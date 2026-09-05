# DVGC/JIT technical handoff — updated 2026-09-05

## Read this first

Current authority is `AGENTS.md`, `JIT/AGENTS.md` and
`JIT/docs/CURRENT_STATUS.md`. The 2026-09-04 causal redesign report is a
historical design record and contains superseded natural-start statements.

## Current method

The active experiment begins from one locked physical jump-start snapshot at
`x = 2.5 m`. Frozen π0 provides the real-frame centerline and proposal prefix.
Bounded action perturbations generate exact candidates only through `env.step`.
Frozen π0, π1 and π2 independently continue from each exact candidate; the
family label is positive if any member reaches first valid landing before
physical failure. Recovery is outside the label.

Keep five objects separate: forward arrival `A`, family witness `E`, physical
cell occupancy `J`, raw/control Tube `S`, and single-Actor realization `r`.
Family OR does not prove a one-Actor full rollout. No formal reachability,
viability or safety claim is active.

## Completed evidence

First family-round root:

`JIT/runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904`

Key results:

```text
TRAIN candidates / family positives        1258 / 1230
upstream / downstream positives             714 / 742, 516 / 516
Tube2 -> Tube3 raw increment                 1159
Tube3 total rows                             4803
new causal TRAIN root cells                  713
control root/full increment                  714 / 897
```

π3 training completed at 10,009,600 transitions with π2 Actor/normalizer warm
start and fresh critic/optimizer. It is a valid trained checkpoint.

## π3 selection caveat

The stored π3 gate is not a fair prospective baseline comparison. Its own
summary identifies the source core as:

```text
baseline:  stable_recovery
candidate: first_valid_landing
```

The stored 3,539→3,598 result, 89 improvements, 30 regressions and 4/12 boundary
gains remain immutable diagnostics. The historical selected manifest remains an
engineering record, but π3 is not current scientific authority for starting
π4. Any same-first-landing reevaluation now is retrospective.

On the 1,258 source TRAIN states, first-landing successes were π0 1,130, π1
1,184, π2 1,222 and π3 1,130. On the 1,159-row Tube3 increment, π3 reached
1,061 first landings. Do not describe this as final task performance.

## Predictor

Implementation:

- `JIT/src/jit_dvgc/family_landing_predictor.py`
- `JIT/cli/fit_family_landing_predictor.py`

The upstream predictor's old development ACCEPTANCE ROC-AUC is 0.89249, with
0.98611 positive recall but 6/9 negatives above the locked threshold.
Downstream was all-positive and not fit. It is advisory only and cannot create
labels or Tube entries.

## Active incomplete round

Root:

`JIT/runs/iteration_auto/pi_3_to_pi_4_pi0_centerline_family_landing_predictor_audit_20260905`

The directory name is identity only and does not authorize π4.

Completed acquisition:

```text
TRAIN         1754 = 1038 upstream + 716 downstream
CALIBRATION    583 =  342 upstream + 241 downstream
ACCEPTANCE     574 =  333 upstream + 241 downstream
```

Pre-outcome upstream scores are locked for all roles. They were computed from
catalog snapshots without loading outcome labels.

Incomplete labels:

- CALIBRATION and ACCEPTANCE completed π0 and π1 serial results;
- both π2 attempts failed after CUDA allocation errors;
- TRAIN π0 failed after 1,409/1,754 candidates;
- failure summaries and incomplete directories are preserved.

## Memory-bounded repair

`jit_dvgc.unified_continuation_shards` now supports a distinct acquisition
policy, first-valid-landing endpoint and evaluator identity. The canonical
family CLI supports evaluator shards and strict merges:

```bash
python JIT/cli/label_policy_family_first_landing.py \
  --catalog <role>/acquisition/catalog.json \
  --acquisition-frozen-policy <pi0-frozen.json> \
  --evaluator-frozen-policy <one-evaluator-frozen.json> \
  --output-dir <role>/labels/evaluator_shards/<pi>/shard_000_of_003 \
  --shard-index 0 --shard-count 3 \
  --max-ticks 400 --protocol-seed <locked-role-seed>
```

Repeat in separate processes for every shard, then merge with repeated
`--merge-shard-dir` arguments into `<role>/labels/per_policy/<pi>`. Limit each
process to at most 600 candidates. Run evaluators serially on one GPU. After all
three canonical per-policy outputs exist, rerun the ordinary role command; it
reuses completed evaluators and creates the family OR/logical role artifacts.

Before large-scale reliance, compare serial and sharded output row-by-row on a
small bank.

## Exact next work

1. Run and merge missing evaluator shards without changing catalog, role seed,
   horizon or first-landing criterion.
2. Recreate complete family labels and role manifests.
3. Join locked scores to fresh upstream labels and report ROC-AUC, PR-AUC,
   recall, FPR, accepted negatives and group-aware intervals.
4. Produce a retrospective same-first-landing π2/π3 diagnostic.
5. Freeze the controlled experiment matrix, total interaction budget and stop
   rule.
6. Do not train π4 until that prospective contract is accepted.

## Paper direction

Working hypothesis: reachability-filtered reset curricula improve a single
unified jumping policy at controlled cost. Required pilot comparisons are
continued PPO/fixed curriculum, static successful Tube-RSI, uniform forward
scan and the iterative method; add RSI-only and predictor ablations as needed.
Use three independent pilot training seeds and preferably five main-result
seeds. Final TEST/JCE/JEL remains untouched.

## Repository safety

- work on `agent/two-phase-soft-tube`;
- use `/home/qy/mujoco_playground/.venv/bin/python`;
- preserve unrelated files and history;
- never reset, clean, stash, rebase or force-push;
- do not modify historical run evidence to repair its interpretation;
- keep logic in `JIT/src/jit_dvgc`, CLIs thin and tests in `JIT/tests`;
- do not repeatedly compute identities already locked by manifests.
