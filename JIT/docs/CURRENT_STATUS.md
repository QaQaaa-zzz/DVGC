# Current JIT status — 2026-09-05

## One-line status

The first fixed-jump-start family-landing round produced Tube3 and trained π3,
but π3's stored selection comparison mixed success endpoints and is quarantined
as historical engineering evidence. The next expanded catalogs and predictor
scores are locked; family labels are incomplete after GPU allocation failures.
No π4 training is authorized.

## Current scientific definition

The active experiment is conditional on the locked `x = 2.5 m` jump start. It
does not currently begin at the natural episode reset.

```text
A = exact states reached by pi_0 + bounded causal action perturbation + env.step
E = exact A states where any frozen pi_0/pi_1/pi_2 evaluator reaches first landing
J = positive A intersect E states projected to fixed physical-resolution cells
S = raw reset/replay Tube, including retained historical rows
r = one frozen Actor's success on a common locked panel
```

These objects answer different questions. A family witness may use π0 for the
prefix and π1 or π2 for the suffix, so it is not a one-Actor rollout. Tube rows
and physical cells are not formal reachable/safe sets. The final target remains
one unified Actor.

## Completed fixed-jump-start family round

Root:

`JIT/runs/iteration_auto/pi_2_to_pi_3_pi0_centerline_family_landing_wide_20260904`

The locked π0 real-frame centerline was sampled with π0 proposals. Every reached
candidate was evaluated by frozen π0, π1 and π2. Positive means at least one
member reached the first valid landing before physical failure; recovery was not
required.

| Quantity | Result |
|---|---:|
| TRAIN candidates | 1,258 |
| family-positive TRAIN | 1,230 |
| family-negative TRAIN | 28 |
| upstream positive | 714 / 742 |
| downstream positive | 516 / 516 |
| Tube2 source rows | 3,644 |
| positive rows before Tube dedup | 1,230 |
| duplicates against Tube2 | 71 |
| Tube3 raw increment | 1,159 |
| Tube3 total rows | 4,803 |
| new causal TRAIN root cells | 713 |
| control root-cell increment | 714 |
| control full-physical increment | 897 |

The strict role-isolation audit passed after outcome-blindly removing holdout
states already present in the target training Tube. The derived CALIBRATION and
ACCEPTANCE views record their exclusions and have zero exact or near TRAIN
overlap under the declared tolerance.

These counts are not interchangeable: raw increment counts rows; causal root
cells deduplicate positive TRAIN evidence; control cells project all Tube rows;
semantic corridor cells additionally exclude post-landing recovery.

## π3 training and the selection quarantine

π3 was Actor-only warm-started from π2 and trained for exactly 10,009,600 PPO
transitions. Actor and observation normalizer were imported; critic and optimizer
were fresh. Training completed and the checkpoint was frozen.

Historical stored diagnostic:

```text
source panel states                 3644
stored baseline successes           3539
stored pi_3 successes               3598
old failures improved                 89
old successes regressed               30
boundary gains                       4 / 12 across 3 groups
downstream                         2895 / 2895 for both
```

This diagnostic cannot support a fair prospective policy selection. The gate's
own `core_source` records:

```text
baseline_success_criterion  = stable_recovery
candidate_success_criterion = first_valid_landing
```

The historical `selected_policy.json` and capability decision remain immutable
records of the engineering decision made at the time. They are no longer active
scientific authority. The 30 regressions and 89 improvements remain useful, but
a same-endpoint rerun is retrospective because π3 has already been trained and
observed. A new prospective selection requires an endpoint-identical baseline
lock made before training.

On the 1,258 source TRAIN candidates, stored first-landing realization was:

```text
pi_0  1130
pi_1  1184
pi_2  1222
pi_3  1130
```

On the actual 1,159-row Tube3 increment, π3 realized 1,061 first landings
(91.54%). This is curriculum-support realization, not final forward-task
generalization. It shows that Tube growth did not automatically transfer to
π3 and motivates controlled forgetting/core-retention studies.

## Advisory landing predictor

An upstream 64x64 tanh predictor was fit from observed family-landing labels.
The downstream phase remained all-positive and was not fit.

Old held-out development result:

| Split/metric | Result |
|---|---:|
| TRAIN upstream | 714 positive / 28 negative |
| CALIBRATION upstream | 227 positive / 4 negative |
| ACCEPTANCE upstream | 216 positive / 9 negative |
| ACCEPTANCE ROC-AUC | 0.89249 |
| positive recall at locked threshold | 0.98611 |
| accepted negatives at locked threshold | 6 / 9 |

High recall on a strongly positive-skewed set does not imply reliable failure
detection. The predictor is advisory only: it cannot establish arrival, create
labels, admit Tube rows or claim safety. Future reports must add PR-AUC, FPR,
group counts and group-aware uncertainty. A same-budget predictor/no-predictor
acquisition comparison is still missing.

## Expanded forward predictor-audit round

Root:

`JIT/runs/iteration_auto/pi_3_to_pi_4_pi0_centerline_family_landing_predictor_audit_20260905`

This name is historical directory identity; it does not authorize π4 training.
The frozen experiment retains π0 as proposer and π0/π1/π2 as evaluator family.
The declared scan expanded lookbacks to 0.1–0.7 m with strengths
0.025/0.05/0.10/0.15/0.20.

Acquisition completed:

| Role | Candidates | Upstream | Downstream | predicted upstream above threshold |
|---|---:|---:|---:|---:|
| TRAIN | 1,754 | 1,038 | 716 | 975 |
| CALIBRATION | 583 | 342 | 241 | 326 |
| ACCEPTANCE | 574 | 333 | 241 | 317 |

Model, normalization, threshold, catalog order and per-candidate scores were
locked before outcome analysis. Scoring read snapshot observations only and did
not read success/failure labels.

Family labeling is not complete. Role-parallel evaluation preserved completed
π0/π1 results for CALIBRATION and ACCEPTANCE, but concurrent π2 evaluators failed
after GPU 32 KB allocation errors. TRAIN π0 later failed after 1,409/1,754
candidates. Partial attempts and failure records are preserved.

## Current code state

Implemented:

- exact fixed-jump-start acquisition and family first-landing labels;
- resolution-aware causal and control occupancy analysis;
- outcome-blind holdout isolation against TRAIN and target Tube;
- Actor-only warm-start compatibility through the common policy loader;
- same-endpoint support in the current baseline/gate implementation;
- completed-evaluator reuse and preservation of incomplete attempts;
- first-landing-aware independent evaluator shards with strict global-index merge;
- advisory predictor fit, pre-outcome score lock and post-label audit join.

The same-endpoint code fix does not repair old results retroactively.

## Exact next actions

1. Run π0, π1 and π2 evaluator shards in separate GPU processes, with no more
   than 600 candidates per process, for each incomplete role.
2. Merge shards by global catalog index and verify equality on a small bank
   against the existing serial labeler before relying on the large merge.
3. Rebuild family OR labels and logical role manifests without changing catalog,
   seed, horizon or first-landing endpoint.
4. Join the locked predictor scores to fresh labels and report ROC-AUC, PR-AUC,
   recall, FPR, accepted negatives and group-aware uncertainty.
5. Run a retrospective π2-vs-π3 same-first-landing core diagnostic, explicitly
   labeled retrospective.
6. Freeze the controlled comparison matrix and total interaction accounting.
7. Do not train π4 until the comparison design, baseline endpoint, budget and
   stopping rule are predeclared.

## Paper readiness

The working hypothesis is that reachability-filtered reset curricula improve a
single unified jumping policy at controlled total interaction cost. Current
evidence does not yet establish that claim or RA-L readiness. Minimum remaining
evidence includes controlled baselines, at least three independent pilot seeds
(five preferred for the main result), group-aware intervals, interaction-cost
accounting and an untouched final distribution.

Final TEST/JCE/JEL remains untouched.
