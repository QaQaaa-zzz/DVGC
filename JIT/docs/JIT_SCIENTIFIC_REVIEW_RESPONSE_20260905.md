# DVGC/JIT scientific review response — 2026-09-05

## Decision

The review changes the project stop boundary. The repository no longer treats
the historical π3 selection as valid prospective evidence, and it does not
authorize π4 merely because Tube3 grew. The immediate priority is endpoint-
identical, budget-controlled and reproducible evidence.

## Findings verified against local artifacts

### 1. Five scientific objects must remain separate

The review's separation of arrival `A`, family witness `E`, physical cells `J`,
raw Tube `S` and single-policy realization `r` is adopted. This corrects a real
risk in earlier wording: family OR can join a π0 prefix to a π1/π2 suffix and
therefore cannot prove one-Actor execution.

### 2. π3's stored core comparison used different endpoints

Verified. The preserved gate summary states:

```text
baseline_success_criterion  stable_recovery
candidate_success_criterion first_valid_landing
```

Therefore 3,539→3,598 is not an endpoint-identical improvement comparison. The
historical selection was not fabricated after the result—the 5%/10% progression
margins pre-existed—but its input measurements are not comparable. The selected
manifest stays immutable and is reclassified as historical engineering
registration. Current code requires/records a common endpoint for new locks;
that code correction does not retroactively validate π3.

### 3. Tube growth did not automatically transfer to π3

Verified at the reported scope. On the same 1,258 source TRAIN states, π2 had
1,222 first landings while π3 had 1,130. π3 realized 1,061/1,159 states on the
actual Tube3 increment. This does not prove π3 is globally worse, but it does
show that curriculum support growth and one-Actor realization must be separate
outcomes and that forgetting/core retention needs a controlled study.

### 4. The current loop has not shown learning-driven discovery

Adopted. The first two scans freeze π0 as proposer and π0/π1/π2 as evaluator
family. Expanding lookbacks/strengths can produce new states without any benefit
from π3. These runs are fixed-family systematic scans, not evidence that the
newly trained policy improved discovery. A learning-guided allocator requires a
same-budget comparison against uniform/fixed-grid acquisition.

### 5. Predictor claims must remain narrow

Adopted. Old ACCEPTANCE ROC-AUC 0.89249 coexists with 6/9 negatives above the
locked threshold. Downstream has no negative class. The predictor is advisory;
all candidates still receive real rollouts. Future reports add PR-AUC, FPR,
group counts, calibration and uncertainty. No sampling-cost claim is made until
a same-budget predictor ablation exists.

### 6. Interaction accounting and statistical independence are incomplete

Adopted. PPO transitions alone undercount proposal prefixes, family rollouts,
development evaluation and failed attempts. Adjacent states from shared
trajectories/perturbation groups are correlated. Main results require
independent training seeds and group-aware resampling.

## Engineering response

Completed in the current source tree:

- common Actor-only warm-start loading without an iteration-specific loader;
- current baseline/gate support for explicit common success endpoint;
- evaluator Actor/payload and first-landing checks during family merge;
- preservation and reuse of completed evaluator attempts;
- first-landing-aware independent evaluator shards with distinct acquisition
  and evaluator identities;
- strict global-index shard merge;
- predictor fit, pre-outcome score lock and post-label exact join;
- authority/status/protocol documentation rewritten around the five objects.

Still requires real execution:

- small-bank serial-versus-sharded row equivalence;
- full family shard completion for the expanded catalogs;
- fresh predictor audit after labels close;
- retrospective same-first-landing π2/π3 diagnostic;
- controlled multi-seed baselines and interaction accounting.

## Current expanded round

The expanded round acquired 1,754 TRAIN, 583 CALIBRATION and 574 ACCEPTANCE
candidates. Upstream predictor scores were locked before outcome analysis. The
initial role-parallel family evaluation failed from CUDA allocation pressure:
CALIBRATION/ACCEPTANCE preserved completed π0 and π1 outputs, while π2 failed;
TRAIN π0 failed after 1,409 candidates. These are engineering failures, not
negative scientific labels.

The repair is at most 600 candidates per evaluator process, evaluator-serial on
one GPU, with strict merge. Catalogs, seeds, horizon and endpoint remain fixed.

## Paper thesis and falsification

Working thesis:

> A reachability-filtered reset curriculum improves one unified jumping policy
> at controlled total interaction cost relative to simpler reset curricula.

Evidence that would falsify or narrow it includes:

- no forward-task improvement over continued PPO or static Tube-RSI at matched
  total cost;
- Tube/cell growth without improved one-Actor realization;
- no discovery-efficiency gain over uniform fixed-grid scanning;
- high predictor false-positive rate without reduced labeling cost;
- effects that disappear across independent training seeds.

If those outcomes occur, the project should narrow to empirical local
capability measurement rather than presenting a closed-loop curriculum claim.

## Minimum pilot matrix

| Condition | Purpose |
|---|---|
| continued PPO / fixed curriculum | isolates benefit of further training |
| static successful Tube-RSI | tests whether dynamic expansion matters |
| uniform fixed-grid forward acquisition | controls for increased scan budget |
| RSI-only matched candidates | tests forward-arrival filtering |
| active reachable-filtered replay | complete proposed method |
| predictor removed | required only if predictor allocates future budget |

Use three independent seeds for the pilot and preferably five for the main
result. Predeclare total interaction budget, final perturbation distribution and
stopping rule. Final TEST/JCE/JEL remains untouched until that freeze.

## Paper positioning

The method should be positioned against reverse curriculum generation,
reference-state initialization, reset-based forward/reverse curricula and
feedback funnel libraries while explicitly distinguishing empirical support from
formal guarantees. Tube size alone should not be a headline figure. The main
figures should show controlled single-policy performance, cost, physical support
versus realization and ablations.

This document records a scientific correction and next experiment design. It is
not a claim that the required RA-L evidence already exists.
