# DVGC / JIT project

## Objective

Test whether a physically reached, empirically landing-capable reset curriculum
improves one unified robotic jumping policy at controlled total interaction
cost. The project is an empirical learning and capability-measurement study, not
a reachability proof or safety certificate.

Suggested working title:

> JIT: Reachability-Filtered Reset Curricula for Single-Policy Robotic Jumping

This is a paper direction, not an achieved result.

## Fixed task

The current task uses one single-track two-wheeled model, 2 kg payload, 0.005 s
MuJoCo simulation substep and 0.020 s control interval. Actions are steering,
rear-wheel drive, hip and knee; hip/knee torque ranges are +/-30 N m. The final
runtime is one Actor without expert switching. Physics, reward, actuator order
and task geometry cannot change silently.

The active experiment begins from the locked ground jump-start state at
`x = 2.5 m`. It does not currently claim connection to the natural episode
reset. Final TEST/JCE/JEL is untouched.

## Scientific objects

### Forward arrival evidence A

Frozen π0 executes from the jump-start snapshot. A bounded predeclared action
perturbation is applied inside a lookback window. A candidate exists only if the
exact state is entered through authoritative `env.step` dynamics.

### Policy-family landing witness E

The reached snapshot is restored separately under frozen π0, π1 and π2. The
candidate is positive if any evaluator reaches the first valid landing before
physical failure. Recovery after landing is outside this label.

This family OR is an offline composition witness. It does not show that one
Actor can execute the π0 prefix and the successful suffix as one rollout.

### Physical capability occupancy J

Positive `A intersect E` states are projected into fixed physical-resolution
cells. The primary geometry uses 0.10 m position and 0.10 m/s velocity bins.
Cell occupancy does not make every state inside a cell feasible and is not a
continuous volume or safety guarantee.

### Raw/control Tube S

The Soft Tube is a replay/reset artifact. It retains historical core rows and
observed new positive states. Its row count and all-state physical occupancy are
not causal Jump-Capability counts.

### Single-Actor realization r

Each frozen Actor is evaluated on common locked panels. Realization measures
whether curriculum support transfers to one policy. It must be reported beside,
not replaced by, family capability growth.

## Centerline

The centerline is a locked real-frame π0 trajectory sampled at 0.1 m x slices
from 2.5 m to the first valid landing or 4.2 m. It is an exploration scaffold,
not a reference-tracking objective, action command, reward target or
interpolated physical trajectory.

## Data roles

- TRAIN can fit models and contribute observed positives to replay.
- CALIBRATION selects thresholds only.
- ACCEPTANCE is locked development evidence; using it for a decision consumes
  it as development data.
- final TEST/JCE/JEL stays isolated until the complete method is frozen.

Role assignment, cross-role deduplication and target-Tube exclusion are outcome
blind. Statistics must respect shared trajectory/perturbation groups rather than
treating adjacent states as independent trials.

## Current empirical state

The first fixed-jump-start family round acquired 1,258 TRAIN candidates and
observed 1,230 family first-landing positives. After deduplication against
Tube2, Tube3 added 1,159 rows and reached 4,803 total rows. Causal TRAIN root
occupancy added 713 cells; all-state Tube control root occupancy added 714.

π3 was trained for 10,009,600 transitions from a π2 Actor/normalizer warm start.
Its historical stored panel diagnostic reported 3,598 successes versus 3,539
for the baseline, with 89 improvements and 30 regressions. That comparison is
not a fair prospective selection result because the baseline core used
`stable_recovery` and the candidate used `first_valid_landing`. The historical
selection artifact is retained but quarantined from current authority.

On the 1,258 source TRAIN states, first-landing successes were π0 1,130, π1
1,184, π2 1,222 and π3 1,130. This is support realization, not final task
performance. It shows why Tube growth and policy learning must be measured
separately.

An upstream family-landing predictor reached old ACCEPTANCE ROC-AUC 0.89249 and
positive recall 0.98611, but accepted 6/9 negatives at its locked threshold.
It is advisory only. Downstream was all-positive and was not fit.

The expanded follow-up round acquired 1,754 TRAIN, 583 CALIBRATION and 574
ACCEPTANCE candidates and locked pre-outcome predictor scores. Family labeling
is incomplete due GPU allocation failures in long-lived evaluator processes.
Independent-process candidate sharding is the active repair. No π4 training is
authorized.

## Prospective loop

```text
lock task, centerline, proposer, evaluator family and roles
-> acquire exact jump-start-connected candidates
-> lock predictor scores without reading outcomes, if auditing a predictor
-> run memory-bounded real evaluator rollouts
-> form family landing labels
-> audit role isolation and physical novelty
-> build TRAIN-only replay expansion
-> lock endpoint-identical baseline and selection contract
-> train one unified candidate
-> evaluate capability growth and single-policy realization separately
-> select or stop
```

No future training begins merely because the Tube grew.

## Paper experiment matrix

The minimum controlled comparisons are:

1. continued PPO/fixed curriculum under matched initialization and budget;
2. static successful Tube-RSI;
3. fixed-grid forward acquisition with uniform budget;
4. RSI-only candidates under matched labeling/training budget;
5. the active reachable-filtered iterative replay method;
6. predictor removed, if the predictor is later allowed to allocate budget.

Primary outcomes are frozen forward-task first-landing success and total
environment interactions required to reach a declared development level.
Secondary outcomes include old/new support realization, regression count,
novel physical cells per million interactions, near-duplicate rate and
resolution sensitivity.

Interaction accounting includes shared bootstrap, acquisition prefixes,
successful and failed family labels, training, development selection and failed
retries. Wall time and hardware are separate operational metrics.

Use at least three independent training seeds for the pilot and preferably five
for the main comparison. Multiple checkpoints from one seed are not repeats.
Freeze the final perturbation distribution and stopping rule before final tests.

## Immediate work

1. Finish evaluator-specific shards and strict merge for the expanded catalogs.
2. Audit the already locked predictor scores on fresh results.
3. Produce a retrospective same-first-landing π2/π3 diagnostic without
   reinterpreting it as prospective.
4. Freeze method variants, budgets, metrics, group resampling and stopping rule.
5. Run the minimal controlled pilot before deciding whether another policy
   iteration is scientifically justified.

The key next result is a same-standard, same-budget, reproducible comparison,
not a larger π index.
