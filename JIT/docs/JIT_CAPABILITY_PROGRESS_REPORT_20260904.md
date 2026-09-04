# DVGC/JIT capability progression report — 2026-09-04

## 1. Executive conclusion

DVGC/JIT has progressed from two phase-specific bootstrap experts to a complete
second empirical Tube and a trained/frozen `pi_2` candidate under one fixed robot
XML and one unchanged task definition.

The most important scientific conclusion is **not** that `pi_2` is a perfect
controller.  The stronger and more accurate conclusion is:

> JIT has accumulated substantially broader empirical evidence of jump-capable
> states under fixed dynamics, and the `pi_2` candidate demonstrates strong new
> frontier capability, while also exposing a major gap between cumulative system
> capability evidence and what one reward-guided unified policy can realize
> consistently in the upstream phase.

The old interpretation treated every single paired core regression as if it
invalidated envelope expansion.  That is too dependent to individual policy
rollout behavior, and it conflates two different research questions:

1. **Capability-envelope progression:** did the fixed robot/task demonstrate new
   jump-capable frontier states?
2. **Single-policy realization:** does the newest unified policy still cover enough
   of the already demonstrated state support, phase by phase, to become the sole
   authority for the next automatic iteration?

These two outcomes are now separated in code and documentation.

Current headline state:

```text
phase experts
  -> Tube_0 = 222
  -> pi_0
  -> C^0
  -> Tube_1 = 3,119
  -> pi_1 repair02
  -> pi_1 frontier v3 / v3b / v3c
  -> C^1 engineering-selected 64x64
  -> Tube_2 = 3,776
  -> pi_2 trained to 10,009,600 transitions
  -> locked pi_1 vs pi_2 comparison
       frontier progression: STRONG
       current-policy upstream coverage: DEGRADED
  -> CURRENT: method semantics revised; pi_2 preserved as capability evidence,
              but not retrospectively promoted to formal next authority
```

Final TEST/JCE/JEL evidence remains untouched.

---

## 2. What JIT is actually trying to identify

### 2.1 The fixed-XML physical question

The authoritative task remains fixed:

- XML: `assets/orange/../orange_bike_4kg_horizontal.xml` (authoritative repository
  path: `assets/orange_bike_4kg_horizontal.xml`)
- XML SHA-256:
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- actual payload: 2 kg; the `4kg` filename token is historical
- control rate: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`

For a fixed dynamics model, task, actuator limits, state definition, and success
criterion, there is conceptually a set of states from which **some admissible
control sequence** can complete the desired jump/recovery behavior.  Call this
unknown physical/task feasibility set `F*`.

JIT does **not** currently prove `F*`.  It is not a reachability solver, viability
kernel computation, optimal-control certificate, or safety proof.

What JIT can build is an increasingly broad empirical approximation supported by
actual successful closed-loop dynamics evidence.

### 2.2 Three layers that must not be conflated

The revised project semantics distinguish three objects.

#### A. Physical/task feasibility `F*`

Conceptual meaning:

> states from which there exists at least one admissible control behavior that
> completes the fixed jump task.

This is the ultimate physical question, but it is not directly observed.

#### B. Cumulative empirical capability evidence `E_k`

Meaning:

> states/regions for which the project has accumulated successful continuation
> evidence from one or more frozen experts or unified policies under the fixed
> task dynamics.

This is what JIT iteratively grows.

`Tube_k` is a structured TRAIN-only support/curriculum artifact derived from this
kind of evidence.  Its cardinality is **not** state-space volume, and a larger
Tube is not by itself proof of a larger physical feasibility set.  However, when
new states are reached through real dynamics, independently labeled, filtered by
continuation evidence, and later demonstrated on a locked frontier, they provide
empirical capability-progression evidence.

#### C. Single-policy realization coverage `R(pi_k, E_k)`

Meaning:

> how much of the cumulative demonstrated capability support one particular
> unified policy can realize under a fixed evaluation panel.

A later policy can forget some earlier behaviors without erasing the historical
fact that those behaviors were previously demonstrated by another frozen policy.
This is why a policy regression and an envelope regression are not identical.

The final deployable controller is still intended to be one unified Actor.  But
that deployment objective is now treated as **policy realization**, not as the
definition of the robot's physical capability envelope.

---

## 3. Why the earlier zero-regression rule was too much in this case

The unified policy observation does not currently contain an explicit requested
jump target such as:

- desired horizontal jump distance;
- desired apex height;
- desired landing location;
- desired recovery state;
- a low-dimensional jump-intent variable.

The policy is instead driven by the reward/task structure toward successful jump
behavior.  Therefore the same state may admit several reasonable actions, and a
stochastic/reward-guided policy is not expected to reproduce one unique action or
one unique successful rollout for every previously successful state.

The old rule was:

```text
if any state is
  pi_k success
  -> pi_(k+1) failure
then the whole iteration fails capability expansion
```

This remains useful as a **strict behavioral-retention diagnostic**, but it is no
longer treated as the definition of envelope progression.

The new interpretation asks:

```text
frontier progression?
        +
phase-aware cumulative Tube coverage retained?
        -> candidate can become next automatic authority
```

Small individual paired regressions are allowed in principle.  A large
phase-specific collapse is not.

---

## 4. Bootstrap: Propulsion-Ascent and Descent-Recovery experts

### 4.1 `pi_up_star`

Role:

- bootstrap Propulsion-Ascent behavior;
- generate launch/rising-flight trajectories and upstream continuation evidence;
- support the first empirical capability Tube.

Training:

- 9,977,856 transitions.

Actor SHA-256:

`f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`

### 4.2 `pi_down_star`

Role:

- bootstrap Descent-Recovery behavior;
- provide evidence for descent, landing, and recovery continuations.

Training:

- 25,600 transitions.

Actor SHA-256:

`7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`

Frozen expert manifest:

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### 4.3 Why experts are not the final controller

The phase experts are best interpreted as **capability probes and bootstrap data
sources**.  They are not intended to be switched at runtime in the final system.
The runtime target remains one unified Actor.

Their scientific purpose is broader than imitation: they reveal portions of the
fixed system's achievable behavior that can be turned into empirical continuation
support for unified-policy learning.

---

## 5. Bootstrap continuation fields and Tube_0

Frozen expert continuation evidence produced bootstrap `V_up` and `V_down`
authorities.  These are expert-conditioned bootstrap fields only; they are not
reused as later unified-policy continuation fields.

Tube_0:

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

Composition:

```text
222 TRAIN states
  upstream   117
  downstream 105
```

Manifest SHA-256:

`c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`

Scientific meaning:

- this is the first compact empirical training/curriculum support extracted from
  two-phase capability evidence;
- it is not a certified safe set;
- it establishes a first state support from which one unified policy can be
  trained.

---

## 6. Unified policy `pi_0`

Frozen policy:

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

Training:

- 10,009,600 PPO transitions.

Identity:

- actor SHA-256:
  `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`
- payload SHA-256:
  `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`

`pi_0` is the first single Actor trained from the empirical Tube support.  It
established that the two-phase bootstrap evidence can be consumed by one unified
policy without runtime expert switching.

---

## 7. `C^0` and the first major envelope-support expansion: Tube_1

After `pi_0` was frozen, continuation evidence was regenerated under the exact
frozen unified policy and used to fit policy-conditioned `C_up^0/C_down^0`.
These fields passed the then-declared fresh validation/calibration path.

Tube_1:

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

Composition:

```text
retained Tube_0 =   222
new expansion   = 2,897
total           = 3,119

upstream   =   427 = 117 retained + 310 expansion
downstream = 2,692 = 105 retained + 2,666? historical summary authority says
                      105 retained + 2,587 expansion = 2,692
```

The authoritative decomposition is:

```text
upstream expansion   = 310
downstream expansion = 2,587
```

Therefore:

- total Tube entries grew from 222 to 3,119, about **14.05x** the original
  cardinality;
- upstream support grew from 117 to 427, about **3.65x**;
- downstream support grew from 105 to 2,692, about **25.64x**.

These are **entry-count growth factors**, not physical state-space volume ratios.

Manifest SHA-256:

`817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`

Entries SHA-256:

`61c6796aaf4c4b1e43624c5cf06bce0d39736a6d1743c5142c6c250d23155ec9`

Task-level significance:

> JIT had moved from a small expert-bootstrapped Tube to a much broader
> policy-conditioned empirical training support while retaining the original
> Tube_0 evidence exactly.

---

## 8. Iteration-1 unified-policy study and `pi_1`

Training directly on the enlarged Tube exposed a central learning problem:
cardinality expansion can dilute replay of earlier support.

The project therefore studied retained-core replay and bounded warm-start
variants.  The study is closed.

Final comparison:

| policy/checkpoint | Tube_0 | regressions | upstream | downstream | boundary | groups |
|---|---:|---:|---:|---:|---:|---:|
| repair02 | **222/222** | **0** | **117/117** | **105/105** | 26/260 | 4 |
| B 1.024M | 217/222 | 5 | 112/117 | 105/105 | 33/260 | 3 |
| B 2.5088M | 206/222 | 16 | 101/117 | 105/105 | 28/260 | 4 |
| B 5.0176M | 214/222 | 8 | 109/117 | 105/105 | 25/260 | 4 |
| B 7.5008M | 217/222 | 5 | 112/117 | 105/105 | 42/260 | 4 |
| B 10.0096M | 212/222 | 10 | 107/117 | 105/105 | 46/260 | 4 |

No B checkpoint simultaneously preserved all 222 Tube_0 states and exceeded the
repair02 boundary result under the then-current retention-first selection rule.

Selected engineering `pi_1`:

`JIT/runs/frozen_unified/pi_1_core_replay75_100096ed00_20260903/...`

Authoritative exact path:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

Identity:

- actor SHA-256:
  `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`
- payload SHA-256:
  `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`

Historical quickcheck:

- Tube_0: 222/222;
- upstream: 117/117;
- downstream: 105/105;
- boundary: 26/260 across 4 parent groups.

Historical claim boundary:

- engineering selection: yes;
- old strict formal Iteration-1 PASS claim: no, because 3 boundary baseline
  reproduction mismatches remain quarantined from the old PRNG hierarchy.

### 8.1 What Iteration-1 taught us

All B core regressions were upstream while downstream remained 105/105.  Boundary
gains were also concentrated upstream.

This was early evidence that **frontier acquisition/learning and old-behavior
retention can interfere strongly inside one reward-guided policy**, especially in
the upstream phase.

The current `pi_2` result later confirms that this is not an isolated artifact.

---

## 9. `pi_1` frontier acquisition: from failed probes to usable two-phase evidence

The Iteration-1 frontier stage exposed that continuation evidence cannot be
assumed to contain both success and failure classes merely because many samples
are collected.

### 9.1 v1

Downstream TRAIN produced no usable negative/transition structure.  This was a
frontier-design failure, not an OOM and not proof that `pi_1` was incapable.

### 9.2 v2

A stronger local single-axis probe still produced downstream all-positive
support.  The frontier remained insufficiently informative.

### 9.3 v3

The acquisition was changed to phase-specific perturbation strength:

- upstream: weaker single-axis probes;
- downstream: stronger two-axis probes.

TRAIN result:

```text
total      1,031
upstream     821 = 785 positive + 36 negative, 9 parent groups
downstream   210 = 182 positive + 28 negative, 3 parent groups
```

This was the first usable two-phase Iteration-1 TRAIN frontier.

### 9.4 v3b calibration repair

Original upstream calibration was all positive, so it could not support the
fixed threshold contract.

A stronger two-axis upstream CALIBRATION probe, using the same already-declared
three calibration parents, produced:

```text
739 upstream calibration candidates
733 positive
6 negative
3 parent groups
```

Downstream calibration remained:

```text
70 candidates
61 positive
9 negative
1 parent group
```

### 9.5 v3c fresh acceptance challenge

Fresh ACCEPTANCE evidence was collected independently:

```text
upstream   516 = 511 positive + 5 negative, 3 parent groups
downstream  70 =  61 positive + 9 negative, 1 parent group
```

The negative challenge support spans three parent groups in total and was later
locked before `pi_2` training.

---

## 10. `C^1`: what worked, what did not, and why the claim is engineering-only

### 10.1 Original small network

The original continuation field was not linear; it was approximately:

```text
76 -> 8 tanh -> 1
625 parameters
```

Upstream calibration:

- ROC AUC: 0.66348;
- positive recall: 0.23465;
- zero accepted negatives;
- failed the declared AUC >= 0.70 gate and failed accepted-positive support in
  every parent.

### 10.2 Standard 64x64 MLP

Architecture:

```text
76 -> 64 tanh -> 64 tanh -> 1
9,153 parameters
```

Using the same TRAIN and v3b CALIBRATION data:

Upstream:

- threshold: 0.9835533512239714;
- ROC AUC: **0.6903137789904502**;
- positive recall: **0.5934515688949522**;
- zero accepted negatives;
- every calibration parent obtained accepted positive support.

This was a substantial practical improvement over the 8-unit network, but it
still missed the original AUC 0.70 threshold by about 0.00969.

### 10.3 128x128 trial

Architecture:

```text
76 -> 128 tanh -> 128 tanh -> 1
26,497 parameters
```

Upstream performance degraded strongly:

- ROC AUC: 0.52956;
- positive recall: 0.05593;
- score gap became negative.

Therefore widening the network further was rejected.

### 10.4 Engineering-selected C_up^1 and formal C_down^1

The user explicitly selected 64x64 as the engineering mainline C^1 architecture.
The original upstream AUC contract was **not rewritten**.

Upstream `C_up^1`:

- architecture: 64x64 tanh;
- AUC: 0.6903137789904502;
- recall: 0.5934515688949522;
- formal upstream AUC gate: false;
- explicit engineering override: true.

Downstream `C_down^1` on the same 64x64 profile:

- candidate count: 70;
- 61 positive / 9 negative;
- ROC AUC: **1.0**;
- positive recall: **1.0**;
- accepted negatives: 0;
- formal calibration: PASS.

This means `C^1` is a deliberately mixed-status artifact:

```text
upstream: engineering-selected, not formal AUC PASS
downstream: formal calibration PASS
```

It must not be described as a clean all-phase formal continuation-model PASS.

---

## 11. Tube_2

Tube_2:

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

Manifest SHA-256:

`135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`

Composition:

```text
source Tube_1 retained exactly = 3,119
new expansion                  =   657
Tube_2 total                   = 3,776

upstream total   =   902
  source Tube_1  =   427
  new expansion  =   475

downstream total = 2,874
  source Tube_1  = 2,067? no — authoritative source Tube_1 downstream is 2,692
  new expansion  =   182
  total           = 2,874
```

Authoritative phase decomposition:

```text
upstream   427 + 475 = 902
downstream 2692 + 182 = 2874
```

Relative cardinality changes:

- Tube_1 -> Tube_2: +657 entries, **+21.06%**;
- upstream: 427 -> 902, **+111.24%**;
- downstream: 2,692 -> 2,874, **+6.76%**;
- Tube_2 / Tube_0 cardinality: about **17.01x**.

Again these are support-entry counts, not geometric state-space volume.

Selection details:

```text
upstream TRAIN positives above C_up^1 threshold   = 475
downstream TRAIN positives above C_down^1 threshold = 182
```

No CALIBRATION, ACCEPTANCE, TEST, or final-evaluation rows were embedded.

### 11.1 Tube_2 smoke

Tube-RSI smoke:

- 8 upstream samples;
- 8 downstream samples;
- all finite;
- `tube_rsi_smoke = GO`;
- TEST/validation unused.

### 11.2 Role-isolation evidence

Exact-state overlaps across TRAIN/CALIBRATION/ACCEPTANCE: zero.

Parent-group overlap: zero.

Near-observation overlap at `atol=0.01`:

```text
TRAIN <-> CALIBRATION       140
TRAIN <-> ACCEPTANCE          0
CALIBRATION <-> ACCEPTANCE  157
```

All observed near-overlap was upstream and between distinct parent groups.

Because candidate training and acceptance remained geometrically isolated,
engineering continuation was allowed with an explicit non-formal isolation
claim:

```text
candidate_training_acceptance_isolation_passed = true
formal_all_role_geometric_isolation_passed = false
```

This is another reason the Iteration-1 -> 2 round must be described as an
engineering mainline round rather than an immaculate prospective formal round.

---

## 12. Locked `pi_1` baseline before `pi_2` training

The new locked-baseline protocol was successfully used before candidate
training.

Source Tube_1:

```text
3,119 states
pi_1 baseline success = 3,115
```

Phase baseline support later reported by the gate:

```text
upstream   423 / 427
downstream 2692 / 2692
```

The acceptance boundary baseline was locked before training, eliminating the
old historical PRNG mismatch caused by rerunning a previously negative baseline
under a different key hierarchy.

This part of the new protocol worked as intended:

```text
boundary baseline reproduction failure count = 0
```

---

## 13. `pi_2` training

Training run:

`JIT/runs/pi_unified/pi_2_tube2_c1_64x64_engineering_core75_natural10_100_09600_seed821101_20260904`

Authoritative run id actually used:

`pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

Training completed:

- requested transitions: 10,009,600;
- completed transitions: 10,009,600;
- outer reset: 90% Tube / 10% natural;
- inside Tube: 75% retained source Tube_1 / 25% Tube_2 newest expansion;
- fresh unified PPO job;
- no expert switching;
- no TEST or validation data.

Final reported training metrics included:

- KL mean: 0.004306;
- policy loss: 0.006256;
- value loss: 4.5473;
- SPS: about 56.8k;
- walltime: about 211 s on the local RTX 4090 D environment.

The exact frozen actor/payload SHA values are bound in the local
`frozen_unified_policy.json` and gate summary.  This report does not invent SHA
values that were not included in the pasted summary evidence.

---

## 14. `pi_1` vs `pi_2`: the result that changes the JIT story

Locked paired result:

```text
strict iteration_accepted = false
```

### 14.1 Old-core comparison

Overall:

```text
states                     3,119
pi_1 baseline successes    3,115
pi_2 candidate successes   3,002
strict regressions           115
strict improvements            2
```

Panel coverage:

```text
pi_1: 3115/3119 = 99.87%
pi_2: 3002/3119 = 96.25%
absolute global drop ≈ 3.62 percentage points
```

If only global coverage were inspected, this might look modest.

But phase-wise analysis reveals the important structure.

Upstream:

```text
states                  427
pi_1 success             423 = 99.06%
pi_2 success             312 = 73.07%
strict regressions       113
strict improvements        2
coverage drop ≈ 25.995 percentage points
```

Downstream:

```text
states                 2692
pi_1 success            2692 = 100.00%
pi_2 success            2690 = 99.93%
strict regressions         2
coverage drop ≈ 0.074 percentage points
```

Therefore the problem is not general policy collapse.  It is an **upstream
policy-realization collapse masked by the much larger downstream state count**.

### 14.2 Locked frontier/boundary comparison

The boundary result is extremely different:

```text
14 locked pi_1-negative challenge states
pi_2 success = 13/14
successful parent groups = 3
minimum required groups = 2
boundary gate = PASS
baseline reproduction failures = 0
```

By phase:

```text
upstream   4 / 5
 downstream 9 / 9
```

This is strong evidence that `pi_2` learned behavior that reaches beyond the
locked `pi_1` boundary challenge.

### 14.3 Correct interpretation

The correct interpretation is **not**:

> pi_2 failed, therefore JIT did not expand capability.

It is:

> The new Tube and training produced a policy with strong local frontier
> progression in both phases, but the same policy lost substantial upstream
> realization coverage over previously demonstrated support.

This distinction is exactly why envelope evidence and policy realization must be
reported separately.

---

## 15. Revised capability-progression decision implemented in code

New analysis capability:

`jit_dvgc.analysis.capability_progression`

CLI:

`JIT/cli/analyze_capability_progression.py`

The v1 decision reports two independent axes.

### 15.1 Frontier progression

Required:

- no baseline-reproduction failure;
- at least one candidate boundary success;
- sufficient independent parent groups;
- candidate boundary success in both upstream and downstream.

This supports an **empirical local frontier progression** claim only.

### 15.2 Policy realization retention

The current engineering proxy is fixed locked-panel success coverage.

Prospective automatic-selection margins are:

```text
maximum allowed global Tube coverage drop = 5 percentage points
maximum allowed per-phase coverage drop   = 10 percentage points
```

These margins deliberately allow some stochastic/single-rollout variation while
preventing a large phase-specific collapse from being hidden by an imbalanced
Tube cardinality.

Zero strict paired regressions are **not** required by the new policy-authority
criterion.

### 15.3 Current `pi_2` under the revised semantics

Retrospective classification:

```text
empirical_envelope_expansion_observed = true
policy_realization global margin       = pass
policy_realization downstream margin   = pass
policy_realization upstream margin     = fail
candidate_policy_authority_eligible    = false
decision = envelope_progressed_but_candidate_policy_coverage_degraded
```

Because this method interpretation was revised **after** the current `pi_2`
result had been observed, the current candidate is not retroactively promoted to
a formal prospective PASS.  Its evidence is preserved, but a future automatic
selection must use the new decision contract prospectively.

---

## 16. What has actually improved at the task level

### Improvement 1 — capability-support breadth

The project progressed from 222 bootstrap Tube states to 3,776 Tube_2 states.

```text
Tube_0  222
Tube_1 3119
Tube_2 3776
```

Tube_2 contains about 17.01 times as many retained/qualified TRAIN support entries
as Tube_0.

This does not mean 17x physical jump range.  It means the empirical training and
continuation support has become dramatically broader while preserving provenance.

### Improvement 2 — two-phase boundary evidence

The v3/v3b/v3c process finally produced informative positive/negative frontier
support in both phases rather than all-positive banks.

This is important because a capability boundary cannot be identified from an
all-success sample cloud.

### Improvement 3 — unified-policy frontier gain

`pi_2` succeeds on 13 of 14 states locked as failures for `pi_1`, spanning three
parent groups and both phases.

This is direct task-level evidence that the training process can create new
closed-loop jump/recovery capability beyond the previous policy's local frontier.

### Improvement 4 — protocol quality

The pi_1 baseline was locked before pi_2 training, and the later boundary
comparison had zero baseline-reproduction mismatch.

This removes the historical PRNG debt that affected the old pi_0 -> pi_1
quickcheck.

### Improvement 5 — clearer scientific object

The project now separates:

- physical/task feasibility;
- cumulative empirical capability evidence;
- current-policy realization coverage;
- curriculum/Tube support.

This is a stronger and more defensible JIT story than treating one policy's
single rollout on every state as the definition of the robot's capability.

---

## 17. What has not yet been solved

### 17.1 JIT has not yet identified the physical limit `F*`

Tube_2 and the frontier results are empirical support, not proof of the fixed
robot's maximum jump distance/height/landing region.

### 17.2 Current continuation fields are still policy-conditioned

`C^k(s)` answers continuation under a particular frozen `pi_k`, not existential
controllability under all admissible controllers.

Therefore a failed `C^k` state may still be physically feasible under another
policy.

### 17.3 The latest unified policy is not goal-conditioned

The current observation does not specify what jump outcome is requested.  This
likely contributes to interference when one policy is asked to cover a broad
collection of upstream behaviors.

### 17.4 Current coverage is a fixed-panel proxy, not calibrated per-state success probability

A stronger future evaluator should run multiple predeclared seeds per state and
estimate success probability or a confidence interval rather than treating one
rollout as the whole stochastic policy response.

### 17.5 Current automatic frontier uses the latest selected policy only

If JIT's scientific target is increasingly policy-independent empirical system
capability, future acquisition should consider a **policy archive** or region-wise
best probe instead of assuming the latest policy is the only valid capability
probe.

That does not imply runtime expert switching.  It is a discovery-time mechanism.
The deployable controller can still remain one unified policy.

---

## 18. Revised JIT story

A concise research narrative is:

> JIT couples empirical capability identification with just-in-time curriculum
> generation.  Under fixed robot dynamics, frozen experts and unified policies
> act as capability probes.  Successful real-dynamics continuation evidence is
> accumulated into a monotonic empirical support, while frontier states close to
> the current success/failure transition generate the next training curriculum.
> The resulting unified policies are evaluated both for new frontier progression
> and for how much of the cumulative capability support they can realize.  The
> latest policy is therefore an implementation of the discovered capability, not
> the definition of the physical capability itself.

The loop is better written as:

```text
capability probe policy/archive
        ↓
real-dynamics frontier discovery
        ↓
continuation evidence
        ↓
cumulative empirical capability support E_k
        ↓
just-in-time Tube curriculum
        ↓
new unified policy probe / realization candidate
        ↓
A. frontier progression?
B. phase-aware policy realization retained?
        ↓
repeat or revise representation/training
```

---

## 19. Automatic iteration status

### 19.1 What is implemented generically

The branch contains reusable automation for `k -> k+1`:

1. newest-shell frontier planning;
2. TRAIN acquisition/labeling;
3. CALIBRATION acquisition/labeling;
4. ACCEPTANCE acquisition/labeling;
5. C^k fit/calibration;
6. Tube_(k+1) construction;
7. Tube-RSI smoke;
8. role isolation audit;
9. pre-candidate baseline lock;
10. candidate training;
11. freeze;
12. locked paired evaluation;
13. **new capability-progression analysis**;
14. prospective policy selection only when both frontier progression and
    phase-aware policy realization pass.

### 19.2 Why the current pi_1 -> pi_2 round was not fully automatic

The generic workflow existed, but this round required explicit engineering
interventions:

- initial frontier panels failed to produce informative downstream support;
- upstream calibration needed v3b stronger acquisition;
- C_up^1 original model failed;
- 64x64 was selected as an engineering architecture on reused data;
- upstream AUC remained 0.6903 < 0.70 and required an explicit engineering
  continuation override;
- strict all-role near-observation isolation failed and was replaced by an
  explicit engineering continuation record after confirming TRAIN <-> ACCEPTANCE
  near-overlap was zero.

Therefore it is inaccurate to claim the entire pi_1 -> pi_2 round was one clean,
untouched automatic execution.

The correct statement is:

> The production DAG exists and most stage semantics are automated, but the
> current round crossed two explicitly recorded engineering decision points.
> Future rounds should use the updated capability-progression decision
> prospectively and should stop rather than silently invent another override.

---

## 20. Current exact project position

Completed:

```text
experts                         DONE
Tube_0                          DONE
pi_0                            DONE
C^0                             DONE
Tube_1                          DONE
pi_1 engineering authority      DONE
pi_1 frontier roles             DONE
C^1 engineering selection       DONE
Tube_2                          DONE
Tube_2 smoke                    GO
role isolation engineering      DONE
pi_1 baseline lock              DONE
pi_2 training                   DONE
pi_2 freeze                     DONE
pi_1 vs pi_2 locked comparison  DONE
capability-semantics revision   CODED
```

Current scientific decision:

```text
pi_2 demonstrated frontier progression
BUT
pi_2 is not retrospectively selected as formal next authority because
upstream policy realization dropped from 99.06% to 73.07% on the locked Tube_1 panel.
```

No `pi_3` work should begin yet.

---

## 21. What should happen next

### Step 1 — generate the retrospective capability decision artifact for current pi_2

Use the new analyzer on the already-completed gate summary with
`--retrospective`.

Expected semantic result:

```text
empirical_envelope_expansion_observed = true
candidate_policy_authority_eligible = false
retrospective_analysis = true
```

This records the new interpretation without rewriting history or selecting pi_2.

### Step 2 — stop treating replay-ratio tuning as the automatic answer

The current pi_2 result should not immediately trigger 75/25 -> 90/10 replay
because the deeper issue is not only replay frequency.  The upstream policy is
being asked to represent a broad set of behaviors without an explicit desired
jump target.

A replay repair may remain an ablation, but it should not be the default next
scientific move.

### Step 3 — define the next representation question

The strongest next research question is whether a single policy should receive a
low-dimensional **jump intent / goal condition**, for example one or more of:

- desired horizontal travel;
- desired apex/clearance;
- desired landing region;
- desired recovery speed/posture;
- a normalized progress/behavior code derived from Tube/frontier structure.

This would preserve the single-policy deployment requirement while allowing the
policy to express different behaviors intentionally rather than relying on reward
alone to pick one.

This change would be a new method version and must not be slipped into the
existing iteration as a hidden repair.

### Step 4 — upgrade capability evaluation from single rollout to success probability

For future prospective gates, lock multiple policy seeds per state before
candidate training and report phase-wise success-rate/confidence estimates.

This is a better match to a stochastic policy than exact one-rollout
reproduction.

### Step 5 — consider a discovery-time policy archive

To approach system capability rather than latest-policy capability, future
frontier acquisition can maintain frozen probe policies and use them only for
scientific discovery.

Conceptually:

```text
E_k = union of capability evidence demonstrated by frozen probes up to k
```

The archive would not be used for runtime switching.  It would prevent a later
policy's local forgetting from erasing discovery access to regions that an older
policy could still probe successfully.

### Step 6 — only then decide pi_3 / next Tube semantics

Before launching another automatic iteration, decide whether the next method
version is:

1. same policy representation + a prospectively declared training repair; or
2. goal-conditioned unified policy; or
3. archive-assisted capability discovery + separate unified realization.

The recommended direction is to evaluate options 2 and 3 as the main scientific
advance because they directly address the conceptual distinction exposed by
pi_2.

---

## 22. Publication/claim boundary after this round

Supported claims:

- phase experts successfully bootstrap a unified JIT pipeline;
- policy-conditioned Tube expansion can dramatically enlarge empirical training
  support;
- Tube_2 retains all prior Tube evidence and adds 657 evidence-backed TRAIN
  states;
- `pi_2` demonstrates strong local frontier progression on the locked pi_1
  challenge: 13/14 successes across 3 parent groups and both phases;
- the new locked-baseline protocol eliminates the historical boundary
  reproduction mismatch in this round;
- cumulative empirical capability progression and single-policy realization are
  measurably different quantities;
- upstream policy interference is a recurring bottleneck.

Not supported:

- Tube_2 is the true physical maximum jumping envelope;
- JIT has computed a viability kernel or safe set;
- C_up^1 formally passed the original AUC contract;
- the current pi_1 -> pi_2 round was fully prospective and automatic from start
  to finish;
- pi_2 is formally selected as the next authority under the newly revised gate;
- final JCE/JEL has been measured;
- the physical system cannot perform states that the latest policy fails.

---

## 23. Artifact map

### Experts

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_100736?`

Authoritative:

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

### pi_1

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

### Iteration-1 -> 2 work root

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3`

### C^1 engineering selection

`.../continuation_C1_standard_mlp64x64_engineering_selected_v1`

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_64x64_engineering_20260904` may be a shorthand.

Authoritative created path:

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

### pi_2 training

`JIT/runs/pi_unified/pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

### pi_2 frozen policy

`JIT/runs/frozen_unified/pi_2_c1_64x64_engingineering_10009600_20260904` is not authoritative spelling.

Use the actually created local path:

`JIT/runs/frozen_unified/pi_2_c1_64x64_engineering_10009600_20260904/frozen_unified_policy.json`

### pi_1 -> pi_2 gate

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3/pi_1_to_pi_2_gate_c1_64x64_engineering/summary.json`

---

## 24. One-sentence JIT definition going forward

> **JIT is an iterative real-dynamics capability-discovery and just-in-time
> curriculum framework that accumulates empirical jump-capability evidence under
> fixed robot dynamics, uses the current frontier to train a single unified
> policy, and separately measures frontier progression and how much of the
> cumulative capability that policy can realize.**
