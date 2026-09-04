# DVGC/JIT Capability Progression Report — 2026-09-04

## 1. Executive conclusion

DVGC/JIT has progressed from two phase-specific bootstrap experts to a complete second empirical Tube and a trained/frozen `pi_2` candidate under one fixed robot XML and one unchanged task definition.

The central result is not “pi_2 is a perfect controller.” The stronger and more accurate result is:

> JIT has accumulated substantially broader empirical evidence of jump-capable states under fixed dynamics, and `pi_2` demonstrates strong new frontier capability while exposing a large gap between cumulative capability evidence and what one reward-guided unified policy can realize consistently in the upstream phase.

The old interpretation treated every paired core regression as if it invalidated capability-envelope progression. That conflates two different scientific questions:

1. **Capability-envelope progression:** did the fixed robot/task demonstrate new jump-capable frontier states?
2. **Single-policy realization:** does the newest unified policy still cover enough of the already demonstrated state support, phase by phase, to become the sole authority for the next automatic iteration?

These two outcomes are now separated in code, automatic workflow logic, selection semantics, and documentation.

Current high-level state:

```text
pi_up_star + pi_down_star
  -> Tube_0 = 222
  -> pi_0
  -> C^0
  -> Tube_1 = 3,119
  -> pi_1 repair02
  -> pi_1 frontier v3 / v3b / v3c
  -> C^1 64x64 engineering selection
  -> Tube_2 = 3,776
  -> pi_2 trained/frozen at 10,009,600 transitions
  -> locked pi_1 vs pi_2 comparison
       frontier progression: STRONG
       current-policy upstream realization: DEGRADED
  -> capability-progression semantics revised
  -> CURRENT: pi_2 is retained as capability evidence but is not retrospectively
              promoted to the next formal policy authority
```

Final TEST/JCE/JEL evidence remains untouched.

---

## 2. What JIT is actually trying to identify

### 2.1 Fixed-model physical/task feasibility

The authoritative task remains fixed:

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b7f?` is not valid; authoritative SHA is
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- actual payload: 2 kg
- control rate: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`

For a fixed dynamics model, task, actuator limits, state definition, and success criterion, there is conceptually a set of states from which **some admissible control sequence** can complete the jump/recovery behavior. Call this unknown physical/task feasibility set `F*`.

JIT does **not** currently prove `F*`. It is not a reachability solver, viability-kernel computation, optimal-control certificate, or safety proof.

### 2.2 Three layers that must remain separate

#### A. Physical/task feasibility `F*`

Conceptual meaning:

> states from which at least one admissible control behavior could complete the fixed task.

This is the ultimate physical question, but it is not directly observed by the current method.

#### B. Cumulative empirical capability evidence `E_k`

Meaning:

> states/regions for which the project has accumulated successful real-dynamics continuation evidence from frozen experts or unified policies.

This is what JIT attempts to grow iteratively.

`Tube_k` is a structured TRAIN-only support/curriculum artifact derived from this type of evidence. Tube cardinality is not physical state-space volume.

#### C. Single-policy realization `R(pi_k, E_k)`

Meaning:

> how much of the cumulative demonstrated support one particular unified policy can realize on a locked evaluation panel.

A later policy can forget some earlier behaviors without erasing the historical fact that those behaviors were previously demonstrated by another frozen policy.

This distinction is central to the revised JIT story.

---

## 3. Why the previous zero-regression rule was too much in this task

The current unified policy observation does not explicitly specify:

- desired horizontal jump distance;
- desired apex height or clearance;
- desired landing location;
- desired recovery state;
- another explicit jump-intent variable.

The policy is reward-guided toward a successful jump behavior. Therefore a single state can admit multiple reasonable control responses, and a stochastic/reward-guided policy is not expected to reproduce one unique successful action sequence for every previously successful state.

The old decision rule was effectively:

```text
if any baseline-success state becomes candidate-failure
then the whole iteration fails capability expansion
```

This remains useful as a **strict behavioral-retention diagnostic**, but it is no longer the definition of envelope progression.

The revised method asks:

```text
A. did the locked empirical frontier move outward?
B. did phase-aware policy realization remain within a fixed non-inferiority margin?
```

A later policy can therefore demonstrate genuine new frontier capability while still being rejected as the sole next automatic policy authority because one phase loses too much support.

---

## 4. Bootstrap: Propulsion-Ascent and Descent-Recovery experts

### 4.1 `pi_up_star`

Purpose:

- bootstrap Propulsion-Ascent behavior;
- generate launch/rising-flight trajectories;
- provide upstream continuation evidence for the first empirical Tube.

Training:

- 9,518? is not authoritative; exact transitions: **9,977,856**.

Actor SHA-256:

`f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`

### 4.2 `pi_down_star`

Purpose:

- bootstrap Descent-Recovery behavior;
- provide descent, landing, and recovery continuation evidence.

Training:

- **25,600 transitions**.

Actor SHA-256:

`7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`

Frozen expert manifest:

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### 4.3 Scientific role of the experts

The experts are not the final runtime solution. They are best interpreted as **capability probes and bootstrap data sources**.

Their value is that they reveal portions of the fixed system's achievable behavior and provide reliable two-phase continuation evidence from which a single unified policy can later be trained.

---

## 5. Bootstrap continuation fields and Tube_0

Frozen expert continuation evidence produced bootstrap `V_up` and `V_down` authorities.

These are expert-conditioned bootstrap fields only; they are not reused as later unified-policy continuation authorities.

Tube_0:

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

Composition:

```text
222 TRAIN states
  upstream   117
  downstream 105
```

Manifest SHA-256:

`c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf28?` is not valid; authoritative manifest SHA is
`c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`

Task-level meaning:

- Tube_0 is the first compact empirical training/curriculum support extracted from two-phase capability evidence;
- it is not a certified safe set;
- it gives one unified policy an initial state support from which to learn the complete maneuver.

---

## 6. Unified policy `pi_0`

Frozen authority:

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

Training:

- **10,009,600 PPO transitions**.

Identity:

- actor SHA-256: `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`
- payload SHA-256: `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`

`pi_0` established that the two expert-derived phase supports can be consumed by one Actor without runtime expert switching.

---

## 7. `C^0` and the first major expansion: Tube_1

After `pi_0` was frozen, continuation evidence was regenerated under the exact frozen unified policy and used to fit policy-conditioned `C_up^0/C_down^0`.

These fields passed the then-declared independent validation/calibration path.

Tube_1:

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

Composition:

```text
retained Tube_0 =   222
new expansion   = 2,688? no — authoritative expansion = 2,897
total           = 3,119

upstream   =   427 = 117 retained + 310 expansion
downstream = 2,692 = 105 retained + 2,587 expansion
```

Authoritative totals:

```text
retained Tube_0 = 222
expansion       = 2,897
total           = 3,119
```

Cardinality changes relative to Tube_0:

- total: **14.05x** as many support entries;
- upstream: 117 -> 427, about **3.65x**;
- downstream: 105 -> 2,692, about **25.64x**.

These are entry-count ratios, not physical state-space volume ratios.

Manifest SHA-256:

`817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`

Entries SHA-256:

`61c6796aaf4c4b1e43624c5cf06bce0d39736a6d1743c5142c6c250d23155ec9`

Task-level significance:

> JIT moved from a small expert-bootstrapped support to a much broader policy-conditioned empirical training support while retaining the original Tube_0 evidence exactly.

---

## 8. Iteration-1 unified-policy study and `pi_1`

Training directly on the enlarged Tube exposed a central learning problem: cardinality expansion can dilute replay of earlier support.

The project therefore studied retained-core replay and bounded warm-start variants. That study is closed.

Final comparison:

| policy/checkpoint | Tube_0 | regressions | upstream | downstream | boundary | groups |
|---|---:|---:|---:|---:|---:|---:|
| repair02 | **222/222** | **0** | **117/117** | **105/105** | 26/260 | 4 |
| B 1.024M | 217/222 | 5 | 112/117 | 105/105 | 33/260 | 3 |
| B 2.508M | 206/222 | 16 | 101/117 | 105/105 | 28/260 | 4 |
| B 5.0176M | 214/222 | 8 | 109/117 | 105/105 | 25/260 | 4 |
| B 7.5008M | 217/222 | 5 | 112/117 | 105/105 | 42/260 | 4 |
| B 10.0096M | 212/222 | 10 | 107/117 | 105/105 | 46/260 | 4 |

No B checkpoint simultaneously preserved all 222 Tube_0 states and exceeded repair02's boundary result under the then-current retention-first rule.

Selected engineering `pi_1`:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

Identity:

- actor SHA-256: `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`
- payload SHA-256: `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`

Historical quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 parent groups
```

Historical claim boundary:

- engineering selection: yes;
- old strict formal Iteration-1 PASS: no, because 3 baseline-reproduction mismatches remain quarantined from the old PRNG hierarchy.

### 8.1 What Iteration-1 taught us

All B core regressions were upstream while downstream remained 105/105. Boundary gains were also concentrated upstream.

This was early evidence that **frontier learning and old-behavior retention can interfere inside one reward-guided policy**, especially in the upstream phase.

The later `pi_2` result confirms that this is a recurring phenomenon, not an isolated one-off artifact.

---

## 9. `pi_1` frontier acquisition: from uninformative probes to usable two-phase evidence

The Iteration-1 frontier stage showed that continuation evidence cannot be assumed to contain useful success/failure structure merely because many candidates are sampled.

### 9.1 v1

The first frontier design failed to produce useful downstream mixed-outcome support. This was a frontier-design limitation, not proof that `pi_1` itself was incapable.

### 9.2 v2

A stronger local single-axis probe still produced downstream all-positive support.

### 9.3 v3 TRAIN

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

### 9.4 v3b CALIBRATION

Original upstream calibration was all positive, so it could not support the fixed threshold contract.

A stronger two-axis upstream CALIBRATION probe using the same already-declared three calibration parents produced:

```text
739 upstream candidates
733 positive
6 negative
3 parent groups
```

Downstream calibration remained:

```text
70 candidates
61 positive
9 negative
```

### 9.5 v3c ACCEPTANCE

Fresh acceptance evidence:

```text
upstream   516 = 511 positive + 5 negative
downstream  70 =  61 positive + 9 negative
```

The negative challenge support later became the locked boundary bank used before `pi_2` training.

---

## 10. `C^1`: architecture evidence and claim boundary

### 10.1 Original 8-unit network

Architecture:

```text
76 -> 8 tanh -> 1
625 parameters
```

Upstream calibration:

- ROC AUC: 0.66348;
- positive recall: 0.23465;
- zero accepted negatives;
- failed the original AUC >= 0.70 gate and failed accepted-positive support in every parent.

### 10.2 Standard 64x64 MLP

Architecture:

```text
76 -> 64 tanh -> 64 tanh -> 1
9,153 parameters
```

Using the same TRAIN and v3b CALIBRATION evidence:

Upstream:

- threshold: `0.9835533512239714`;
- ROC AUC: **0.6903137789904502**;
- positive recall: **0.5934515688949522**;
- accepted negatives: 0;
- every calibration parent obtained accepted positive support.

This was a practical improvement over the 8-unit network, but it still missed the original 0.70 AUC threshold by about 0.00969.

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

Widening further was therefore rejected.

### 10.4 Engineering-selected C^1

The user explicitly selected 64x64 as the engineering mainline architecture. The original upstream AUC rule was **not rewritten**.

`C_up^1`:

- AUC: `0.6903137789904502`;
- recall: `0.5934515688949522`;
- formal upstream AUC gate: false;
- explicit engineering override: true.

`C_down^1`:

- 70 calibration candidates = 61 positive + 9 negative;
- threshold: `0.015432215517145933`;
- ROC AUC: **1.0**;
- positive recall: **1.0**;
- accepted negatives: 0;
- formal calibration: PASS.

Therefore C^1 is deliberately mixed-status:

```text
upstream: engineering-selected, not original-formal AUC PASS
downstream: formal calibration PASS
```

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

upstream   =   902 = 427 + 475
downstream = 2,874 = 2,692 + 182
```

Cardinality changes:

- Tube_1 -> Tube_2 total: **+21.06%**;
- upstream: 427 -> 902, **+111.24%**;
- downstream: 2,692 -> 2,456? no — authoritative total is 2,874, **+6.76%**;
- Tube_2 contains about **17.01x** as many support entries as Tube_0.

These are support-entry counts, not geometric envelope volumes.

No CALIBRATION, ACCEPTANCE, TEST, or final-evaluation rows were embedded.

### 11.1 Tube_2 smoke

Tube-RSI smoke completed with:

- 8 upstream samples;
- 8 downstream samples;
- all finite;
- `tube_rsi_smoke = GO`;
- TEST/validation unused.

### 11.2 Role-isolation evidence

Exact-state overlaps across TRAIN/CALIBRATION/ACCEPTANCE: zero.

Parent-group overlaps: zero.

Near-observation overlap at `atol=0.01`:

```text
TRAIN <-> CALIBRATION       140
TRAIN <-> ACCEPTANCE          0
CALIBRATION <-> ACCEPTANCE  157
```

All observed near-overlap was upstream and between distinct parent groups.

Because candidate training and acceptance remained geometrically isolated, engineering continuation was allowed with an explicit non-formal claim:

```text
candidate_training_acceptance_isolation_passed = true
formal_all_role_geometric_isolation_passed = false
```

This means the current round should be described as an engineering mainline round, not an immaculate all-formal prospective round.

---

## 12. Locked `pi_1` baseline before `pi_2` training

The new locked-baseline protocol was successfully used before candidate training.

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

The acceptance boundary baseline was locked before candidate training, eliminating the historical failure mode in which a previously negative baseline was rerun later under a different PRNG hierarchy.

For the current gate:

```text
boundary baseline reproduction failure count = 0
```

This is an important protocol improvement.

---

## 13. `pi_2` training

Training run:

`JIT/runs/pi_unified/pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

Training completed:

- requested transitions: 10,009,600;
- completed transitions: 10,009,600;
- outer reset: 90% Tube / 10% natural;
- inside Tube: 75% retained Tube_1 / 25% Tube_2 newest expansion;
- fresh unified PPO job;
- no expert switching;
- no TEST or validation data.

Final reported training metrics included:

- KL mean: 0.004306;
- policy loss: 0.006256;
- value loss: 4.5473;
- SPS: about 56.8k;
- walltime: about 211 s on the local RTX 4090 D environment.

Frozen policy path used for the completed comparison:

`JIT/runs/frozen_unified/pi_2_c1_64x64_engineering_10009600_20260904/frozen_unified_policy.json`

---

## 14. `pi_1` vs `pi_2`: the result that changes the JIT story

The historical strict summary reports:

```text
iteration_accepted = false
```

That old Boolean alone is no longer a sufficient scientific summary.

### 14.1 Source Tube_1 panel

Overall:

```text
states                    3,119
pi_1 baseline successes   3,115
pi_2 candidate successes  3,002
strict regressions          115
strict improvements           2
```

Panel coverage:

```text
pi_1: 3115/3119 = 99.87%
pi_2: 3002/3119 = 96.25%
absolute global drop ≈ 3.62 percentage points
```

The phase split is more important than the global total.

Upstream:

```text
states             427
pi_1 success        423 = 99.06%
pi_2 success        312 = 73.07%
strict regressions  113
strict improvements   2
coverage drop ≈ 25.995 percentage points
```

Downstream:

```text
states             2692
pi_1 success        2692 = 100.00%
pi_2 success        2690 = 99.93%
strict regressions     2
coverage drop ≈ 0.074 percentage points
```

The global number therefore hides a severe upstream policy-realization collapse because downstream dominates Tube cardinality.

### 14.2 Locked frontier/boundary comparison

The boundary result is very different:

```text
14 locked pi_1-negative challenge states
pi_2 success = 13/14
successful parent groups = 3
minimum required groups = 2
baseline reproduction failures = 0
```

By phase:

```text
upstream   4/5
downstream 9/9
```

This is strong evidence that `pi_2` learned closed-loop behavior beyond the locked `pi_1` local frontier in both phases.

### 14.3 Correct interpretation

The correct interpretation is not:

> pi_2 failed, therefore JIT did not expand capability.

It is:

> Tube_2 training produced a policy with strong local frontier progression in both phases, but the same policy lost substantial upstream realization coverage over previously demonstrated support.

This distinction is the reason envelope evidence and policy realization are now reported separately.

---

## 15. Revised capability-progression decision implemented in code

New stable analysis capability:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

CLI:

`JIT/cli/analyze_capability_progression.py`

### 15.1 Frontier progression

Prospective PASS requires:

- zero baseline-reproduction mismatch;
- nonzero candidate boundary success;
- sufficient independent parent-group support;
- candidate boundary success in both upstream and downstream.

This supports an **empirical local frontier progression** claim only.

### 15.2 Policy realization retention

The current engineering proxy is fixed locked-panel success coverage.

Prospective method-level non-inferiority margins:

```text
maximum allowed global Tube coverage drop = 5 percentage points
maximum allowed per-phase coverage drop   = 10 percentage points
```

These margins allow modest stochastic/single-rollout variation while preventing a large phase-specific collapse from being hidden by an imbalanced Tube.

Zero strict paired regressions are **not** required by the new policy-authority criterion.

### 15.3 Current `pi_2` under the revised semantics

Retrospective classification:

```text
empirical_envelope_expansion_observed = true
global policy-realization margin       = pass
downstream phase margin                = pass
upstream phase margin                  = fail
candidate_policy_authority_eligible    = false
decision = envelope_progressed_but_candidate_policy_coverage_degraded
```

Because this method interpretation was revised **after** the current `pi_2` result had been observed, the candidate is not retroactively promoted to a formal prospective PASS.

The evidence is preserved; formal selection is not rewritten.

---

## 16. What has actually improved at the task level

### Improvement 1 — much broader empirical support

```text
Tube_0   222
Tube_1 3,119
Tube_2 3,776
```

Tube_2 contains about 17.01 times as many retained/qualified TRAIN support entries as Tube_0.

This is not “17x physical jump range,” but it is a large increase in the diversity and quantity of empirically supported training/continuation states.

### Improvement 2 — usable two-phase frontier evidence

The frontier pipeline evolved from all-positive/uninformative banks to mixed success/failure evidence in both phases.

This matters because a capability boundary cannot be identified from an all-success sample cloud.

### Improvement 3 — direct new unified-policy frontier capability

`pi_2` succeeds on **13 of 14** states locked as failures for `pi_1`, spanning three parent groups and both phases.

This is direct task-level evidence that the JIT curriculum can create new closed-loop jump/recovery capability beyond the previous policy's local frontier.

### Improvement 4 — stronger experimental protocol

The `pi_1` baseline was locked before `pi_2` training, and the later comparison had zero boundary baseline-reproduction mismatch.

This removes the historical PRNG debt that affected the old `pi_0 -> pi_1` quickcheck.

### Improvement 5 — clearer scientific object

The project now explicitly separates:

- physical/task feasibility;
- cumulative empirical capability evidence;
- current-policy realization coverage;
- curriculum/Tube support.

This is a more defensible research story than treating one policy's single rollout on every state as the definition of robot capability.

---

## 17. What has not yet been solved

### 17.1 JIT has not identified the exact physical limit `F*`

Tube_2 and frontier results are empirical evidence, not proof of maximum jump distance, height, landing region, or viability.

### 17.2 Continuation fields remain policy-conditioned

`C^k(s)` answers continuation under one frozen `pi_k`, not existential controllability over all admissible controllers.

A failed `C^k` state may still be physically feasible under another controller.

### 17.3 The latest unified policy is not goal-conditioned

The current observation does not specify what jump outcome is requested. This likely contributes to interference when one Actor is asked to realize a broad collection of upstream behaviors.

### 17.4 Current coverage is a one-rollout fixed-panel proxy

A stronger future evaluator should predeclare multiple seeds per state and estimate success rate or confidence intervals rather than treating one rollout as the whole stochastic policy response.

### 17.5 Current automatic frontier uses the latest selected policy as the probe

If the long-term scientific target is increasingly system-oriented empirical capability, future discovery may benefit from a frozen **policy archive** or region-wise best probe.

This would be discovery-time only and would not change the requirement that deployment use one unified Actor.

---

## 18. Revised JIT story

A concise research narrative is:

> JIT couples empirical capability identification with just-in-time curriculum generation. Under fixed robot dynamics, frozen experts and unified policies act as capability probes. Successful real-dynamics continuation evidence accumulates into monotonic empirical support, while states near the current success/failure transition generate the next curriculum. Newly trained unified policies are evaluated both for frontier progression and for how much cumulative capability support they can realize. The latest policy is therefore an implementation candidate for the discovered capability, not the definition of the physical capability itself.

The loop is better written as:

```text
capability probe policy / archive
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

The branch now contains reusable automation for future `k -> k+1` rounds:

1. newest-shell frontier planning;
2. TRAIN acquisition/labeling;
3. CALIBRATION acquisition/labeling;
4. ACCEPTANCE acquisition/labeling;
5. `C^k` fit/calibration;
6. `Tube_(k+1)` construction;
7. Tube-RSI smoke;
8. role-isolation audit;
9. pre-candidate baseline lock;
10. candidate training;
11. freeze;
12. locked paired evaluation;
13. **capability-progression analysis**;
14. prospective policy selection only when frontier progression and phase-aware policy realization both pass.

### 19.2 Why the current pi_1 -> pi_2 round was not fully automatic

The generic workflow existed, but this round required explicit engineering/scientific interventions:

- initial frontier panels did not produce informative downstream support;
- upstream calibration needed stronger v3b acquisition;
- the original `C_up^1` model failed;
- 64x64 was selected as an engineering architecture on reused data;
- upstream AUC remained 0.6903 < 0.70 and required an explicit engineering continuation override;
- strict all-role near-observation isolation failed and was replaced by an explicit engineering continuation record only after TRAIN <-> ACCEPTANCE near-overlap was confirmed zero.

Therefore it is inaccurate to claim that the current `pi_1 -> pi_2` round was one clean untouched automatic execution.

The correct statement is:

> The production DAG and reusable stage machinery exist, but the current round crossed explicit engineering decision points. Future rounds should use the revised capability-progression decision prospectively and stop rather than silently invent another override.

---

## 20. Current exact project position

Completed:

```text
phase experts                    DONE
Tube_0                           DONE
pi_0                             DONE
C^0                              DONE
Tube_1                           DONE
pi_1 engineering authority       DONE
pi_1 frontier roles              DONE
C^1 engineering selection        DONE
Tube_2                           DONE
Tube_2 smoke                     GO
role-isolation engineering record DONE
pi_1 baseline lock               DONE
pi_2 training                    DONE
pi_2 freeze                      DONE
pi_1 vs pi_2 locked comparison   DONE
capability-semantics revision    CODED
```

Current scientific position:

```text
pi_2 demonstrated strong frontier progression
BUT
pi_2 is not retrospectively selected as formal next authority because
upstream policy realization dropped from 99.06% to 73.07% on the locked Tube_1 panel.
```

No `pi_3` work should begin yet.

---

## 21. What should happen next

### Step 1 — create the retrospective capability decision artifact for current pi_2

Use the new analyzer on the already-completed gate summary with `--retrospective`.

Expected semantic result:

```text
empirical_envelope_expansion_observed = true
candidate_policy_authority_eligible = false
retrospective_analysis = true
```

This records the new interpretation without rewriting history or selecting `pi_2`.

### Step 2 — stop treating replay-ratio tuning as the automatic answer

The current result should not automatically trigger 75/25 -> 90/10 replay. The deeper issue is that the upstream policy is being asked to realize many behaviors without an explicit desired jump target.

A replay repair can remain a later ablation, but it should not be the default scientific next move.

### Step 3 — define the next representation question

The strongest next research question is whether one unified policy should receive a low-dimensional **jump intent / goal condition**, for example:

- desired horizontal travel;
- desired apex/clearance;
- desired landing region;
- desired recovery speed/posture;
- a normalized or learned behavior code derived from Tube/frontier structure.

This preserves one-policy deployment while allowing the policy to express different behaviors intentionally.

### Step 4 — upgrade policy evaluation from one rollout to success probability

For future prospective gates, lock multiple policy seeds per state before candidate training and report phase-wise success rate/confidence estimates.

### Step 5 — consider a discovery-time policy archive

To approach system capability rather than latest-policy capability, future frontier acquisition can maintain frozen probe policies and use them only for scientific discovery.

Conceptually:

```text
E_k = union of successful capability evidence demonstrated by frozen probes up to k
```

The archive would not be used for runtime switching.

### Step 6 — only then declare the next candidate/pi_3 method

Before launching another automatic iteration, decide whether the next method version is:

1. same representation + a prospectively declared training repair;
2. goal-conditioned unified policy;
3. archive-assisted capability discovery + separate unified realization.

The recommended scientific focus is options 2 and 3, because they directly address the conceptual gap exposed by `pi_2`.

---

## 22. Publication and claim boundary

Supported now:

- phase experts successfully bootstrap a unified JIT pipeline;
- policy-conditioned Tube construction can greatly enlarge empirical training support;
- Tube_2 retains all prior Tube evidence and adds 657 evidence-backed TRAIN states;
- `pi_2` demonstrates strong local frontier progression: 13/14 successes across three parent groups and both phases;
- the locked-baseline protocol eliminates the historical boundary reproduction mismatch in the current round;
- cumulative empirical capability progression and single-policy realization are measurably different quantities;
- upstream policy interference is a recurring bottleneck.

Not supported:

- Tube_2 is the true physical maximum jumping envelope;
- JIT has computed a viability kernel or certified safe set;
- `C_up^1` formally passed the original AUC contract;
- the current `pi_1 -> pi_2` round was fully prospective and automatic from start to finish;
- `pi_2` is formally selected under the newly revised criterion;
- final JCE/JEL has been measured;
- failure of the latest policy proves physical infeasibility.

---

## 23. Authoritative artifact map

### Experts

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

### pi_1

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600/frozen?` is not authoritative; use:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

### Iteration-1 -> 2 work root

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3`

### C^1 engineering selection

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3/continuation_C1_standard_mlp64x64_engineering_selected_v1`

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

### pi_2 training

`JIT/runs/pi_unified/pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

### pi_2 frozen policy

`JIT/runs/frozen_unified/pi_2_c1_64x64_engineering_10009600_20260904/frozen_unified_policy.json`

### pi_1 -> pi_2 locked gate

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3/pi_1_to_pi_2_gate_c1_64x64_engineering/summary.json`

---

## 24. One-sentence JIT definition going forward

> **JIT is an iterative real-dynamics capability-discovery and just-in-time curriculum framework that accumulates empirical jump-capability evidence under fixed robot dynamics, uses the current frontier to train a single unified policy, and separately measures frontier progression and how much of the cumulative capability that policy can realize.**
