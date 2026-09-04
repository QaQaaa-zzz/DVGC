# JIT Capability-Progression Iteration Protocol

## Status — 2026-09-04

This document defines the active scientific contract for future JIT iterations
after completion of the engineering `pi_1 -> C^1 -> Tube_2 -> pi_2` round.

The central revision is:

> cumulative empirical capability progression and latest-policy behavioral
> retention are related but distinct quantities.

Historical strict zero-regression gates remain valid diagnostics and
reproducibility evidence. They are no longer the sole definition of envelope
progression.

Final TEST/JCE/JEL remains untouched.

---

## 1. Research objects

### 1.1 Physical/task feasibility `F*`

Conceptual set of states from which there exists at least one admissible control
behavior that can complete the fixed jump task.

JIT does not prove or exactly compute `F*`.

### 1.2 Cumulative empirical capability evidence `E_k`

States/regions supported by successful real-dynamics continuation evidence from
frozen experts or unified policies up to iteration k.

A Tube is structured TRAIN support/curriculum derived from this evidence. Tube
cardinality is not geometric state-space volume.

### 1.3 Single-policy realization `R(pi_k, E_k)`

How much of cumulative support one unified policy realizes on a locked evaluation
panel.

A later policy failing one earlier rollout does not erase historical capability
provenance. But a large phase-specific coverage collapse can make that policy
unsuitable as the sole authority for the next automatic round.

---

## 2. Scientific chain

```text
frozen phase experts
  -> bootstrap V_up / V_down
  -> Tube_0
  -> unified pi_0
  -> selected frozen probe pi_k
  -> predeclare frontier roles
  -> real-dynamics frontier acquisition
  -> pi_k-conditioned continuation evidence
  -> C_up^k / C_down^k
  -> core-retaining Tube_(k+1)
  -> lock pi_k evaluation baseline
  -> train/freeze unified pi_(k+1)
  -> locked paired evaluation
  -> capability-progression decision
       A. empirical frontier progression
       B. phase-aware policy realization retention
  -> if A+B pass prospectively: select pi_(k+1)
  -> otherwise: preserve evidence and stop for a method decision
  -> after declared stopping only: final JCE/JEL evaluation
```

The newest unified policy is a capability probe and realization candidate. It is
not the definition of physical feasibility.

---

## 3. Immutable task contract

Preserve unless a new research question explicitly changes the task:

- XML: `assets/orange_bike_4kg_horizontal.xml`;
- XML SHA-256:
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`;
- payload: 2 kg;
- control: 50 Hz;
- hip/knee torque limits: +/-50 Nm;
- action order: `[steer, rear-wheel drive, hip, knee]`;
- reward semantics;
- snapshot semantics;
- no expert switching in unified runtime;
- development-role isolation;
- final TEST/JCE/JEL isolation.

Do not create a replacement XML merely to fix the historical filename.

---

## 4. Bootstrap expert semantics

`pi_up_star` and `pi_down_star` are capability probes/data sources for bootstrap.
They are not final runtime controllers.

`V_up/V_down` are expert-conditioned continuation authorities for Tube_0 only and
must not be silently reused as later unified-policy continuation fields.

---

## 5. Policy-conditioned continuation fields

For frozen `pi_k`:

```text
C^k(s) = empirical continuation score under exact frozen pi_k
```

`C^k` is a proposal/filtering tool for frontier/Tube construction. It is not an
existential statement that no other controller can solve a state when `pi_k`
fails.

Therefore:

- the same state may fail under one policy and succeed under another;
- PPO critic/value is not `C^k`;
- cumulative capability evidence must not be collapsed into latest-policy
  continuation alone.

---

## 6. Real-dynamics acquisition

Expansion/frontier evidence must be reached through authoritative dynamics.
Do not manufacture favorable states by directly widening qpos/qvel ranges.

Allowed mechanisms include:

- successful frozen-policy trajectories;
- states reached just outside current support;
- bounded predeclared action perturbations from audited snapshots;
- other explicitly predeclared real-dynamics probes.

The current generic automatic frontier uses the newest Tube expansion shell and
does not silently fall back to the full Tube.

A future archive-assisted discovery method may choose among frozen probe policies,
but that would be a new declared method version and would remain discovery-time
only, not runtime expert switching.

---

## 7. Outcome-blind data roles

For prospective automatic iterations:

- `TRAIN`: may fit `C^k` and contribute qualifying Tube expansion;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: candidate-blind development frontier comparison only;
- final TEST/JCE/JEL: untouched.

Role assignment must be outcome-blind and parent-group disjoint before outcomes
are observed.

CALIBRATION and ACCEPTANCE rows never enter a Tube.

The current Iteration-1 -> 2 round contains an explicit engineering
near-observation-isolation exception. That historical artifact does not silently
weaken the prospective generic rule.

---

## 8. Tube construction

Structural rule:

```text
Tube_(k+1)
  = every Tube_k entry retained exactly
  + qualifying logical-TRAIN expansion
```

An expansion row must:

- have positive continuation evidence under the declared policy probe;
- pass the frozen phase continuation threshold;
- reproduce state/snapshot identity;
- not duplicate existing Tube support;
- originate from TRAIN, never CALIBRATION/ACCEPTANCE/TEST.

Structural retention preserves cumulative evidence/provenance. It does **not**
require the newest policy to reproduce every old state in one rollout.

---

## 9. Policy training contract

Current generic baseline recipe remains:

```text
outer reset:
  90% Tube RSI
  10% natural

inside Tube RSI:
  75% retained source Tube_k
  25% newest expansion
```

For pi_2 this correctly treated all 3,119 Tube_1 states as retained source
support.

The 75/25 choice is a training recipe, not an acceptance definition. A candidate
failure must not automatically trigger replay-ratio tuning.

---

## 10. Locked evaluation baseline

Before prospective candidate training:

- lock the source-Tube panel under selected `pi_k`;
- bind exact states, phases, and deterministic evaluation seeds;
- lock acceptance/frontier challenge identity and baseline evidence;
- do not reroll a previously locked baseline boundary under another PRNG hierarchy
  after candidate training.

The current pi_1 -> pi_2 round confirmed this removes the historical boundary
reproduction mismatch: reproduction failures were zero.

---

## 11. Capability-progression decision v1

Implementation:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

CLI:

`JIT/cli/analyze_capability_progression.py`

### 11.1 Empirical frontier progression

Prospective PASS requires:

1. zero baseline boundary reproduction mismatch;
2. nonzero candidate frontier success;
3. successes across at least the predeclared minimum number of parent groups;
4. at least one candidate success in both upstream and downstream.

This supports an empirical **local frontier progression** claim only. It does not
prove the physical feasibility limit and does not require zero strict core
regressions.

### 11.2 Policy realization retention

The v1 engineering proxy is success coverage on the locked full source-Tube
panel.

Method-level non-inferiority margins:

```text
maximum global coverage drop = 0.05
maximum per-phase coverage drop = 0.10
```

Both must pass.

The phase rule prevents a large upstream/downstream collapse from being hidden by
Tube cardinality imbalance.

### 11.3 Authority eligibility

A candidate becomes the sole automatic `pi_(k+1)` authority only when:

```text
frontier progression PASS
AND
policy realization retention PASS
```

Zero paired regressions are not required. The strict regression count remains a
diagnostic.

### 11.4 Retrospective analyses cannot select policies

If capability semantics are revised after candidate outcomes are observed, run:

```text
--retrospective
```

A retrospective artifact may describe evidence but may not formally select the
candidate. `select_iteration_policy.py` rejects retrospective capability
decisions.

This prevents post-hoc method changes from becoming fake prospective evidence.

---

## 12. Current pi_2 motivating result

Source Tube_1 panel:

```text
pi_1 baseline: 3115/3119 = 99.87%
pi_2:          3002/3119 = 96.25%
```

Global drop is about 3.62 percentage points and would pass the v1 global margin.

Phase split:

```text
upstream:
  pi_1 423/427 = 99.06%
  pi_2 312/427 = 73.07%
  drop ≈ 25.995 percentage points -> FAIL phase margin

downstream:
  pi_1 2692/2692 = 100.00%
  pi_2 2690/2692 = 99.93%
  drop ≈ 0.074 percentage points -> PASS phase margin
```

Locked frontier challenge:

```text
pi_2 13/14
3 parent groups
upstream 4/5
downstream 9/9
baseline reproduction failures 0
```

Revised retrospective interpretation:

```text
frontier progression = PASS
policy realization = FAIL because upstream coverage collapsed
candidate authority eligible = false
```

Preserve this as evidence that JIT can expand local capability while one
reward-guided policy still suffers representation/interference limits.

---

## 13. Current C^1 engineering claim boundary

The completed current round is not a clean all-formal prospective round.

64x64 `C_up^1`:

- AUC `0.6903137789904502`;
- recall `0.5934515688949522`;
- original AUC >= 0.70 rule remains false;
- explicit engineering selection.

64x64 `C_down^1`:

- AUC 1.0;
- recall 1.0;
- formal calibration PASS.

Tube_2 was constructed under an explicit engineering C^1 override. Do not
describe C^1 as a clean all-phase formal pass.

---

## 14. Automatic workflow contract

Future prospective workflow:

```text
prepare frontier plan
  -> TRAIN
  -> CALIBRATION
  -> ACCEPTANCE
  -> fit/calibrate C^k
  -> Tube_(k+1)
  -> Tube-RSI smoke
  -> role isolation
  -> lock pi_k evaluation baseline
  -> train pi_(k+1)
  -> freeze pi_(k+1)
  -> locked paired evaluation
  -> capability-progression analysis
  -> select pi_(k+1) only if frontier + realization pass
```

The runner remains scientifically non-adaptive:

- workflow config SHA is immutable after state creation;
- completed stages are revalidated before reuse;
- failures stop the workflow;
- the runner never changes reward, replay ratio, PPO hyperparameters, network
  architecture, physics, frontier panel, continuation threshold, or capability
  margins to force progress;
- final TEST/JCE/JEL is never part of the iteration workflow.

The current pi_1 -> pi_2 history required explicit engineering interventions and
is not evidence of fully hands-off automation.

---

## 15. Selection semantics and backward compatibility

Historical repair02/pi_1 selection remains reproducible through the strict
zero-regression path. Do not rewrite it using the new criterion.

Future selection should provide:

```text
--gate-summary <locked paired summary>
--capability-decision <prospective capability-progression decision>
```

Selection must fail if the decision is retrospective or candidate policy
authority eligibility is false.

---

## 16. Next method question

Do not launch pi_3 automatically.

The pi_2 result suggests a deeper bottleneck than replay quantity: the unified
policy is not explicitly conditioned on which jump behavior it should realize.

Priority next decisions:

1. **goal-/intent-conditioned unified policy** while retaining one runtime Actor;
2. **multi-seed success probability / confidence evaluation** instead of one
   rollout per state;
3. **discovery-time frozen policy archive** to approximate system capability using
   cumulative successful probes without runtime policy switching;
4. only then decide whether a same-representation replay repair is worth another
   formal candidate.

Any method revision must be predeclared before new candidate outcomes are
inspected.

---

## 17. Stopping and final JCE/JEL

Possible project-level stopping signals include:

- frontier progression saturation;
- negligible new evidence support;
- repeated inability of a unified realization to cover accumulated capability
  without phase collapse;
- reaching a declared physical/task target;
- resource/diminishing-return limits.

Only after method/stopping decisions are frozen should final selected-policy and/or
capability-discovery outputs be evaluated on untouched final TEST/JCE/JEL evidence.

Final claims must distinguish:

- empirical capability evidence;
- policy realization coverage;
- any final policy-conditioned JCE/JEL;
- the unproven physical feasibility limit `F*`.
