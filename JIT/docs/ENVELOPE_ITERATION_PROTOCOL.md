# JIT Policy-Conditioned Soft-Tube Envelope Iteration Protocol

## Status — 2026-09-02

This document defines the active scientific contract for iterative empirical
jumping-capability-envelope expansion.

The first Tube_1 policy candidate was rejected because core preservation failed.
A retained-core replay repair was predeclared, fresh frozen-`pi_0` acceptance
evidence was generated, and the repaired iteration-1 candidate has now completed
training and freeze.

**The current stage is repaired iteration-1 acceptance.** Training completion or
freezing does not accept an iteration. The repaired `pi_1` must still pass both
core preservation and fresh boundary gain before it becomes the next authority.

Final TEST/JCE/JEL remains untouched.

## 1. Research objective

The target is one unified policy whose empirically demonstrated jumping
capability envelope expands across iterations under fixed task physics and action
semantics.

The scientific chain is:

```text
frozen phase experts
  -> bootstrap V_up / V_down
  -> Tube_0
  -> unified pi_0
  -> freeze pi_k
  -> real-dynamics TRAIN acquisition under pi_k
  -> pi_k continuation labels
  -> policy-conditioned C_up^k / C_down^k
  -> independent fresh validation/calibration
  -> core-retaining Tube_(k+1)
  -> unified pi_(k+1)
  -> freeze pi_(k+1)
  -> paired core-preservation + boundary-gain gate
  -> FAIL: preserve, diagnose, predeclare repair
  -> PASS: accept next authority and repeat
  -> only after iteration stopping: final frozen-policy JCE/JEL
```

The Soft Tube is training guidance/support/curriculum. It is not a certified safe
set, viability kernel, invariant-set proof, or formal reachability certificate.

Natural cold-start failure outside the declared jump-capability state domain is a
diagnostic, not the central capability gate.

## 2. Immutable task contract

Unless a new research question is explicitly opened, preserve:

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: steer / rear-wheel drive / hip / knee
- reward semantics
- snapshot semantics
- no expert switching in unified policy operation
- validation/TEST isolation

## 3. Policy-conditioned continuation authority

`V_up/V_down` are bootstrap expert-conditioned authorities used to construct
Tube_0. They must not be silently reused as continuation authority for later
unified policies.

For every frozen unified policy `pi_k`, continuation is policy-conditioned:

`C(s | pi_k)`.

Therefore the same state may fail under `pi_k` and succeed under `pi_(k+1)`.
Every continuation dataset/model/report must bind the exact frozen policy identity
and protocol identity.

## 4. Real-dynamics acquisition only

Expansion evidence must come from authoritative dynamics. Do not directly widen
coordinate ranges or manually mutate `qpos/qvel` to manufacture favorable states.

Allowed acquisition mechanisms include:

- successful frozen-policy trajectories;
- states reached just outside current Tube support under real dynamics;
- bounded predeclared action perturbations from audited real snapshots;
- other explicitly predeclared real-dynamics probes.

The acquisition implementation may use configurable sparse action directions.
`active_action_dimensions=1` preserves historical one-axis behavior; higher
values may systematically enumerate coupled directions. The direction family is
an experimental protocol field, not a new iteration-specific production module.

The acquisition implementation must be iteration-generic: `source_iteration=k`
and `candidate_iteration=k+1`.

## 5. Data-role isolation

Each iteration separates:

- expansion TRAIN: may train `C^k` and contribute to `Tube_(k+1)`;
- expansion validation: may calibrate/select declared continuation decisions but
  must never enter a Tube;
- final TEST/JCE/JEL: untouched until the final policy and stopping decision are
  frozen.

Consumed validation or acceptance evidence is consumed for tuning/selection.
Parent trajectories and near-duplicate states must be excluded across roles where
required by the predeclared protocol. Seed disjointness alone is insufficient.

## 6. Fresh acceptance bank before repaired-candidate training

When a completed candidate gate informs a method repair, the old boundary bank is
consumed selection evidence. A repaired candidate requires new non-final boundary
acceptance evidence generated without candidate-policy information.

For the iteration-1 repair, fresh-bank readiness was predeclared as at least:

- 10 frozen-`pi_0` negative states per phase; and
- 3 independent negative parent groups per phase.

These readiness rules were not changed after outcomes were seen.

The first two single-axis probes failed readiness:

- 659-state support-wide probe: 12 negatives, all upstream, 2 upstream parent
  groups, downstream 0;
- 1,272-state stronger single-axis probe: 58 negatives, all upstream, still 2
  upstream parent groups, downstream 0.

Those results were preserved and used only to revise the baseline-only acquisition
method. They were not used to inspect the repaired candidate.

The final two-axis acquisition produced 3,720 fresh TRAIN candidates and, after
frozen-`pi_0` labeling, a locked 260-state negative bank:

- upstream: 246 negatives across 4 parent groups;
- downstream: 14 negatives across 5 parent groups;
- Tube_1 overlap: 0.

Therefore pre-training fresh-bank readiness passed before repaired `pi_1` was
trained.

## 7. Labeling execution and engineering sharding

Scientific labeling semantics are defined by the full candidate catalog, exact
frozen policy, deterministic mode, protocol seed, continuation horizon and
success/failure rule.

Execution may be partitioned across multiple fresh processes **only as an
engineering implementation detail** when all of the following remain unchanged:

- exact candidate set and order;
- exact frozen policy;
- seed;
- horizon;
- snapshot semantics;
- physics;
- label rule;
- acceptance criteria.

All shards must be merged back into one logical label artifact in original
catalog order before scientific readiness or bank locking is evaluated.

This rule was required on 2026-09-02 because long single-process frozen-`pi_0`
labeling exhausted CUDA/Warp allocator state. Four sequential processes of 930
candidates completed the same 3,720-state logical job. Process sharding must not
be used for adaptive candidate selection or post-hoc threshold tuning.

## 8. Core-retaining Tube construction

The structural rule is:

```text
Tube_(k+1) = retained Tube_k core union accepted TRAIN expansion_k
```

Retaining entries structurally is not sufficient. The runtime must also declare
sampling probability mass so expansion cardinality cannot starve the old core.

After the first iteration-1 rejection, the repaired Tube-RSI sampling contract is:

```text
phase selection
  -> retained source core vs current expansion
  -> entry within selected source
```

For repaired iteration 1:

- upstream/downstream phase mixture: 0.5 / 0.5;
- inside each phase: retained core 0.5 / expansion 0.5;
- retained core sampled uniformly;
- expansion sampled by the existing value/continuation weighting;
- outer reset mixture remains 0.9 Tube / 0.1 natural.

Thus all-episode reset mass is 45% retained core, 45% expansion and 10% natural.
The 0.5/0.5 replay split is a predeclared method repair, not a hyperparameter sweep
against acceptance outcomes.

## 9. Policy improvement

Training `pi_(k+1)` is a separate predeclared experiment.

For the repaired iteration-1 candidate, the following stayed fixed relative to
the rejected candidate:

- Tube_1 support;
- PPO hyperparameters and exact 10,009,600-transition budget;
- seed 821101;
- fresh actor/critic/optimizer initialization;
- reward and physics;
- action semantics;
- natural reset probability;
- phase mixture.

Only the within-Tube retained-core replay contract changed.

Because the policies were fresh-initialized, the first candidate's loss of core
competence is described as **core-support under-replay / capability regression
under the expanded training distribution**, not classical forgetting from a
warm-started `pi_0`.

## 10. Paired acceptance gate

A larger Tube or higher PPO reward does not prove envelope expansion.

Before `pi_(k+1)` becomes the accepted authority, one locked paired audit evaluates
frozen `pi_k` and frozen `pi_(k+1)` on the exact same states.

### Core preservation

- bank: every declared `Tube_k` source-core state;
- baseline and candidate use the same deterministic continuation runtime;
- PASS only when baseline-success -> candidate-failure regressions equal zero;
- baseline must have successful states in each phase so the gate cannot pass
  vacuously.

For repaired iteration 1 the structural core bank is all 222 Tube_0 entries.

### Boundary reproduction and gain

- bank: a fresh locked frozen-`pi_k` continuation-negative TRAIN challenge bank;
- no state admitted to `Tube_(k+1)` may appear in the boundary bank;
- all baseline-negative outcomes must reproduce under the gate runtime;
- candidate must convert failures to successes in at least 2 distinct parent
  groups.

The `>=2 parent groups` rule is the original paired-gate rule. It must not be
raised or lowered after repaired-candidate outcomes are observed.

For repaired iteration 1 the formal boundary bank is the fresh 260-state two-axis
bank, not the consumed old 56-state bank.

### Gate isolation

During the paired gate:

- training transitions: 0;
- expert switching: false;
- validation data: false;
- TEST/final JCE/JEL data: false.

The locked bank must be written/bound before candidate outcomes are inspected.
A completed FAIL is immutable scientific evidence and must never be converted to
PASS by changing bank, threshold, replay ratio, reward, PPO or acceptance rules.

## 11. Current completed chain

```text
frozen experts
  -> Tube_0 (222)
  -> frozen pi_0
  -> pi_0-conditioned C^0 with fresh validation
  -> Tube_1 (3,119 = 222 core + 2,897 expansion)
  -> first pi_1 candidate
  -> old paired gate: boundary PASS / core FAIL
  -> zero-interaction replay diagnosis
  -> retained-core replay repair predeclared
  -> fresh readiness probe 1 FAIL
  -> fresh readiness probe 2 FAIL
  -> acquisition generalized to two-axis sparse directions
  -> 3,720-state two-axis acquisition
  -> single-process labeling engineering OOM
  -> four-process equivalent labeling + merge
  -> 260-state fresh acceptance bank PASS
  -> repaired pi_1: 10,009,600 transitions
  -> repaired pi_1 frozen
  -> CURRENT: paired repaired-pi1 acceptance gate
```

Repaired formal run:

`JIT/runs/pi_unified/pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902`

Frozen candidate:

`JIT/runs/frozen_unified/pi_1_core_replay50_10009600_20260902/frozen_unified_policy.json`

Known exact local identities:

- final checkpoint payload SHA-256: `ea93a534c2c6bb3bf145684cbea82df94fefa2df8099dcdcdd9492bd8007e205`
- frozen manifest file SHA-256: `d5a1658530d475a67264aa5c621283d71c823200dbee6068f93413b93d06b7a8`

## 12. Immediate order from here

1. Preserve all consumed readiness probes, OOM attempts, merged fresh bank and
   repaired training/freeze artifacts.
2. Bind the existing generic paired gate to:
   - frozen baseline `pi_0`;
   - frozen repaired `pi_1`;
   - all 222 Tube_0 core states;
   - the locked fresh 260-state two-axis negative bank;
   - the correct acquisition snapshot provenance root.
3. Run zero-interaction preflight/identity checks.
4. Execute exactly one formal paired acceptance gate.
5. If core PASS and boundary PASS: accept iteration 1 and authorize frozen-`pi_1`
   continuation evidence, fresh validation and Tube_2 construction.
6. If either gate FAILS: preserve and diagnose. Do not tune the same gate.
7. Generalize remaining iteration-0-specific continuation/Tube code before
   unattended `pi_1 -> pi_2` execution.
8. Keep final TEST/JCE/JEL untouched until iteration stopping and final policy
   selection are frozen.

## 13. Convergence and final empirical envelope

Iteration stopping criteria must be declared before the final stopping decision.
Possible non-final audit quantities include support gain, boundary movement,
continuation-success improvement and repeated no-material-gain rounds, always
subject to core preservation.

Only the final selected frozen unified policy may be evaluated on the untouched
final envelope bank. The resulting claim is an **empirical policy-conditioned
jumping capability envelope**, not a formal safety or viability proof.
