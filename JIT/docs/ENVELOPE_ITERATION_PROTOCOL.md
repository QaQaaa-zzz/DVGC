# JIT Policy-Conditioned Soft-Tube Envelope Iteration Protocol

## Status — 2026-09-03

This document defines the active scientific contract for iterative empirical
jumping-capability-envelope expansion.

Iteration-1 policy initialization/replay ablations are complete. `repair02` is
selected as the **engineering pi_1 authority** because it is the only tested
candidate that preserved all 222 Tube_0 core states while adding positive
boundary capability.

The historical repair02 quick gate is **not** a publication-level strict PASS:
3 locked baseline-negative states changed outcome under the historical paired
re-roll PRNG protocol. That mismatch is preserved as historical protocol debt;
it is not rewritten away.

The active mainline is now:

```text
selected pi_1
  -> pi_1-conditioned frontier evidence
  -> C^1
  -> Tube_2
  -> pi_2
```

For future rounds, a pre-candidate locked-baseline gate eliminates the historical
negative-reproduction mismatch by never re-rolling a locked baseline boundary
outcome under a different PRNG hierarchy.

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
  -> freeze/select pi_k
  -> predeclare outcome-blind frontier roles
  -> real-dynamics frontier acquisition under pi_k
  -> pi_k continuation labels
  -> fit C_up^k / C_down^k on TRAIN only
  -> calibrate thresholds on disjoint CALIBRATION only
  -> core-retaining Tube_(k+1)
  -> isolate/audit TRAIN vs CALIBRATION vs ACCEPTANCE
  -> lock pi_k acceptance baseline before candidate training
  -> unified pi_(k+1)
  -> freeze pi_(k+1)
  -> strict locked-baseline core-preservation + boundary-gain gate
  -> FAIL: preserve and diagnose
  -> PASS: select next authority and repeat
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
- TRAIN/CALIBRATION/ACCEPTANCE role isolation
- TEST/final isolation

## 3. Policy-conditioned continuation authority

`V_up/V_down` are bootstrap expert-conditioned authorities used to construct
Tube_0. They must not be silently reused as continuation authority for later
unified policies.

For every selected frozen unified policy `pi_k`, continuation is
policy-conditioned:

`C(s | pi_k)`.

Therefore the same state may fail under `pi_k` and succeed under `pi_(k+1)`.
Every continuation dataset/model/report must bind the exact frozen policy actor,
payload, XML, source Tube, and protocol identity.

PPO critic/value is not `C^k`.

## 4. Real-dynamics acquisition only

Expansion evidence must come from authoritative dynamics. Do not directly widen
coordinate ranges or manually mutate `qpos/qvel` to manufacture favorable states.

Allowed acquisition mechanisms include:

- successful frozen-policy trajectories;
- states reached just outside current Tube support under real dynamics;
- bounded predeclared action perturbations from audited real snapshots;
- other explicitly predeclared real-dynamics probes.

The generic automatic frontier uses the **newest expansion shell of Tube_k** as
its parent pool. It intentionally does not fall back to the full Tube when the
newest shell is absent or lacks sufficient phase/group support. Such a failure
requires a new parent-generation decision rather than silent protocol widening.

The current fixed automatic probe panel uses the stable real-dynamics boundary
acquisition capability. Probe strengths/durations/seeds and role assignment are
committed before outcomes are observed.

## 5. Outcome-blind data-role isolation

For automatic k >= 1 iterations, every frontier parent is assigned before
outcomes to one of three logical roles:

- `TRAIN`: the only role that may fit `C^k` and contribute expansion candidates
  to `Tube_(k+1)`;
- `CALIBRATION`: threshold calibration only; rows never enter TRAIN or a Tube;
- `ACCEPTANCE`: candidate-blind pi_k boundary audit only; rows never train C^k,
  calibrate C^k, or enter a Tube.

Parent-group disjointness is required across roles. Seed disjointness alone is
not sufficient. The role-isolation audit must report zero prohibited overlap
before candidate-policy training starts.

Final TEST/JCE/JEL is a fourth, untouched role and is not part of the iteration
workflow.

## 6. C^k fitting and calibration

Later iterations do not reopen continuation architecture search unless a new
scientific question is explicitly declared.

The generic continuation stage:

1. fits the frozen selected tiny/shared continuation architecture on logical
   TRAIN rows only;
2. requires both outcome classes where demanded by the implementation contract;
3. calibrates one phase threshold from disjoint CALIBRATION evidence;
4. records model, normalization, threshold, source-policy and source-Tube
   identities;
5. authorizes Tube construction only after its fixed calibration contract passes.

A calibration failure stops the workflow. Do not inspect the failure and then
silently change architecture, AUC rule, threshold objective, role split, or data
membership inside the same predeclared round.

## 7. Core-retaining Tube construction

The structural rule is:

```text
Tube_(k+1)
  = every Tube_k entry retained exactly
  + qualifying logical-TRAIN expansion states
```

A candidate expansion state is admitted only when:

- its pi_k continuation label is positive;
- its frozen `C^k` phase score is strictly greater than the disjoint calibration
  threshold;
- it is not already present in Tube_k or an earlier accepted expansion row;
- its snapshot/state hash and active phase reproduce exactly.

CALIBRATION and ACCEPTANCE rows are never embedded.

For Tube_2 specifically, **all 3,119 Tube_1 states become the retained source
core**. Retention does not mean only the original 222 Tube_0 entries.

## 8. Retained-core replay contract

Structural retention alone is not sufficient because expansion cardinality can
starve earlier support during PPO reset sampling.

Iteration-1 experiments established the current mainline replay choice:

```text
outer reset:
  90% Tube RSI
  10% natural

inside Tube RSI:
  75% retained source Tube_k
  25% newest expansion
```

For repair02 on Tube_1 this yielded approximately:

```text
67.5% retained Tube_0 core
22.5% Tube_1 expansion
10.0% natural
```

and achieved Tube_0 `222/222` with positive boundary gain.

For pi_2, the same generic contract refers to the **entire Tube_1 support** as the
retained source core. Do not hard-code the old 222-state Tube_0 subset.

This 75/25 choice is the selected mainline method, not an invitation to sweep
replay ratios against every new gate outcome.

## 9. Iteration-1 initialization study — closed

The completed comparison is:

| policy/checkpoint | Tube_0 | regressions | upstream | downstream | boundary | groups |
|---|---:|---:|---:|---:|---:|---:|
| repair02 | **222/222** | **0** | **117/117** | **105/105** | 26/260 | 4 |
| B 1.024M | 217/222 | 5 | 112/117 | **105/105** | 33/260 | 3 |
| B 2.5088M | 206/222 | 16 | 101/117 | **105/105** | 28/260 | 4 |
| B 5.0176M | 214/222 | 8 | 109/117 | **105/105** | 25/260 | 4 |
| B 7.5008M | 217/222 | 5 | 112/117 | **105/105** | 42/260 | 4 |
| B 10.0096M | 212/222 | 10 | 107/117 | **105/105** | 46/260 | 4 |

No B checkpoint achieved both `222/222` retention and boundary success greater
than repair02's 26/260. Warm-start A was also inferior and is discarded.

All B core regressions were upstream while downstream remained 105/105. The
supported interpretation is **upstream expansion/retention policy interference**.
The nonmonotonic regression sequence rules out a simple monotonic "trained too
long" explanation.

Warm-start + explicit retention constraints may become a future ablation/method
extension. Do not reopen it during the current mainline pi_1 -> pi_2 round.

## 10. Historical Iteration-1 selection and claim boundary

Selected engineering pi_1:

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

Known identity:

- actor SHA-256: `85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a`
- payload SHA-256: `3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc`

Historical quickcheck:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_repair02_quickcheck_20260903/summary.json`

Observed:

- core: 222/222, zero regression;
- boundary candidate: 26/260 across 4 parent groups;
- baseline reproduction failures: 3.

The 3 reproduction failures arise because historical continuation labeling and
paired-gate rerolling did not use the same PRNG hierarchy. Therefore:

- repair02 is selected for **engineering continuation**;
- the historical Iteration-1 gate is **not** claimed strict formal PASS;
- `select_iteration_policy.py --allow-baseline-reproduction-mismatch` must encode
  the quarantine explicitly;
- no historical artifact may be edited to erase the mismatch.

## 11. Future locked-baseline acceptance gate

For k >= 1, acceptance is precommitted before pi_(k+1) training.

### 11.1 Core baseline lock

Before candidate training:

- evaluate every Tube_k retained-core state under selected pi_k;
- bind exact state identity, phase/local index and deterministic core seed;
- record pi_k success/outcome and interactions;
- require non-vacuous successful baseline support in the paired summarizer.

### 11.2 Boundary baseline lock

The ACCEPTANCE role has already been labeled under pi_k. Before candidate
training:

- select only pi_k-negative ACCEPTANCE rows;
- require boundary-negative support in both phases;
- require the predeclared minimum number of independent parent groups;
- bind exact snapshot/state, pi_k labeling seed and candidate index;
- lock the negative outcome itself.

The locked boundary baseline is **not re-run after pi_(k+1) is trained**.

### 11.3 Candidate evaluation

After pi_(k+1) freeze:

- candidate core uses the exact locked core seeds;
- candidate boundary starts from the exact locked snapshots;
- candidate boundary reproduces the original pi_k labeling key hierarchy:
  `labeling_seed -> candidate_index -> tick`;
- core PASS requires zero baseline-success -> candidate-failure regressions;
- boundary gain requires successes in at least the predeclared number of parent
  groups;
- no training, calibration, validation, TEST, final evidence or expert switching
  is used.

Because the baseline boundary is locked rather than re-rolled, the historical
"baseline-negative but baseline rerun positive" failure mode is removed from
future rounds by protocol design.

## 12. Workflow automation contract

The production operator entry point is:

```bash
python JIT/cli/run_iteration_workflow.py --config <workflow.json> --execute
```

A workflow is generated by:

`JIT/cli/prepare_iterative_envelope_workflow.py`.

The workflow is resumable but scientifically non-adaptive:

- each stage declares immutable prerequisites and a completion artifact;
- completion JSON assertions are checked before advancement;
- already-completed stages are revalidated before reuse;
- workflow config SHA drift after state creation is rejected;
- any engineering or scientific stage failure stops execution;
- the runner never changes reward, replay ratio, PPO settings, model architecture,
  threshold rules, physics, acquisition panel or acceptance criteria in response
  to a failure;
- the workflow contains no final TEST/JCE/JEL stage.

The current generic round stages are:

```text
prepare frontier plan
  -> run TRAIN role
  -> run CALIBRATION role
  -> run ACCEPTANCE role
  -> fit/calibrate C^k
  -> build Tube_(k+1)
  -> Tube-RSI smoke
  -> role-isolation audit
  -> lock pi_k acceptance baseline
  -> prepare pi_(k+1) training
  -> train pi_(k+1)
  -> freeze pi_(k+1)
  -> strict candidate gate
  -> select pi_(k+1) only on PASS
```

## 13. Current completed chain

```text
frozen experts
  -> Tube_0 (222)
  -> frozen pi_0
  -> pi_0-conditioned C^0
  -> fresh validation PASS
  -> Tube_1 (3,119 = 222 core + 2,897 expansion)
  -> first pi_1 / repair01 / repair02
  -> warm-start A/B ablations
  -> B intermediate checkpoint sweep complete
  -> no B point beats repair02 under retention-first rule
  -> repair02 selected as engineering pi_1
  -> historical formal gate mismatch quarantined
  -> CURRENT: automatic pi_1 -> C^1 -> Tube_2 -> pi_2
```

## 14. Immediate order from here

1. Register repair02 as `jit_selected_iteration_policy_v1` with the explicit
   baseline-reproduction quarantine flag.
2. Generate one `pi_1 -> pi_2` workflow from selected repair02 + exact Tube_1.
3. Dry-run the workflow plan before execution.
4. Execute the same immutable workflow config.
5. Let the runner stop if frontier support, C^1 calibration, Tube_2 isolation,
   training/freeze, core retention or boundary gain fails.
6. If the new strict pi_1 -> pi_2 gate PASSes, select pi_2 and continue with the
   same generic k -> k+1 machinery.
7. If it FAILs, preserve the artifact and diagnose before declaring any new
   repair or experiment.
8. Keep final TEST/JCE/JEL untouched until iteration stopping and final policy
   selection are frozen.

Exact launch commands and current artifact paths are maintained in
`JIT/docs/CURRENT_STATUS.md`.

## 15. Convergence and final empirical envelope

Iteration stopping criteria must be declared before the final stopping decision.
Candidate non-final criteria include:

- boundary gain saturation;
- negligible new Tube support;
- repeated inability to expand without retained-core regression;
- reaching a predeclared physical envelope target;
- resource budget exhaustion.

Do not continue merely because another round is mechanically possible.

Only the final selected frozen unified policy may be evaluated on the untouched
final envelope bank. The resulting claim is an **empirical policy-conditioned
jumping capability envelope**, not a formal safety or viability proof.
