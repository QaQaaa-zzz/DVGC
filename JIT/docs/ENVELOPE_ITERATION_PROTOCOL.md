# JIT Policy–Tube Envelope Iteration Protocol

## Status

This document defines the active iterative research contract after the completed
bootstrap, `pi_0`, policy-conditioned `C^0`, `Tube_1`, formal `pi_1` training,
immutable `pi_1` freeze, the first completed paired `pi_0 -> pi_1` gate, and the
zero-interaction diagnosis of that rejected candidate.

The first candidate iteration was **not accepted** because core preservation
failed even though boundary gain passed. The diagnosis established material
retained-core replay dilution under the original Tube_1 sampler. A repaired
iteration-1 policy-improvement method is therefore predeclared before another
candidate is trained.

An envelope iteration is not accepted merely because training or freezing
finishes. The next policy authority is accepted only after its declared
core-preservation and boundary-gain gates both pass.

## 1. Research objective

The JIT objective is not universal cold-start locomotion. The scientific domain
begins from a declared jump-capability state distribution around the learned
Tube and asks whether one frozen unified policy can complete the remaining jump
and recovery maneuver.

The research target is an empirical jumping capability envelope under the fixed
robot model, payload, action limits, control rate, obstacle/task semantics, and
policy class. The envelope is policy- and protocol-dependent. It is not a
formal invariant set, a viability kernel proof, or a certified safe set.

The authoritative progression is:

```text
phase experts
  -> frozen expert continuation labels
  -> expert-conditioned V_up / V_down
  -> bootstrap Soft Tube_0
  -> unified policy pi_0
  -> freeze pi_k
  -> real-dynamics TRAIN boundary acquisition under pi_k
  -> pi_k-conditioned continuation labels
  -> policy-conditioned continuation fields C_up^k / C_down^k
  -> independently calibrated core-retaining Soft Tube_{k+1}
  -> unified policy pi_{k+1}
  -> freeze pi_{k+1}
  -> core-preservation + boundary-gain gates
  -> if FAIL: stop, diagnose, predeclare a repaired method
  -> if PASS: accept next authority and repeat
  -> independent frozen-policy empirical JCE/JEL after iteration stopping
```

Natural-start behavior before entry into the declared jump-capability domain is
not a final JCE/JEL gate. Natural-start diagnostics may remain useful for
engineering analysis, but failure outside the declared Tube/envelope domain
cannot by itself reject a policy whose intended deployment/evaluation domain is
the Tube-conditioned jump maneuver.

## 2. Bootstrap versus iterative authorities

`V_up` and `V_down` learned from frozen `pi_up_star` and `pi_down_star` are the
bootstrap feasibility fields for `Tube_0`. They remain immutable provenance.
They must not be silently reinterpreted as feasibility under a later unified
policy.

For iteration `k >= 0`, expansion evidence is generated under one frozen
unified policy `pi_k`. The corresponding policy-conditioned continuation fields
are working objects:

- `C_up^k(s)`: estimated continuation score from an upstream/jump-approach state
  under frozen `pi_k` to the Apex transition condition and onward through the
  unified maneuver as declared by the iteration protocol;
- `C_down^k(s)`: estimated continuation score from an Apex/descent state under
  frozen `pi_k` through valid landing and stable recovery.

A score learned under `pi_k` is not interchangeable with a score learned under
`pi_j`, an expert, or a different checkpoint. Every dataset, model, Tube entry,
and report must bind the exact policy payload hash and protocol hash.

## 3. Candidate acquisition: real dynamics only

Envelope expansion must not be implemented by manually widening coordinate
bounds or directly mutating `qpos`/`qvel` to create favorable states.

The preferred expansion mechanism reuses the repository's existing boundary
principle: restore provenance-complete real snapshots and obtain new candidate
states by stepping the authoritative dynamics under bounded, predeclared action
perturbations or policy rollouts.

Candidate sources may include:

1. successful `pi_k` trajectories launched from the current TRAIN Tube;
2. states encountered just outside the current Tube support during successful
   real-dynamics rollouts;
3. bounded action-basis perturbation trajectories from audited boundary
   anchors;
4. separately declared physical-validity probes used only for acquisition,
   never for independent final evaluation.

Direct state-space dilation, copied histories, synthetic FIFO reconstruction,
or offline kinematic states cannot establish reachable expansion evidence.

## 4. Expansion splits and leakage boundary

Each iteration has three logically separate data roles:

- **expansion TRAIN**: may train `C_up^k` / `C_down^k` and contribute to
  `Tube_{k+1}`;
- **expansion validation**: may select score calibration, admission thresholds,
  model hyperparameters, and convergence settings, but may not enter the
  training Tube;
- **final envelope evaluation**: must remain untouched by Tube construction,
  policy training, checkpoint selection, threshold selection, and iteration
  stopping decisions.

Parent trajectories and near-duplicate physical states must be group-disjoint
across these roles. Seed disjointness alone is insufficient when deterministic
or nearly deterministic resets create the same physical state. Consumed
validation is consumed and must not be reused to retune a later field or Tube.

## 5. Frozen-policy continuation labels

Every expansion candidate is labeled only after `pi_k` is frozen. Labeling uses
no expert switching and no PPO updates.

A positive label requires completion of the declared continuation event chain;
"alive for N ticks" is never positive evidence. Outcome accounting must keep
success, physical failure, task failure, timeout, and invalid snapshot states
separate.

Branch counts are declared before labeling. Boundary states may receive more
branches than high-confidence interior states. Branch budgets are an empirical
measurement choice and never convert a learned Tube into a certified set.

## 6. Constructing and sampling Tube_{k+1}

`Tube_{k+1}` is a TRAIN-only soft guidance distribution constructed from:

- retained support from `Tube_k`;
- newly accepted real-dynamics expansion states;
- policy-conditioned continuation scores;
- support/confidence information;
- explicit physical-validity filters;
- phase-balanced or otherwise predeclared sampling weights.

The retained-core rule is structural:

```text
Tube_(k+1) = retained Tube_k core ∪ accepted TRAIN expansion_k
```

Structural presence is necessary but not sufficient. The policy-improvement
runtime must also declare how much probability mass is assigned to the retained
source core versus the newly added expansion. Otherwise a rapidly growing Tube
can make old support effectively disappear from training even though every old
entry remains present in the artifact.

After the rejected first Tube_1 candidate, the active repaired sampling
principle is:

```text
phase selection
  -> retained source core vs current expansion
  -> entry within selected source
```

For the predeclared iteration-1 replacement candidate:

- upstream/downstream phase mixture remains 0.5 / 0.5;
- within each selected phase, retained source core receives probability 0.5 and
  current expansion receives probability 0.5;
- retained core is sampled uniformly so low historical V scores cannot starve a
  state that is nevertheless part of the core-preservation contract;
- expansion remains value/continuation-weighted using the existing Tube entry
  sampling weights.

The 0.5 / 0.5 source split is a single predeclared method repair, not a grid or
post-hoc tuned gate threshold. A later change to that split is another method
revision and cannot be silently tuned against a consumed acceptance bank.

A thresholded `C^k` level set alone is not an acceptable replacement for the
existing core. Expansion must be evidence-driven; coordinate dilation alone is
not admissible evidence.

Every newly accepted state must carry provenance linking it to its source
snapshot/trajectory, physical-state hash, acquisition protocol, frozen `pi_k`
payload hash, continuation outcomes, split, continuation-field identity, and
Tube admission/weighting rule.

Every Tube artifact continues to declare:

```text
training_guidance_only = true
certified_safe = false
```

## 7. Policy improvement

Training `pi_{k+1}` from `Tube_{k+1}` is a separate, predeclared policy-
improvement experiment. The environment, reward semantics, action mapping,
physical limits, control rate, and task definition remain fixed unless a new
research question explicitly changes them.

Initialization is a method variable. The rejected first Tube_1 candidate used
fresh actor/critic/optimizer initialization with the same seed and PPO settings
as pi_0. The repaired iteration-1 candidate keeps that fresh initialization,
seed, PPO budget, Tube_1 support, physics, reward, action semantics, natural
reset probability, and phase mixture fixed. Its only method change versus the
rejected candidate is the declared within-Tube retained-core replay contract.

Because the candidates are fresh-initialized, a loss of pi_0 core competence is
best described as **core-support under-replay / capability regression under the
expanded training distribution**, not classical parameter forgetting from a
pi_0 warm start.

The final deployment/evaluation controller remains one unified Actor. Local
experts are bootstrap/data-generation tools and are never switched at runtime.

## 8. Core-preservation and boundary-gain gate

An expanded policy is not automatically better because its Tube is larger.
Before accepting `pi_{k+1}` as the next iteration authority, it must demonstrate
both core preservation and boundary gain under one locked paired-policy audit.

The current generic paired-gate contract is:

1. **core bank**: every declared `Tube_k` source-core state is evaluated under
   frozen `pi_k` and frozen `pi_{k+1}` using the same deterministic continuation
   runtime. Core preservation passes only when there are zero states for which
   the baseline succeeds and the candidate fails. The baseline must have at
   least one successful core state in each phase so the gate cannot pass
   vacuously;
2. **boundary bank**: use frozen TRAIN frontier states labeled continuation-
   negative under `pi_k` and exclude any state already admitted to
   `Tube_{k+1}`. The exact same locked state is evaluated under both policies.
   The baseline-negative outcome must reproduce, and the candidate must convert
   failures to successes in at least two distinct parent groups;
3. the locked bank is written and self-hashed before either policy rollout;
4. validation, final TEST/JCE/JEL data, expert switching, and policy training
   are forbidden during this gate.

This audit is an iteration-selection diagnostic, not final JCE/JEL evidence.
A Tube being numerically larger, PPO reward being larger, or a continuation
model producing larger scores is not by itself a boundary-gain proof.

A completed FAIL is immutable scientific evidence. It must not be converted to
a PASS by changing the consumed audit bank, threshold, or acceptance rule.
After a method revision informed by a completed gate, the old boundary bank is
consumed as selection/diagnostic evidence. The repaired candidate therefore
requires a newly predeclared fresh non-final boundary challenge bank for its
formal acceptance decision. The structural all-core bank may be repeated because
it is the explicit support-preservation contract rather than held-out boundary
selection evidence.

## 9. Convergence

"Converged" must be defined before the final iteration. Candidate stopping
criteria include all of the following, evaluated on non-final audit data:

- newly admitted support/coverage gain below `epsilon_support`;
- boundary displacement or score-frontier movement below `epsilon_boundary`;
- continuation-success improvement below `epsilon_success`;
- no material gain for a predeclared number of consecutive iterations;
- core-preservation gate remains satisfied.

The exact metric representation must be chosen before it is used for a
scientific convergence claim. Raw high-dimensional convex-hull volume is not
assumed to be meaningful by default.

## 10. Final empirical Jump Capability Envelope

Only after iteration stopping is frozen may the final policy be evaluated on a
separately predeclared and disjoint envelope-evaluation bank.

The final report must bind:

- final policy checkpoint and payload SHA-256;
- immutable XML/config/action mapping;
- declared initial-state distribution and bank-generation protocol;
- independent physical-state identities and duplicate audit;
- success / physical failure / task failure / timeout counts;
- stratified success across the envelope dimensions used in the paper;
- JCE/JEL coverage summaries and confidence intervals where meaningful;
- zero training transitions and zero expert switching during evaluation.

The result is an **empirical, policy-conditioned jumping capability envelope**.
It must not be described as a formal safe set, guaranteed viability kernel, or
proof of reachability outside the measured distribution.

## 11. Completed baseline, rejected candidate, and diagnosis

Completed chain:

```text
frozen phase experts
  -> V_up / V_down
  -> Tube_0 (222 TRAIN entries)
  -> pi_0 (10,009,600 PPO transitions, frozen)
  -> pi_0 TRAIN boundary evidence
  -> C_up^0 / C_down^0 with independent validation
  -> core-retaining Tube_1 (3,119 TRAIN entries)
  -> Tube_1 engineering gate GO
  -> first pi_1 candidate (10,009,600 PPO transitions)
  -> frozen comparison authority
  -> paired pi_0 vs pi_1 gate
  -> boundary PASS / core FAIL
  -> zero-interaction replay diagnosis
```

Frozen rejected-candidate identities:

- checkpoint payload SHA-256: `fb5c364057933d62c4e1b6ed49f3181cd36584c5b270f305eef18dff150e68e5`
- actor SHA-256: `cc8f202075479aa90c0451732773016dfa25ef8a0c849278b5917319070284f0`
- critic SHA-256: `1519803788302a936d3129eb7010de5fe153ea65410a49c35c1328e22fdf8d8f`
- normalizer SHA-256: `94ed2587601dbe187774f664a741f427d943148a0c5391cdf57a45939c50a191`

Completed paired-gate protocol SHA-256:

`24a126ee94472eebbcb59fff66618ae00dae41074a1d1cfee8bb816afaff410a`

Gate result:

- pi_0 core success: 222 / 222;
- rejected pi_1 core success: 201 / 222;
- core regressions: 21 = 16 upstream + 5 downstream;
- **core preservation: FAIL**;
- boundary bank: 56 frozen pi_0-negative TRAIN states;
- rejected pi_1 new successes: 12 across 5 parent groups;
- **boundary gain: PASS**;
- **iteration accepted: false**.

Core-regression diagnosis SHA-256:

`61e12385ef0e77180b773a2e0de04b36e2a27f649c24d509b4f18345c00a7689`

Sampling diagnosis under the rejected candidate:

- upstream retained-core sampling-weight share: 24.87%;
- downstream retained-core sampling-weight share: 3.10%;
- Tube-conditional probability of any old-core reset: 13.98%;
- all-episode old-core reset probability: 12.59%;
- all-episode expansion reset probability: 77.41%;
- all 21 regressions terminated as physical failures.

This proves replay dilution was material, while not proving it was the only
mechanism. The repaired candidate therefore changes the sampling contract once
and must still pass a new scientific gate.

Predeclared replacement config:

`JIT/configs/pi_unified_iter1_tube1_core_replay50_natural10.json`

## 12. Immediate implementation order

1. Preserve the rejected candidate, completed FAIL gate, and diagnosis artifacts
   unchanged.
2. Compile/regression-test the generic retained-core replay sampler and canonical
   formal wrapper.
3. Run a zero-interaction real-Tube_1 preflight and verify exactly 0.5 retained
   core / 0.5 expansion probability inside both phases.
4. Before inspecting replacement-candidate outcomes, lock a fresh non-final
   boundary-audit acquisition protocol that excludes the consumed 56-state bank,
   its parent/near-state neighborhood, Tube_1 admitted states, validation, and
   TEST.
5. Train exactly one fresh iteration-1 replacement candidate from
   `pi_unified_iter1_tube1_core_replay50_natural10.json`.
6. Freeze its exact final checkpoint as a candidate comparison authority.
7. Re-run the structural 222-state core-preservation contract and evaluate
   boundary gain on the new fresh challenge bank. The old consumed boundary bank
   may be reported only as descriptive regression evidence.
8. If either gate fails, stop and preserve the result; do not sweep replay ratios
   against the same gate.
9. Only after both gates pass may iteration 1 be accepted and authorize
   policy-conditioned `C^1` evidence and core-retaining Tube_2.
10. Keep final TEST/JCE/JEL untouched until iteration stopping and final-policy
    selection are frozen.
