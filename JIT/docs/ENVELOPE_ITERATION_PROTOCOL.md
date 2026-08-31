# JIT Policy–Tube Envelope Iteration Protocol

## Status

This document defines the next research stage after the first completed unified
Tube-RSI policy. It is a design and experiment contract, not an implementation
claim. No envelope-iteration run may be advertised as complete until its code,
tests, declarations, frozen-policy evidence, and independent evaluation are
closed.

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
  -> frozen pi_k boundary acquisition
  -> pi_k-conditioned continuation labels
  -> policy-conditioned continuation fields C_up^k / C_down^k
  -> expanded Soft Tube_{k+1}
  -> unified policy pi_{k+1}
  -> repeat until predeclared expansion/evaluation convergence
  -> independent frozen-policy empirical JCE/JEL
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
perturbations or policy rollouts. The current `upstream_boundary.py` contract
already follows this rule for Phase-U boundary acquisition.

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
  training Tube as supervision unless a later protocol explicitly re-splits
  and re-declares the data;
- **final envelope evaluation**: must remain untouched by Tube construction,
  policy training, checkpoint selection, threshold selection, and iteration
  stopping decisions.

Parent trajectories and near-duplicate physical states must be group-disjoint
across these roles. Seed disjointness alone is insufficient when deterministic
or nearly deterministic resets create the same physical state.

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

## 6. Constructing Tube_{k+1}

`Tube_{k+1}` is a TRAIN-only soft guidance distribution constructed from:

- retained support from `Tube_k` needed to prevent catastrophic forgetting;
- newly accepted real-dynamics expansion states;
- policy-conditioned continuation scores;
- support/confidence information;
- explicit physical-validity filters;
- phase-balanced or otherwise predeclared sampling weights.

Expansion must be evidence-driven. "Increase every state bound by 10%" is not
an admissible capability-envelope update.

A newly accepted state must carry provenance linking it to:

- source snapshot/trajectory;
- physical-state hash;
- acquisition protocol;
- frozen `pi_k` payload hash;
- continuation-label protocol and outcomes;
- split assignment;
- continuation-field model and calibration rule;
- Tube admission/weighting rule.

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

Actor initialization is itself a method variable. A later implementation may
choose previous-actor warm start or fresh initialization, but the choice must be
fixed in the iteration protocol and cannot be changed after inspecting final
held-out results. Critic and optimizer provenance must be explicit.

The final deployment/evaluation controller remains one unified Actor. Local
experts are bootstrap/data-generation tools and are never switched at runtime.

## 8. Core-preservation gate

An expanded policy is not automatically better because its Tube is larger.
Before accepting `pi_{k+1}` as the next iteration authority, it must demonstrate
both:

1. **core preservation**: performance on a locked, non-final audit subset of the
   previous Tube does not degrade beyond a predeclared tolerance;
2. **boundary gain**: performance/coverage on newly acquired boundary states
   improves beyond a predeclared minimum.

These are iteration-selection diagnostics, not final JCE/JEL evidence.

## 9. Convergence

"Converged" must be defined before the final iteration. Candidate stopping
criteria include all of the following, evaluated on non-final audit data:

- newly admitted support/coverage gain below `epsilon_support`;
- boundary displacement or score-frontier movement below `epsilon_boundary`;
- continuation-success improvement below `epsilon_success`;
- no material gain for a predeclared number of consecutive iterations;
- core-preservation gate remains satisfied.

The exact metric representation (occupancy coverage, alpha-shape/hull volume,
stratified grid coverage, or another declared measure) must be chosen before it
is used for a scientific convergence claim. Raw high-dimensional convex-hull
volume is not assumed to be meaningful by default.

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

## 11. Iteration-0 interpretation

The existing 222-entry TRAIN-only learned Soft Tube is `Tube_0` bootstrap
support. The completed Round-1 unified checkpoint is a candidate `pi_0` for
Tube-conditioned expansion work because its final fixed TRAIN Tube panel reached
16/16 success. That TRAIN result is evidence of learned Tube competence, not an
independent JCE/JEL result.

The Round-1 canonical natural-start rollout ended before the jump zone at
`yaw_limit`. Under the revised task scope this is retained as an out-of-domain
cold-start diagnostic, not the next scientific gate.

The predeclared Round-2 `natural50` experiment is therefore superseded before
launch. Its files remain immutable historical provenance; they are not deleted,
rewritten, or represented as a completed run.

## 12. Immediate implementation order

1. Preserve Round-1 artifacts and the unused Round-2 natural50 prelaunch
   declaration as provenance.
2. Freeze the exact `pi_0` checkpoint identity used for expansion acquisition.
3. Implement a generic unified-policy boundary-acquisition capability by
   extending/reusing the real-dynamics boundary machinery rather than creating
   a version-suffixed parallel stack.
4. Implement frozen unified-policy continuation labeling with phase-aware event
   accounting and no expert switching.
5. Implement policy-conditioned continuation model training and calibration.
6. Build `Tube_1` without touching final evaluation data.
7. Predeclare and train `pi_1` under the selected policy-improvement
   initialization rule.
8. Apply core-preservation and boundary-gain gates.
9. Repeat until the predeclared convergence rule triggers.
10. Freeze the final policy and run the independent JCE/JEL bank exactly once
    per declared protocol.
