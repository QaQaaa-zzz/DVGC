# Two-Phase Learned Soft-Feasibility-Tube Method

## Status

This document defines the approved research method and its current iterative
extension. It is a design contract, not permission to claim unvalidated
capabilities. The bootstrap chain through frozen phase experts, first-pass
`V_up`/`V_down`, a TRAIN-only learned Soft Tube, and one completed unified PPO
run now exists. Policy-conditioned Tube expansion and final empirical JCE/JEL
remain to be implemented and experimentally closed.

## 1. Guideline

`data/reference_jump.csv` is a kinematic guideline and weak prior. It may
provide broad jump-space intervals, Apex/descent/recovery kinematic envelopes,
hip/knee motion trends, initial threshold suggestions, physical seed proposals,
and weak reward/evaluation priors. It is not a pointwise trajectory-tracking
target, expert policy, trained policy, or authoritative dynamic controller for
the current 2 kg, +/-50 N m model. Complete open-loop replay is not a Gate B
requirement, and a reference state is not presumed reachable or safe.

The retained prelaunch and roll-limit replay evidence records that full dynamic
compatibility was not demonstrated. It is provenance for this method revision,
not a reason to tune actions, windows, thresholds, posture limits, or the model.

## 2. Local phase experts

The experts are not the final method output. Their primary roles are to explore
states that occur under the authoritative dynamics, provide checkpoint-bound
continuation controllers, and generate the distributions needed to learn
phase-specific continuation feasibility. The only deployment controller is the
later unified policy.

Expert training and feasibility-data acquisition may overlap in time. Formal
feasibility labels are normalized by re-labeling accumulated candidates under
the selected frozen phase expert before training the bootstrap feasibility
fields.

Phase-expert checkpoints contain the observation normalizer and policy/value
parameters used for inference. They do not contain optimizer or environment
rollout state and therefore support an auditable warm start, not bitwise
full-state continuation.

### Propulsion-Ascent

The upstream expert owns ground propulsion, takeoff, and rising flight. Its
local objective is to reach the Apex transition band with physically valid
state, sufficient forward progress, and continuation-compatible motion.
Reference states and actions are never replayed to initialize or control Phase
U, and the guideline is never used for behavior cloning, imitation learning,
or pointwise time-indexed trajectory tracking.

### Descent-Recovery

The downstream expert starts in the Apex transition band and owns descending
flight, landing, and stable recovery. Its objective uses the actual terminal
task reward and distinguishes physical failure from finite-horizon timeout.

After Phase U is frozen, real online `pi_up` rollouts at Apex pre/nearest/post
and early descent are the primary formal Phase-D source. Preliminary
reference-derived states, if any, are physical proposals only and never
reachability or safety evidence.

An expert never trains from a learned Tube or a later unified-policy
continuation field.

## 3. Apex transition band

The Apex transition band is the overlap/interface between experts, not an
independently trained phase. Its contract specifies observable physical
features, direction of travel, admissible rates, event timing, and provenance.
Membership means a downstream continuation may be attempted; it does not mean
the state is certified safe or guaranteed recoverable.

## 4. Candidate snapshots, perturbations, and labels

Checkpoint policies generate real online snapshots at stratified locations from
pre-window approach through Apex post. A snapshot preserves the physical state
and online control context needed for consistent replay, including observation
history, last action/control state, estimator/event state, timing fields,
parent trajectory identity, and XML/config/policy provenance. History must be a
real consecutive FIFO; CSV reconstruction, copied frames, and offline kinematic
states are not online policy snapshots.

Successful, near-success, failure, high-attitude, low-clearance,
low-forward-speed, mistimed-ascent, and Apex-boundary states are useful
candidate classes. Boundary acquisition should prefer real-dynamics generation:
restore an audited real snapshot and step the authoritative environment under a
bounded, predeclared action perturbation or policy rollout. Direct qpos/qvel
state dilation is not evidence that a capability state is reachable.

Labels answer phase-specific continuation questions under a frozen policy and a
frozen protocol. Failures are evidence under the evaluated controller/data
protocol, not claims of physical impossibility. Physical failures, task
failures, timeouts, and invalid snapshots remain separate outcomes.

Continuation accounting records rollout count, success count, empirical rate,
physical-failure rate, timeout rate, closed outcome counts, source-policy hash,
and protocol hash. "Alive for N ticks" is never a positive label: the declared
Apex or stable-recovery completion must occur.

## 5. Bootstrap feasibility fields

Two phase-conditioned models are learned from frozen phase-expert continuation
data:

- `V_up(s)`: estimated expert-conditioned upstream continuation score;
- `V_down(s)`: estimated expert-conditioned downstream recovery score.

Labels are policy-dependent: fields learned under different checkpoints or
controllers are not interchangeable. The formal bootstrap datasets use
parent-disjoint train/validation/test splits, leakage tests, class-imbalance
handling, ranking evaluation, and calibration diagnostics.

After selecting `pi_up_star` or `pi_down_star`, accumulated candidates are
re-labeled under that one frozen expert before training the corresponding
formal bootstrap field. Scores are continuation-feasibility scores, not true
safety probabilities unless a narrower calibrated claim is independently
supported.

`V_up` and `V_down` are the authority used to construct `Tube_0`; they are not
silently reused to represent feasibility under a later unified policy.

## 6. Learned soft feasibility tubes

A soft Tube is a thick weighted TRAIN distribution induced by a learned
continuation score, support confidence, and explicit physical-validity filters.
A policy trajectory is a source of candidate states and is never by itself a
Tube.

Soft-Tube records identify the model, dataset, split, threshold/weighting rule,
XML/config/action mapping, and source policy. They must explicitly record:

```text
training_guidance_only = true
certified_safe = false
```

No fixed top-percent threshold, fixed coordinate dilation, or universal branch
funnel is assumed to define a capability boundary. Final evaluation budgets are
separate from training admission rules.

## 7. Unified Tube-RSI PPO

One unified policy is trained from phase-balanced or otherwise predeclared
soft-Tube resets. Training combines:

- Tube-RSI reset sampling across both phases and the transition band;
- observable jump/phase signals;
- the final task reward and failure/timeout semantics;
- soft feasibility guidance with bounded, documented influence.

The unified policy preserves one action mapping and one observation/runtime
contract. No expert switching is allowed in the final controller. Training
outputs are policy candidates until frozen and evaluated under the next
protocol stage.

The scientific JIT domain is Tube-conditioned jump execution and recovery. A
cold-start locomotion prefix before Tube entry is not required for the final
JCE/JEL unless a separately declared research question explicitly includes it.

## 8. Policy–Tube capability-envelope iteration

The first learned Tube is `Tube_0`, bootstrapped from expert-conditioned
`V_up`/`V_down`. Once a unified policy `pi_k` is frozen, that unified policy
becomes the continuation authority for the next expansion iteration.

Iteration `k` uses the following loop:

```text
Tube_k
  -> train/freeze unified pi_k
  -> real-dynamics boundary acquisition
  -> frozen pi_k continuation labeling
  -> policy-conditioned C_up^k / C_down^k
  -> evidence-backed Tube_{k+1}
  -> train/freeze unified pi_{k+1}
```

`C_up^k` and `C_down^k` are policy-conditioned continuation fields learned from
labels generated under the exact frozen `pi_k` payload and protocol. They must
not be mixed with expert-conditioned labels or another policy iteration.

Boundary expansion is evidence-driven. Candidate states are produced from
provenance-complete real snapshots and authoritative dynamics, including
successful trajectories and bounded action-basis perturbation rollouts around
audited boundary anchors. Simply increasing coordinate ranges is not a
capability-envelope update.

`Tube_{k+1}` retains sufficient `Tube_k` core support to control catastrophic
forgetting and adds newly supported boundary states. Each new state binds its
source trajectory/snapshot, physical-state hash, acquisition protocol, frozen
policy hash, continuation outcomes, split, continuation-field model, and Tube
admission/weight rule.

Expansion TRAIN, expansion validation, and final envelope-evaluation data are
separate. Parent trajectories and near-duplicate physical states must be
split-disjoint; seed disjointness alone is not enough.

Policy improvement may use a previous-Actor warm start or fresh Actor only if
that initialization rule is predeclared as part of the method experiment. It
cannot be changed after inspecting held-out final results. Critic and optimizer
provenance remain explicit.

Each accepted iteration must satisfy both a core-preservation gate and a
boundary-gain gate on non-final audit data. A larger Tube alone does not imply a
stronger policy.

Detailed iteration, leakage, and convergence rules are defined in
`JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`.

## 9. Final empirical Jump Capability Envelope

The iteration loop stops only under a predeclared convergence criterion, such
as small new-support gain, small boundary movement, small continuation-success
gain over consecutive iterations, and preserved core performance. The exact
coverage representation must be chosen before it is used to claim convergence.

Only after stopping is frozen may one final unified policy be evaluated on a
separately predeclared, disjoint envelope-evaluation bank with immutable
XML/config/action mapping and zero training transitions.

The final JCE/JEL reports empirical success/coverage over a declared
initial-state distribution and explicitly separates success, physical failure,
task failure, timeout, and invalid states. Expert-conditioned rollouts, learned
scores, TRAIN Tube membership, TRAIN panels, expansion validation, and
iteration-selection audits cannot themselves support the final JCE/JEL claim.

The final object is an **empirical policy-conditioned Jump Capability Envelope**.
It is not a formal invariant set, proof of reachability, guaranteed viability
kernel, or certified safe region.

## 10. Current implementation order

1. Preserve the frozen experts, expert-conditioned `V_up`/`V_down`, and the
   existing 222-entry TRAIN-only `Tube_0` as bootstrap provenance.
2. Preserve the completed Round-1 unified checkpoint and freeze its exact
   identity as candidate `pi_0` for Tube-conditioned expansion work.
3. Treat the Round-1 natural-start `yaw_limit` result as a retained cold-start
   diagnostic, not the final JIT gate under the Tube-conditioned task scope.
4. Preserve the preflighted but unlaunched `natural50` Round-2 as superseded
   provenance; do not run it merely to solve the out-of-domain prefix.
5. Reuse/generalize the existing real-dynamics boundary machinery for unified
   policy expansion rather than creating a version-suffixed parallel stack.
6. Implement frozen unified-policy continuation labels for upstream and
   downstream boundary candidates with no expert switching.
7. Train/calibrate policy-conditioned `C_up^k` / `C_down^k` on expansion TRAIN
   data only and build `Tube_{k+1}` with core retention.
8. Predeclare and train the next unified policy under a fixed initialization
   rule; apply core-preservation and boundary-gain gates.
9. Iterate until the predeclared convergence rule triggers.
10. Freeze the final unified policy and run the independent empirical JCE/JEL
    bank.

No later step may be advertised as complete before its code, tests, inputs,
hashes, and outputs have been validated.

## 11. Phase U reward and run discipline

Phase U reward is bounded observable task progress: forward propulsion;
one-time legal-window entry; post-entry ascent, clearance, and Apex-band
approach; online Apex success; attitude/rate, illegal-contact, action magnitude
and smoothness penalties; and physical/task failure penalties. Jump/ascent
progress is zero before the legal-window-entry event. Early airborne is
diagnostic only: it neither fails nor succeeds and cannot unlock post-window
reward.

The guideline supplies broad engineering scales only. No reward term uses a
reference time, index, action, future state, success dataset label, or pointwise
reference distance. Every component is bounded and recorded independently.

Training, fixed TRAIN diagnostics, snapshot acquisition, continuation labeling,
expansion acquisition, runtime smoke, iteration audit, and final JCE/JEL
interactions are accounted separately. Reward or physical contracts are not
rewritten merely because an early policy checkpoint has low success on an
out-of-domain diagnostic.
