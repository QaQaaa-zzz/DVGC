# Two-Phase Learned Soft-Feasibility-Tube Method

## Status

This document defines the approved research method. It is a design contract,
not an implementation claim. The repository does not yet contain the final
two-phase experts, `V_up`/`V_down`, soft-Tube builder, unified two-phase PPO, or
formal two-phase pipeline CLI.

## 1. Guideline

`data/reference_jump.csv` is a kinematic guideline and weak prior. It may
provide broad jump-space intervals, Apex/descent/recovery kinematic envelopes,
hip/knee motion trends, initial threshold suggestions, physical seed proposals,
and weak reward/evaluation priors. It is not a pointwise trajectory-tracking
target, expert policy, trained policy, or authoritative dynamic controller for
the current 4 kg, +/-50 N m model. Complete open-loop replay is not a Gate B
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

Expert training and feasibility-data acquisition overlap in time. Candidate
states may be collected as soon as a checkpoint has repeatable physical
success, while the expert continues training. Formal feasibility labels are
later normalized by re-labeling accumulated candidates under the selected
frozen phase expert.

### Propulsion-Ascent

The upstream expert owns ground propulsion, takeoff, and rising flight. Its
local objective is to reach the Apex transition band with physically valid
state, sufficient forward progress, and continuation-compatible motion.
It trains directly from audited natural stable resets. Reference states and
actions are never replayed to initialize or control Phase U, and the guideline
is never used for behavior cloning, imitation learning, or pointwise
time-indexed trajectory tracking.

### Descent-Recovery

The downstream expert starts in the Apex transition band and owns descending
flight, landing, and stable recovery. Its objective uses the actual terminal
task reward and distinguishes physical failure from finite-horizon timeout.

Its preliminary smoke/early-pilot seeds may begin as reference-derived
kinematic proposals only after MuJoCo forward, finite-state, penetration,
geometry, short-horizon dynamics, real three-frame FIFO, and timing-explicit
snapshot validation. Such records are labeled only
`physically_validated_descent_seed`, never reachable, expert, Tube, safe, or
certified. After Phase U is frozen, real online `pi_up` rollouts at Apex
pre/nearest/post and early descent become the primary formal Phase D source.

An expert never trains from a learned Tube or provisional feasibility score.
A provisional feasibility model may guide where the data-acquisition process
adds continuation probes, but it must not shape expert reward, expert reset
sampling, or expert training distribution.

## 3. Apex transition band

The Apex transition band is the overlap/interface between experts, not an
independently trained phase. Its contract must specify observable physical
features, direction of travel, admissible rates, event timing, and provenance.
Membership means the downstream expert may attempt continuation; it does not
mean the state is certified safe or guaranteed recoverable.

## 4. Candidate snapshots, perturbations, and labels

Checkpoint experts generate real online snapshots at stratified locations from
pre-window approach through Apex post. A snapshot preserves
the physical state and online control context needed for consistent replay,
including observation history, last action/control state, estimator/event
state, delay/timing fields when applicable, parent trajectory identity, and
XML/config/policy provenance. History must be a real consecutive three-frame
FIFO; CSV reconstruction, copied frames, and offline kinematic states are not
expert snapshots.

Successful, near-success, failure, high-attitude, low-clearance,
low-forward-speed, mistimed-ascent, and Apex-boundary states are all required.
Small approved physical perturbations provide thickness around real online
states. Every perturbation must pass finite, joint-limit, geometry,
non-penetration, timing, and snapshot validation. A short settle check is a
physical-validity diagnostic, not proof that the perturbed state is reachable.

Labels answer phase-specific downstream-completion questions under a frozen
data-generation protocol:

- upstream label: valid continuation from a Propulsion-Ascent snapshot into the
  Apex transition band;
- downstream label: valid continuation from an Apex/Descent snapshot through
  landing to stable recovery.

Failures are evidence under the evaluated controller/data protocol, not claims
of physical unreachability. Physical failures, timeouts, and invalid snapshots
remain separate outcomes. Independent audit labels are never fed back into the
training split that produced a model.

Continuation accounting records rollout count, success count, empirical rate,
physical-failure rate, timeout rate, closed outcome counts, source-policy hash,
and protocol hash. A one-branch screen may cover many nominal states; 4-branch
probes cover representative states; 4–8 branches cover calibration strata;
8–32 branches are reserved for key boundaries. "Alive for N ticks" is never a
positive label: the corresponding Apex or stable-recovery completion must
occur.

## 5. Provisional and formal feasibility fields

Two phase-conditioned models are learned from continuation data:

- `V_up(s)`: estimated probability/score of valid upstream continuation;
- `V_down(s)`: estimated probability/score of downstream recovery.

Labels are policy-dependent: `V_up(s | pi_up@100k)` and
`V_up(s | pi_up@1M)` are not interchangeable. Early checkpoint data may train a
provisional model for acquisition guidance only. It requires a deployable
feature allowlist, parent-disjoint train/validation/test splits, leakage tests,
class-imbalance handling, ranking evaluation, and calibration diagnostics.

After selecting `pi_up_star` or `pi_down_star`, all accumulated candidates are
re-labeled under that one frozen policy. Only this normalized dataset may train
the corresponding formal feasibility field. Scores are continuation
feasibility scores, not true or safety probabilities, unless calibration
evidence supports a narrower claim.

## 6. Learned soft feasibility tubes

A soft Tube is a thick weighted region induced by a learned feasibility score,
support confidence, and explicit physical-validity filters. An expert rollout
is only a source of candidate centers and is never itself a Tube. The Tube may
contain validation-selected core, boundary, and exploration strata; no fixed
top-percent threshold is assumed before calibration.

Soft-Tube records must identify the model, dataset, split, threshold/weighting
rule, XML/config/action mapping, and source policy. They must not use
`certified_safe`, `certified_tube`, or equivalent formal-safety language.
They must explicitly record `certified_safe=false` and
`training_guidance_only=true`.

The previous 4 -> 8 -> 16/32 branch funnel is not a universal admission rule
for a learned training Tube. Any final evaluation branch budget is defined
separately and cannot convert training data into safety evidence.

## 7. Unified Tube-RSI PPO

One unified policy is initialized and trained from phase-balanced soft-Tube
resets. Training combines:

- Tube-RSI reset sampling across both phases and the transition band;
- observable jump/phase signals;
- the final task reward and failure/timeout semantics;
- soft feasibility guidance with bounded, documented influence.

The unified policy must preserve one action mapping and one observation/runtime
contract. Training outputs are proposals until the policy is frozen and
evaluated independently.

## 8. Final empirical Jump Capability Envelope

The final JCE/JEL is measured only with one frozen unified policy, disjoint
evaluation seeds, immutable XML/config/action mapping, and explicit physical
failure versus timeout reporting. It reports empirical success/coverage over a
declared initial-state distribution; it is not a formal invariant set or a
proof of safety.

Expert-conditioned rollouts, learned feasibility scores, and soft-Tube training
membership cannot themselves support the final JCE/JEL claim.

## 9. Interleaved implementation order

1. Gate A/B: specify and validate static semantics, pure-JAX runtime, geometry,
   thresholds, natural reset, seed protocol, and timing-explicit snapshots.
2. Gate C1: implement the unified phase-expert entrypoint and Phase U smoke
   capability; run smoke only under its explicit authorization.
3. Train Phase U from natural resets with checkpoints at requested 100k, 250k,
   500k, 750k, and 1M total environment transitions. Aligned effective counts
   are recorded separately when PPO rollout blocks cannot end at the requested
   number exactly.
4. As soon as fixed evaluation is nonzero and at least eight independent
   successful online parents exist, collect provenance-complete Phase U
   candidates and begin bounded checkpoint-specific continuation diagnostics.
5. Continue Phase U while candidate coverage and boundary information improve;
   a low 100k success rate alone is not a pause condition.
6. Once real Apex coverage exists, collect real Apex/early-descent seeds and
   begin the separately gated Phase D smoke/pilot without waiting for Tube-up.
7. Use provisional feasibility models only to guide further data acquisition.
   Select each formal phase expert, re-label all candidates under it, then train
   the formal `V_up` or `V_down` and construct the learned soft Tube.
8. Combine Tube-up and Tube-down in a separately validated Tube-RSI curriculum,
   freeze the unified policy, and run independent empirical JCE/JEL evaluation.

No later step may be advertised as complete before its code, tests, inputs,
hashes, and outputs have been validated.

## 10. Phase U reward and run discipline

Phase U reward is bounded observable task progress: forward propulsion;
one-time legal-window entry; post-entry ascent, clearance, and Apex-band
approach; online Apex success; attitude/rate, illegal-contact, action magnitude
and smoothness penalties; and physical/task failure penalties. Jump/ascent
progress is exactly zero before the legal-window-entry event. Early airborne is
diagnostic only: it neither fails nor succeeds and cannot unlock post-window
reward.

The guideline supplies broad engineering scales only. No reward term uses a
reference time, index, action, future state, success dataset label, or pointwise
reference distance. Every component is bounded and recorded independently.

The current Phase U training authorization is capped at 1,000,000 training
transitions, not a mandatory stopping target. Training, Brax evaluation, fixed
evaluation, snapshot acquisition, continuation labeling, and runtime-smoke
interactions are counted separately. Pause immediately on numerical or
contract failure, state corruption, hash drift, clear reward hacking, severe
action saturation, or repeated held-out degradation; do not rewrite reward
merely because one early checkpoint has low success.
