# Two-Phase Learned Soft-Feasibility-Tube Method

## Status

This document defines the approved research method. It is a design contract,
not an implementation claim. The repository does not yet contain the final
two-phase experts, `V_up`/`V_down`, soft-Tube builder, unified two-phase PPO, or
formal two-phase pipeline CLI.

## 1. Guideline

A physically meaningful jump guideline defines broad phase timing, admissible
state envelopes, and event anchors. It may initialize reset distributions and
help interpret behavior, but it is not a pointwise trajectory-tracking target
and cannot label a state feasible or safe by geometric proximity alone.

## 2. Two experts

### Propulsion-Ascent

The upstream expert owns ground propulsion, takeoff, and rising flight. Its
local objective is to reach the Apex transition band with physically valid
state, sufficient forward progress, and continuation-compatible motion.

### Descent-Recovery

The downstream expert starts in the Apex transition band and owns descending
flight, landing, and stable recovery. Its objective uses the actual terminal
task reward and distinguishes physical failure from finite-horizon timeout.

Experts are trained without feasibility networks or learned Tubes. This keeps
expert data generation causally prior to feasibility learning.

## 3. Apex transition band

The Apex transition band is the overlap/interface between experts, not an
independently trained phase. Its contract must specify observable physical
features, direction of travel, admissible rates, event timing, and provenance.
Membership means the downstream expert may attempt continuation; it does not
mean the state is certified safe or guaranteed recoverable.

## 4. Snapshots and labels

Frozen experts generate snapshots at event-aligned states. A snapshot preserves
the physical state and online control context needed for consistent replay,
including observation history, last action/control state, estimator/event
state, delay/timing fields when applicable, and XML/config/policy provenance.

Labels answer phase-specific continuation questions under a frozen data
generation protocol:

- upstream label: valid continuation from a Propulsion-Ascent snapshot into the
  Apex transition band;
- downstream label: valid continuation from an Apex/Descent snapshot through
  landing to stable recovery.

Failures are evidence under the evaluated controller/data protocol, not claims
of physical unreachability. Physical failures, timeouts, and invalid snapshots
remain separate outcomes. Independent audit labels are never fed back into the
training split that produced a model.

## 5. Feasibility fields

Two phase-conditioned models are trained after expert data collection:

- `V_up(s)`: estimated probability/score of valid upstream continuation;
- `V_down(s)`: estimated probability/score of downstream recovery.

Training must use provenance-safe splits such as parent/trajectory-disjoint
validation. Calibration, coverage, uncertainty, and class balance must be
reported. The model predicts empirical continuation feasibility under its data
protocol; it does not certify safety.

## 6. Learned soft feasibility tubes

A soft Tube is a weighted region induced by calibrated `V_up` or `V_down`
scores and explicit physical validity filters. It supplies reset weights,
curriculum support, and optional shaping for unified training.

Soft-Tube records must identify the model, dataset, split, threshold/weighting
rule, XML/config/action mapping, and source policy. They must not use
`certified_safe`, `certified_tube`, or equivalent formal-safety language.

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

## 9. Required implementation order

1. Specify and test the two expert and Apex-band contracts.
2. Train and freeze the two experts.
3. Collect provenance-complete snapshots and continuation labels.
4. Train and validate `V_up` and `V_down`.
5. Construct learned soft-Tube banks.
6. Train one unified Tube-RSI PPO.
7. Freeze it and run an independent empirical JCE/JEL evaluation.

No later step may be advertised as complete before its code, tests, inputs,
hashes, and outputs have been validated.
