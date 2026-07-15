# Jump Envelope Learning RA-L Core Method Contract

## Scope

The paper-facing method is Jump Envelope Learning (JEL).  The repository keeps
the historical `dvgc` package name, but the empirical objects are defined as:

- a phase-wise Final-Recovery Tube is a policy-conditioned recoverable set;
- the union of phase-wise Tubes over task conditions is the empirical Jump
  Capability Envelope;
- Chain is only recursive progress into the certified successor entry set.

The concise implementation contains:

- event-anchored Landing-first backward bootstrap;
- separate recursive Chain and end-to-end Final-Recovery labels;
- Beta-posterior empirical Tube classification;
- Final-safe/boundary Tube-guided RSI;
- downstream rehearsal for a shared Actor;
- independent candidate and branch seeds for construction and audit;
- final natural-start evaluation.

Learned GRU phase estimation, same-physical/different-belief variants,
Physical-Belief viability, and trigger-budgeted relabeling are deferred and
must not be claimed as completed contributions.

## Control Contract

The policy action is ordered as:

```text
[steer, rear-wheel drive, hip, knee]
```

This order is authoritative for code, figures, equations, manifests, and
tables. The archived v23 document's `[hip, knee, steer, drive]` equation is not
the active implementation contract.

The Actor receives deployable proprioception, IMU-derived event-filter output,
task geometry, action history, and observation history. Oracle phase and
collision semantics may be used for labels and the privileged Critic, but not
for Actor input.

## Empirical Tube

For each physical candidate state and frozen policy, branches independently
sample declared dynamics and future action noise. Every branch records:

- Chain success;
- Final Recovery success;
- physical termination;
- timeout truncation;
- branch seed and dynamics variant.

Final-Recovery outcomes define Safe, Dead, Boundary, and Unknown:

- Safe: posterior lower quantile is at least `safe_threshold`;
- Dead: posterior upper quantile is below `dead_threshold`;
- Boundary: posterior mean lies between the two thresholds and posterior width
  is at most `boundary_max_width`;
- Unknown: minimum evidence or one of the decision requirements is missing.

Only Final-safe states form the high-mass RSI core. Chain-safe sets are stage
connection targets, not substitutes for end-to-end recoverability.

## Flight-to-Landing Entry Contract

The Flight successor set `C_L` is a canonical Landing-entry bank, not the
entire Landing Final-safe Tube.  Proposals are captured from the frozen Landing
policy at the first confirmed valid landing contact.  A source snapshot that
already starts after contact is admissible only when its contact age is within
the declared three-control-step entry window.  Every proposal is deduplicated,
then independently Final-Recovery certified; only Final-safe entries belong to
`C_L`.

Matching uses the declared task-relative 20-dimensional entry feature and
robust physical-unit scale floors.  Its radius is calibrated exclusively from
Landing entry construction/certification data.  Flight Chain is latched when a
state matches `C_L` during the fixed three-step handoff window; Flight Final is
the later end-to-end Recovery event.  Flight outcomes and audit labels cannot
calibrate the matcher.

## Minimum Main Experiments

Use five independent training seeds for:

1. natural-start PPO;
2. CoM/reference-envelope RSI;
3. backward curriculum without Final-Recovery Tube selection;
4. DVGC-Physical.

Report natural-start Final-Recovery rate, first-success steps, total interaction
cost, phase visitation, Chain-to-Final false progress, timeout and physical
failure rates, and independent Tube precision/recall/coverage. Reward and PPO
budgets must be shared across methods except for the explicit reward-shaping
diagnostic.
