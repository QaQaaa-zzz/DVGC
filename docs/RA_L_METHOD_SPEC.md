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
- stage-expert discovery with irreversible certified handoffs;
- joint consolidation into one deployable shared Actor;
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

## Stage-Expert Discovery and Formal JEL

Sequential shared-Actor backward training is not the active discovery route.
Discovery uses independently owned frozen controllers in the stack
`pi_A -> pi_T -> pi_F -> pi_L`.  An upstream expert is trained only to enter
the fixed certified canonical entry set of its successor.  A handoff is
irreversible and continues the same physical state, observation/action
history, PolicyState, terrain, command, disturbance, and episode seed.  Chain
denotes the canonical-entry event; Final denotes end-to-end Recovery under the
complete downstream controller stack.  Final trajectories that never match
the canonical entry are reported separately as Chain-missed Final.

Any Tube certified under a composite expert stack is provisional recoverable
support (an expert-conditioned or discovery Tube).  Its manifest binds every
expert and entry-set hash, the controller-stack hash, candidate bank, XML and
runtime hashes, branch seed/dynamics variant, oracle phase, and PolicyState
provenance.  It is not a formal Jump Capability Envelope.

After all four seed-0 expert stacks pass their composite audits, a new shared
Actor is initialized by phase-balanced, label-aware joint distillation and is
then trained jointly from all provisional Tubes.  Oracle stage and teacher ID
remain excluded from Actor input.  Only phase-wise Tubes independently
recertified under the frozen final shared policy may be named the formal JEL.

The three evidence objects are deliberately non-interchangeable:

| Object | Controller semantics | Permitted claim |
| --- | --- | --- |
| Chain entry | The active expert reaches the immutable canonical successor entry set | Recursive progress only |
| Expert bootstrap envelope | Final-Recovery under an immutable composite expert stack | Expert-conditioned provisional recoverability; RSI/distillation data only |
| Final shared-policy JEL | Final-Recovery branch recertification under one frozen shared Actor | Formal phase-wise Tube and empirical JEL |

The Flight expert `pi_F` is optimized only for `Flight -> C_L`.  It stops its
Chain episode at the first valid canonical match and has no Landing-retention
objective.  End-to-end evaluation switches irreversibly to frozen `pi_L`
without resetting physics, observation/action history, event-filter state, or
any other PolicyState field.  Flight reset support unlocks in the fixed order
late descent -> descent -> apex -> ascent.  The same ownership and handoff
rules apply to `pi_T` and `pi_A` once their successor entry sets exist.

The final shared Actor is a new policy, not an alias for any expert.  Its
initialization dataset is phase-balanced across expert trajectories and keeps
teacher actions, physical state, deployable observation/history and empirical
labels, while stage oracle and teacher identity are supervision metadata only.
Joint RSI PPO may consume safe/boundary states from expert-conditioned
provisional envelopes, but those labels do not survive the policy change.
Every candidate is relabeled by fresh independent branches after the shared
Actor is frozen; only that recertification defines the paper's final JEL.

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
