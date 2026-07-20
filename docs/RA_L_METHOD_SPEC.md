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
`pi_A -> pi_T -> pi_U -> pi_X -> pi_D -> pi_L`, where `pi_U`, `pi_X`, and
`pi_D` own Ascent, Apex, and Descent respectively.  An upstream expert is trained only to enter
the frozen, protocol-defined next-stage entry region of its successor.  Only
Descent->Landing uses the independently certified canonical `C_L`.  A handoff is
irreversible and continues the same physical state, observation/action
history, PolicyState, terrain, command, disturbance, and episode seed.  Chain
denotes the canonical-entry event; Final denotes end-to-end Recovery under the
complete downstream controller stack.  Final trajectories that never match
the canonical entry are reported separately as Chain-missed Final.

Any Tube certified under a composite expert stack is provisional recoverable
support (an expert-conditioned or discovery Tube).  Its manifest binds every
expert and entry-set hash, the controller-stack hash, candidate bank, XML and
runtime hashes, branch seed/dynamics variant, oracle phase, and PolicyState
provenance.  Local entry events are `next_stage_reach`; `composite_chain` is
reserved for recursive expert connection to `C_L`.  Neither is a formal Jump
Capability Envelope.

After each stage has enough local next-stage evidence for labels,
reachability estimation, and provisional proposals, a new shared Actor is
initialized by phase-balanced, label-aware joint distillation and is then
trained jointly from that evidence.  A complete immutable expert-stack Final
evaluation, when available, forms a stronger provisional-recoverability
subset; it is not a mandatory pass gate for every local expert.  Oracle stage and teacher ID
remain excluded from Actor input.  Only phase-wise Tubes independently
recertified under the frozen final shared policy may be named the formal JEL.

The three evidence objects are deliberately non-interchangeable:

| Object | Controller semantics | Permitted claim |
| --- | --- | --- |
| Local proposal support | A local expert supplies `next_stage_reach` evidence for a frozen protocol-defined successor region | Labels, acquisition and controller proposals only |
| Expert bootstrap envelope | Final-Recovery under an immutable composite expert stack | Expert-conditioned provisional recoverability; RSI/distillation data only |
| Final shared-policy JEL | Final-Recovery branch recertification under one frozen shared Actor | Formal phase-wise Tube and empirical JEL |

Intermediate training gates are exclusively `next_stage_reach`:
`Takeoff -> Ascent`, `Ascent -> Apex`, `Apex -> Descent`,
`Descent -> Landing/C_L`, and `Landing -> Stable`.  Only `pi_D` uses `C_L`;
neither `pi_X`, `pi_U`, nor `pi_T` may use C_L, Full Chain, or Final Recovery
as a training/unlock gate.  The frozen Flight bootstrap stack
`pi_U -> pi_X -> pi_D` may be abbreviated `pi_F` only when the component
responsibilities remain explicit.  A local failure is a support gap under the
current controller bank and cannot block acquisition in independent stages.
Composite evaluation switches irreversibly at each valid entry without
resetting physics, observation/action history, event-filter state, or any
other PolicyState field.

Local rollout positives, boundaries, reference states, and reachability-model
proposals form `proposal_support_bank` artifacts only.  End-to-end recovery
under a fully immutable expert stack may be named
`expert_conditioned_provisional_envelope`; it remains distinct from both local
proposal support and the final shared-policy JEL.

The final shared Actor is a new policy, not an alias for any expert.  Its
initialization dataset is phase-balanced across expert trajectories and keeps
teacher actions, physical state, deployable observation/history and empirical
labels, while stage oracle and teacher identity are supervision metadata only.
Joint RSI PPO may consume Final-positive/boundary states independently
evaluated under an immutable expert stack.  Ordinary local proposals retain
only positive/boundary/unknown next-stage labels and are never called safe.
None of those labels survive the policy change.
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
