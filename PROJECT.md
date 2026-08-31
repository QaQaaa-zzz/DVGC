# DVGC Project

## Publication target

DVGC targets a concise IEEE RA-L method for identifying and expanding the
jumping capability of a single-track two-wheeled robot with learned
continuation feasibility and one unified deployment policy.

The current method is not "RL learns one jump from a natural reset." The
research contribution is a policy-conditioned Tube / policy co-evolution:

```text
kinematic guideline / real physical states
  -> local phase experts
  -> frozen-expert continuation labels
  -> expert-conditioned V_up and V_down
  -> bootstrap learned Soft Tube_0
  -> unified Tube-RSI policy pi_0
  -> freeze pi_k
  -> real-dynamics boundary acquisition
  -> pi_k-conditioned continuation labels
  -> policy-conditioned continuation fields C_up^k / C_down^k
  -> expanded Soft Tube_{k+1}
  -> unified policy pi_{k+1}
  -> repeat until predeclared expansion/evaluation convergence
  -> independent frozen-policy empirical Jump Capability Envelope
```

The Apex transition band connects the two physical phases. It is a band of
admissible handoff states, not a separately owned expert and not a certified
safe set. The input reference is a kinematic guideline and weak prior, not an
expert or authoritative dynamic controller.

The local phase experts are bootstrap/data-generation tools, not deployment
outputs. The only final controller is one unified Actor. `V_up` and `V_down`
are expert-conditioned bootstrap fields for `Tube_0`; later Tube expansion is
conditioned on the frozen unified policy that will actually execute the
maneuver.

The scientific deployment/evaluation domain begins in a declared
jump-capability state distribution around the learned Tube. Ordinary natural
locomotion before Tube entry is not part of the final JCE/JEL requirement unless
a separate experiment explicitly expands the research question to cold-start
locomotion.

## Method contract

- `Propulsion-Ascent` learns launch and rising-flight behavior needed to reach
  the Apex transition band.
- `Descent-Recovery` learns from the Apex transition band through landing and
  stable recovery.
- Phase-expert checkpoints generate real online candidate states. Bounded
  real-dynamics perturbation trajectories provide boundary information; an
  expert trajectory itself is not a Tube.
- Frozen expert policies generate the first policy-dependent continuation
  labels. Formal expert datasets are normalized under selected
  `pi_up_star`/`pi_down_star` authorities.
- `V_up` and `V_down` estimate expert-conditioned phase continuation and create
  the bootstrap TRAIN-only `Tube_0`.
- One unified Tube-RSI Actor is trained from the Tube and task reward. No expert
  switching is permitted in the final controller.
- After each unified policy `pi_k` is frozen, boundary acquisition uses the
  authoritative dynamics and provenance-complete snapshots; direct qpos/qvel
  dilation is not capability evidence.
- Expansion candidates are labeled under frozen `pi_k`, producing
  policy-conditioned continuation fields `C_up^k` / `C_down^k`. Expert fields
  are not silently reused as unified-policy feasibility estimates.
- `Tube_{k+1}` retains core support and adds evidence-backed boundary support.
  A larger coordinate box by itself is not an expanded capability Tube.
- Policy improvement and Tube expansion repeat under predeclared
  core-preservation, boundary-gain, and convergence gates.
- Only an independent evaluation of the final frozen unified policy over a
  disjoint declared initial-state bank establishes the empirical Jump
  Capability Envelope (JCE/JEL).
- Every learned Tube remains `training_guidance_only=true` and
  `certified_safe=false`.

The final JCE/JEL is empirical and policy-conditioned. It is not a formal
invariant set, guaranteed viability kernel, or proof of safety.

See `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md` for the iteration and leakage
contract.

## Implemented today

The active JIT path has already produced:

- frozen `pi_up_star` and `pi_down_star` phase experts;
- locked first-pass expert-conditioned `V_up` / `V_down` artifacts;
- a 222-entry TRAIN-only learned Soft Tube (117 upstream, 105 downstream);
- validated Tube-RSI runtime with no expert switching;
- completed single-Actor unified PPO training at exactly 10,009,600 PPO
  transitions;
- a Round-1 final checkpoint with successful restore and finite parameters;
- fixed TRAIN Tube panels that reached 16/16 success at the final two
  milestones;
- an independent canonical natural-start diagnostic showing the Round-1 policy
  fails before the jump zone at `yaw_limit`.

The natural-start result is retained as an out-of-domain cold-start diagnostic
under the revised JIT scope. It does not invalidate Tube-conditioned jump
competence and is not the final JCE/JEL gate.

A proposed 50% natural-reset Round-2 was fully preflighted but was superseded
before launch after the research scope was corrected. Its configuration and
prelaunch evidence remain immutable provenance; no Round-2 training run is
claimed.

## Not implemented

The following work remains separately gated:

- freeze the exact current unified checkpoint as the `pi_0` expansion
  authority;
- generic unified-policy real-dynamics boundary acquisition for both phases;
- frozen unified-policy continuation labeling;
- policy-conditioned continuation models `C_up^k` / `C_down^k`;
- evidence-backed `Tube_{k+1}` construction with core retention;
- iterative unified-policy improvement and convergence accounting;
- a final, disjoint empirical JCE/JEL evaluation bank and report.

The existing Soft Tube and TRAIN panels are training artifacts. They cannot be
used as independent final envelope evidence. No current file may claim JCE/JEL,
formal safety, or a final deployment policy until the new protocol is
implemented and experimentally closed.

## Immutable physical contracts

- Model: `assets/orange_bike_4kg_horizontal.xml`
- Historical filename retained: yes; the `4kg` token is not the current mass
- Current run-bound XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- Payload: 2.0 kg
- Hip/knee actuator force limits: +/-50 N m
- Action order: `[steer, rear-wheel drive, hip, knee]`
- Control rate: 50 Hz
- No replacement XML, mesh/collision edits, obstacle changes, matcher-radius
  changes, or hidden task-semantic changes are permitted by envelope iteration.

## Runtime and validation

The configured runtime is:

```text
/home/qy/mujoco_playground/.venv/bin/python
```

Repository validation uses the JIT local preflight and GPU-marked tests. Smoke
and diagnostic interactions must remain separately accounted from PPO training
and final evaluation.

See `docs/METHOD_TWO_PHASE_SOFT_TUBE.md` for the base two-phase method,
`JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md` for the current iterative extension,
`docs/REPOSITORY_LAYOUT.md` for cleanup decisions, and
`docs/EXPERIMENT_STATE.md` plus dated JIT handoffs for recoverable experiment
provenance.
