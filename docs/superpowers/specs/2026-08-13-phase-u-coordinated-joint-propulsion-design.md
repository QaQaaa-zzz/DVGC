# Phase U Coordinated Joint-Propulsion Credit Design

## Objective

Test one explanation for the completed v7 Phase U failure: the physically
correct synchronized-wheel upward-velocity signal is still downstream of the
launch action and therefore too sparse for PPO to discover a useful jump from
natural starts. Add a bounded signal for actual, coordinated pre-liftoff joint
motion inside the legal jump window without tracking the kinematic guideline
or rewarding action commands.

## Evidence

The v7 run completed all 998,400 authorized PPO-training transitions. Every
fixed panel at 0, 102,400, 256,000, 505,600, 755,200, and 998,400 reached the
legal jump window in 8/8 rollouts, but all 48 rollouts remained grounded and
ended at `takeoff_missed_liftoff_deadline`. The deterministic policies retained
the natural posture: knee control stayed at 2.5 rad and hip control remained
near -1.2 rad. Their synchronized-wheel upward-velocity credit was exactly
zero. The final stochastic aggregate obtained only 0.00167 of that component.

Existing fixed diagnostics establish the action direction without a new
physical search. Positive hip impulses can produce confirmed liftoff, but
hip-only launches later violate pitch. Adding positive knee action increases
height but does not by itself solve attitude. Negative knee commands at the
natural 2.5-rad upper limit are clipped and do not launch. Therefore neither
hip-only credit nor more knee exploration is an acceptable hypothesis.

The authoritative mapping and physical state establish:

```text
hip qvel > 0   : hip advances away from the -1.2-rad compressed start
knee qvel < 0  : knee extends away from the 2.5-rad compressed upper limit
```

Across all six grounded v7 deterministic traces, the largest post-window
minimum of those two signed velocities was 0.1285 rad/s. The new deadband is
therefore 0.15 rad/s. A 2.0 rad/s full target is a broad engineering scale:
well below the multi-rad/s movement seen in existing physical launch
diagnostics and the retained kinematic guideline, but more than an order of
magnitude above grounded background motion.

## Alternatives

1. **Coordinated actual joint-velocity progress (selected).** It supplies
   action-direction feedback before liftoff, requires both launch joints to
   advance, and is derived from deployable physical state.
2. Reward positive knee action. Rejected because it rewards a command rather
   than a physical result, can be exploited at saturation, and duplicates an
   already rejected exploration-only hypothesis.
3. Reward hip position or velocity alone. Rejected because existing hip-only
   diagnostics lift off but end in pitch failure.
4. Lower the wheel-velocity deadband or increase its weight. Rejected because
   that signal remains downstream of liftoff; lower thresholds risk restoring
   solver/contact-jitter credit rather than launch-direction credit.

## Reward Contract

Replace only the v7 `dual_wheel_lift_progress` source with a v8 coordinated
joint-propulsion source while retaining its public component name and weight:

```text
coordinated_joint_propulsion_weight: 4.0
coordinated_joint_velocity_deadband: 0.15 rad/s
coordinated_joint_velocity_target: 2.0 rad/s
```

For each real control tick, compute from the resulting physical state:

```text
hip_progress_velocity = max(hip_qvel, 0)
knee_progress_velocity = max(-knee_qvel, 0)
synchronized_velocity = min(hip_progress_velocity,
                            knee_progress_velocity)
normalized = clip((synchronized_velocity - 0.15) / (2.0 - 0.15), 0, 1)
credit = 4.0 * normalized * angular_rate_quality
```

For stable logging compatibility, retain the metric name
`dual_wheel_lift_progress` for this one hypothesis, but change reward semantics
and manifest fields so no artifact can confuse v7 wheel velocity with v8 joint
propulsion. The design does not claim the metric name describes a formal
physical contract; the hashed reward manifest is authoritative.

Credit is exactly zero unless the existing legal jump window is active. It is
also exactly zero if either joint fails to advance in the approved direction,
at or below the evidence-derived deadband, or when the existing angular-rate
quality is zero. It does not use action, control target, reference index/time,
future information, success, termination reason, or any metadata. It does not
imply liftoff, stable airborne, Apex success, continuation feasibility, done,
or safety.

## Adapter and Runtime Boundaries

`PhaseExpertEnvAdapter` reads the actual hip and knee qvel values from the
post-step MJX state using immutable joint DoF addresses resolved from the
authoritative model. The adapter may accept an injected extractor in unit
tests. It must not modify `dvgc/env.py`, the actor observation, history,
snapshot semantics, reset, action mapping, XML, event latches, deadlines, or
physical-failure selection.

The v7 wheel-clearance runtime helper remains implemented and tested because
it is part of the formal two-phase geometry contract. It is no longer consumed
by this Phase U reward hypothesis, so the adapter-owned previous-clearance
timing field is removed rather than maintained as unused mutable state.

## Test Contract

Red-green tests must prove:

- exact schema rejection and reward-hash drift for the three new fields;
- zero credit before the legal window;
- zero at 0.14 and 0.15 rad/s synchronized velocity;
- half credit at 1.075 rad/s and full credit at/above 2.0 rad/s;
- one joint moving alone earns zero;
- the wrong sign on either physical joint earns zero;
- angular-rate quality still qualifies the component;
- credit does not imply liftoff, Apex success, task failure, physical failure,
  timeout, or done;
- the adapter reads real post-step qvel addresses and does not derive the
  signal from action or control targets;
- pre-window task-progress reward remains zero, including early airborne.

## Qualification and Experiment Gate

After focused tests, run compileall, the full suite, local preflight, and one
fresh 64+32 managed runtime gate. Refresh only the source-bound threshold
provenance hash; threshold values remain byte-identical.

Then run one fresh 256-environment, 6,400-training-transition engineering
smoke under a run-bound authorization. A clean smoke qualifies exactly one
fresh-initialization formal run capped at 998,400 aligned PPO-training
transitions with the unchanged fixed evaluation seeds and checkpoint schedule.
The v7 run is complete and must not be resumed.

The hypothesis is supported only if fixed evaluation begins producing real
liftoff/stable-airborne coverage without reintroducing persistent roll/pitch
failure, and ultimately Apex parents. It is falsified if the full authorized
run again remains grounded with zero joint-propulsion credit, or if reward
improves while physical launch metrics degrade. No result automatically
declares `pi_up_star`, `V_up`, a Soft Tube, Phase D training, or unified PPO.
