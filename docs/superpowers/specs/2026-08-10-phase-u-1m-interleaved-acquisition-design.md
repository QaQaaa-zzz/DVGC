# Phase U 1M Interleaved Acquisition Design

## Scope

This design extends the validated Gate C1 smoke implementation into one
authorization-gated, persistent Phase U run capped at 1,000,000 total
environment transitions. It changes one scientific hypothesis: the Phase U
reward. Natural reset, XML, physics, action mapping, network, PPO optimization
settings, observation normalization, episode horizon, safety limits, and the
two-phase success contract remain unchanged.

The run produces checkpoint controllers and, only when evidence permits,
candidate snapshots and bounded continuation diagnostics. It does not produce
a formal `V_up`, learned soft Tube, Phase D expert, unified PPO, or JCE/JEL.

## Guideline boundary

`data/reference_jump.csv` supplies only broad engineering scales for velocity,
clearance, posture, and the Apex spatial band already frozen in the threshold
manifest. Phase U always starts at the audited natural stable reset. The run
must not restore a reference state, replay a reference action, imitate the
reference, or compute distance to a time-indexed reference point.

## Reward contract

Each reward component is finite, independently bounded, and published as a
static Brax metric. Online physical events are allowed; dataset labels, future
information, reference time/index, legacy phase, and matcher results are not.

The components are:

- `forward_propulsion`: `0.5 * clip(vx / 3.75, 0, 1)`. This is the only positive
  task-progress term available before legal window entry.
- `jump_window_progress`: one-time bonus `2.0` on the monotonic transition from
  `jump_window_entered=false` to `true`.
- `ascent_progress`: after legal window entry,
  `4.0 * clip(com_vz / 1.0, 0, 1)`.
- `clearance_progress`: after legal window entry, `2.0` times the clipped linear
  progress from a declared `-0.30 m` clearance floor to the immutable Apex
  minimum clearance.
- `apex_approach`: after legal window entry plus stable-airborne and ascending
  latches, `2.0` times the mean of bounded proximity scores for the Apex
  vertical-speed, clearance, roll, pitch, angular-speed, forward-velocity, and
  obstacle-relative-x contracts. These are distances to the physical band,
  never to a reference trajectory point.
- `apex_success_bonus`: one-time `30.0` bonus on physical Apex-band entry.
- `attitude_penalty`: `-0.5` times the clipped mean normalized absolute roll and
  pitch.
- `angular_rate_penalty`: `-0.25 * clip(angular_speed / limit, 0, 1)`.
- `illegal_contact_penalty`: `-20.0` when the deployable signal reports illegal
  contact.
- `action_smoothness_penalty`: `-0.02 * mean((a_t-a_(t-1))^2)`.
- `action_magnitude_penalty`: `-0.005 * mean(a_t^2)`.
- `physical_failure_penalty`: one-time `-20.0` on a retained physical failure.
- `task_failure_penalty`: one-time `-20.0` on a post-window task deadline.

The total is clipped to `[-50, 50]`. `prelaunch_airborne` remains telemetry and
does not appear in the reward. Legal-window entry is monotonic for reward
gating, so all jump/ascent terms are zero before first legal entry but remain
available during subsequent ascent after leaving the narrow ground window.

## Reset and evaluation

Phase U reset is the unchanged audited natural stable reset with bank resets,
domain randomization, and observation noise disabled. Fixed held-out evaluation
uses the unchanged deterministic policy and fixed seed namespace. It reports
outcomes plus window reach, liftoff, stable-airborne, ascending, clearance,
Apex success, roll/pitch violations, illegal contact, physical failure,
forward-velocity retention, signal extrema/margins, reward-component sums, and
action saturation.

Because the fixed environment and deterministic policy can produce identical
rollouts, independent parent diversity is measured in a separately seeded
stochastic online acquisition protocol. A unique parent requires a distinct
rollout seed and trajectory content hash.

## Budget and checkpoints

The requested training ceiling is 1,000,000 total environment transitions.
Requested checkpoints are 0, 100k, 250k, 500k, 750k, and 1M. PPO updates remain
whole rollout blocks; each requested checkpoint records its first non-lower
effective aligned transition count. Training, Brax evaluation, external fixed
evaluation, stochastic acquisition, continuation labeling, and runtime smoke
are separate counters.

The runner stays one persistent process. It emits atomic `status.json`, append-
only `metrics.jsonl`, checkpoint identity sidecars, source/config/XML hashes,
PID/process records, and a reproducible warm-start resume command. The installed
Brax checkpoint contains observation normalizer plus policy/value parameters,
not optimizer state. Sidecars must record this truth and must never claim
`full_training_state=true`; restart is an auditable warm start, not bitwise
continuation.

## Candidate acquisition

Candidate acquisition is eligible only when fixed held-out Apex success is
nonzero, at least eight unique successful stochastic parent trajectories exist,
and there is no numerical or contract failure. Eligible rollouts harvest real
timing-explicit v4 snapshots across pre-window, entry, propulsion, liftoff,
ascent, and Apex strata, including success, near-success, failure, and physical
boundary cases. Candidate states are not a Tube.

Continuation probing is separately budgeted and checkpoint-policy-bound. Early
labels are provisional acquisition diagnostics. Formal Phase U labels require
re-labeling accumulated candidates under the later selected `pi_up_star`.

## Pause conditions

Pause on NaN/Inf, optimizer failure, timing/history/snapshot corruption, source
hash drift, non-closed continuation accounting, feature/parent leakage, severe
action saturation, obvious reward hacking, or repeated held-out physical
degradation. Low success at 100k, absence of a Tube at an early checkpoint, or
not yet reaching 1M is not by itself a pause condition.
