# JIT Phase U Reference Reward, Airborne RSI, and Diagnostics Design

## Scope

This change keeps one task: `Propulsion-Ascent`. Entering the Apex transition
band remains terminal success. Platform traversal, landing recovery, and the
reference implementation's post-platform deceleration branch are outside this
task and are not represented as reachable Phase U reward stages.

The file `/home/qy/下载/平面跳奖励函数.py` is a read-only formula reference. It is not a runtime dependency and none of its target-position logic is retained.

## Jump Zone and Jump Signal

The authoritative jump zone uses root position in world coordinates:

```text
2.5 m <= root x <= 3.1 m
```

`jump_signal` is an explicit scalar Actor observation with one-shot interval
semantics. It is one only while the root is inside the jump zone on its first
visit. On the first exit from that interval it changes to zero and can never
reopen during the same episode, including if the robot travels backward into
the interval. It is not inferred by the policy from a reward, contact, outcome,
or privileged identifier.

Reset initializes this state from root x without an artificial transition:

- `x < 2.5 m`: signal zero and the one-shot interval remains available;
- `2.5 m <= x <= 3.1 m`: signal one and the one-shot interval is active;
- `x > 3.1 m`: signal zero and the one-shot interval is already consumed.

The approximate front/rear wheel-support booleans are removed from both policy
inputs, event logic, and task diagnostics. The platform-relative structure
clearance field is replaced in the Actor frame by directly measured root
height. Each sensor frame therefore shrinks from 27 to 25 values, and its
three-frame history shrinks from 81 to 75 values. The current `jump_signal` is
appended once, outside that history, so the Actor input is 76 values and is
immediately visible on an airborne RSI reset. The critic contains the complete
76-value Actor input plus 30 privileged values, for 106 values total. Existing
checkpoints are incompatible by design and must fail identity validation.

MJX-Warp's runtime `Data` does not expose geom-paired contacts, so no approximate
wheel-support replacement is introduced. Phase U does not need one: its Apex
event is defined only from root height and vertical motion. Prohibited body and
wheel penetration remain independent physical-failure checks, not support
observations.

Every altitude-dependent shaping reward is multiplied by the current
`jump_signal`. Consequently it is exactly zero before legal jump-zone entry and
again after the first exit. The one-time Apex success bonus requires historical
proof that the jump zone was visited, rather than requiring the current signal
to remain one at Apex. Ordinary posture, speed, survival, action, energy, and
failure terms remain active throughout the short Phase U episode.

## Airborne RSI

Training reset uses an auditable two-component mixture:

- probability `0.95`: the unchanged natural keyframe reset;
- probability `0.05`: a bounded airborne RSI reset centered at
  `x=2.8 m`, `z=2.0 m`, and `vx=2.0 m/s`.

The airborne reset samples independently and reproducibly from:

```text
x  in [2.7, 2.9] m
z  in [1.8, 2.2] m
vx in [1.8, 2.2] m/s
vz in [0.8, 1.2] m/s
```

It retains the authoritative keyframe's y position, unit orientation, joint
positions, and all other velocities at zero. Because every RSI x sample lies
inside the first-visit interval, its `jump_signal` starts at one and its consumed
flag starts false. Event state records that the jump zone has been visited but
does not pre-mark ascent, height qualification, or Apex; those events must still
be observed on subsequent physical steps. The positive RSI `vz` prevents a
high reset from receiving immediate success merely by falling under gravity.

Each state exposes a fixed numeric `reset/source_airborne_rsi` metric. Formal
held-out evaluation always forces the natural reset, and any optional RSI
diagnostic is reported separately. Natural-start success is the only result
that can support expert selection; RSI success cannot inflate that rate.

## Reward Contract

The reward follows the reference formula in degrees for Euler-angle terms and
rad/s for angular-rate terms. It excludes every target-position, distance,
direction, target-reached, platform, and deceleration term.

### Positive components

- `roll`: coefficient `3.0`. For absolute roll `r` in degrees,
  `raw=1-r/5` for `r<=5`, otherwise `raw=-0.1*(r-5)`.
- `pitch`: coefficient `1.0`. Its raw piecewise curve is `[1.0,0.9]` over
  `0..3 deg`, `[0.9,0.5]` over `3..8 deg`, `[0.5,0]` over `8..10 deg`, and
  `-0.1*(p-10)` above `10 deg`.
- `yaw`: coefficient `0.3`. Its raw piecewise curve is `[1.0,0.9]` over
  `0..3 deg`, `[0.9,0.5]` over `3..8 deg`, `[0.5,0.2]` over `8..15 deg`,
  `[0.2,0.05]` over `15..25 deg`, and `max(0.05-0.002*(y-25),0)` above
  `25 deg`.
- `speed`: coefficient `0.2`, with
  `exp(-0.5*((vx-3.5)/0.5)^2)`.
- `survival`: constant `1.5` on each non-reset transition.
- `height`: coefficient `20.0`, and exactly zero unless `jump_signal=1` and
  root `z>=0.35 m`. Its raw value rises linearly from `1.0` at `0.35 m` to
  `1.5` at `0.5 m`, declines linearly to `0.6` at `0.8 m`, and is `0.4`
  above `0.8 m`.
- `apex_success`: one-time `50.0` when Apex is first reached after historical
  jump-zone visitation, even if the one-shot current signal has already closed.

### Negative components

- `action_smoothness`: `-1.5 * 0.0001 * ||a_t-a_(t-1)||^2`.
- `action_magnitude`: `-1.5 * 0.1 * sum(abs(a_t)^1.5)`.
- `pitch_rate`: `-0.15 * 0.125 * omega_pitch^2`.
- `roll_rate` and `yaw_rate`: zero in Phase U because the removed
  post-platform stage was their only activation region.
- `joint_energy`: `-2.0 * dt * (abs(tau_hip*qdot_hip) +
  abs(tau_knee*qdot_knee))`, using measured actuator force and `dt=0.02 s`.
  The reference file's accidental second multiplication by `3.5` is not
  copied.
- `illegal_contact`: `-30.0` on a prohibited-contact transition.
- `physical_failure`: one-time `-30.0` on entry to physical failure.
- `timeout`: one-time `-10.0` at the horizon.

The sum is finite-clipped to `[-50, 50]`. All weighted components and the
unclipped sum are recorded separately, so clipping cannot hide the diagnostic
composition.

## Termination

Apex success is the first transition satisfying all of:

```text
the one-shot jump zone was visited
root z reached at least 0.5 m
vertical velocity previously reached at least +0.05 m/s after reset
current vertical velocity is at most -0.05 m/s
no physical failure is active
```

This is immediate successful termination. It deliberately replaces the prior
wheel-support, structure-clearance, obstacle-relative-x, roll/pitch target-band,
and angular-speed Apex gates. Roll/pitch remain safety limits and reward terms,
not Apex targets. Existing non-finite, roll-limit, pitch-limit,
prohibited-contact, illegal-wheel-contact, and backward-exit failures remain
terminal. The platform-overrun condition is removed from Phase U because it
belongs to a later trajectory region and could otherwise be mistaken for an
intended deceleration boundary. Timeout remains the 200-control-tick horizon.

No Phase U config contains an active platform-start or deceleration-zone reward
parameter after this change.

## Diagnostics and Artifacts

Every rendered representative trace produces:

- the physical MP4 with a synchronized telemetry panel;
- a full-trajectory PNG dashboard;
- a compressed NPZ containing every plotted numeric series;
- a JSON report naming and hashing the artifacts.

The dashboard contains:

1. every weighted reward component, the unclipped sum, and clipped total;
2. root `x/y/z`, obstacle-relative x, the `0.5 m` height threshold, and
   jump-zone bounds;
3. roll/pitch/yaw in degrees and angular velocity x/y/z in rad/s;
4. forward/lateral/vertical velocity;
5. normalized action, control target, hip/knee velocity, force, and mechanical
   power;
6. jump signal, RSI reset source, ascent/height-qualified/Apex and terminal
   events, and end reason.

Plot generation is Host-only and must never step the environment or consume
training/evaluation interactions. Reset states and transitions remain one-to-one
with encoded video frames and diagnostic samples.

## Validation

Tests must demonstrate:

- exact reference piecewise reward values and target-term absence;
- exact zero altitude reward while `jump_signal=0`;
- exact `0 -> 1 -> 0` one-shot signal behavior and no reopening after re-entry;
- reset-time signal/consumed initialization below, inside, and above the zone;
- immediate Actor visibility of `jump_signal=1` on airborne RSI reset;
- reproducible 5% reset selection and bounded RSI state values;
- natural-only held-out evaluation despite training RSI configuration;
- Apex termination and absence of an unreachable deceleration/platform branch;
- fixed JAX pytree keys and finite batched GPU reset/step;
- absence of approximate wheel-support booleans from Actor and critic inputs;
- exact height-qualified, prior-ascent, and descending Apex event behavior;
- plot data/PNG/video/report existence, aligned sample counts, and no renderer
  environment step;
- checkpoint rejection across the changed observation/config identity.
