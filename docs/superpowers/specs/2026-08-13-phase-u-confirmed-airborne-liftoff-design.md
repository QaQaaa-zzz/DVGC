# Phase U Confirmed-Airborne Liftoff Design

## Evidence and root cause

The completed 2 kg Phase U run
`phase_u_2kg_env256_airborne_gate_998400_20260813_seed720802` used 998,400
training transitions and produced no Apex-success parent.  Fixed evaluation
improved from no liftoff to an apparent 8/8 `liftoff_seen` at 505,600,
755,200, and 998,400 transitions, but never reached stable airborne,
ascending, clearance, or Apex.

Offline reconstruction of the saved timing-aligned traces against the
authoritative XML proves that this apparent liftoff was a one-wheel event.  At
the 998,400 checkpoint, tick 22 had front tire bottom `-0.0017 m` and rear tire
bottom `+0.0127 m`; the deployable dual-wheel liftoff predicate was false on
every tick.  The same pattern occurs at 505,600 and 755,200.  Nevertheless,
the external event runtime latched `liftoff_seen` because both wheels were no
longer simultaneously in the narrow stable-support band.  This unlocked the
legal-liftoff bonus and airborne shaping while the retained physical takeoff
deadline correctly remained armed.

## Selected contract

`liftoff_seen` becomes true only when all conditions hold:

```text
previous.jump_window_entered
AND current ApexBandSignals.stable_airborne
AND no physical failure
```

`ApexBandSignals.stable_airborne` is the existing pure-JAX deployable signal
derived from the environment's confirmed-airborne counter.  Under the current
authoritative configuration, Takeoff confirmation is driven by the physical
dual-wheel liftoff predicate for two consecutive control ticks.  No legacy
phase name, reward result, success label, matcher, reference index, or future
information enters the external event adapter.

The latch remains monotonic.  A one-wheel lift, momentary support loss, or
early airborne state:

- does not terminate;
- is not penalized;
- is not Phase U success;
- does not set `liftoff_seen`;
- does not unlock legal-liftoff, stable-airborne, ascent, clearance, or Apex
  reward.

The jump-window latch, physical failure gates, roll/pitch/contact/nonfinite
limits, retained takeoff deadlines, XML, action mapping, thresholds, reward
weights, PPO layout, optimizer, reset, observation, and horizon remain
unchanged.

## Alternatives rejected

Reading `dual_wheel_liftoff_seen` directly from legacy environment info would
match the current model but would couple the formal two-phase event runtime to
legacy Takeoff bookkeeping.  Requiring only both wheels to be unsupported
would still accept a one-tick bounce.  The selected confirmed-airborne signal
is already part of the formal pure-JAX Apex schema and supplies the required
temporal persistence.

## Verification and next run

Red-green tests must prove that post-window support loss without confirmed
airborne does not latch liftoff or unlock rewards, while confirmed airborne
does latch monotonically and retains JIT/vmap behavior.  Event-order tests must
reflect the explicit temporal dependency.  After targeted tests, compileall,
full pytest, local preflight, and a fresh runtime gate, run one new
256-environment 6,400-transition formal-path smoke.  Only a passing smoke may
create a new exact run-bound authorization for another aligned 998,400-
transition Phase U run.  No old checkpoint is resumed, and no formal Tube is
declared.
