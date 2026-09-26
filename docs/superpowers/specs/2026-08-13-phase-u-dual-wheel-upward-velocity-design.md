# Phase U Dual-Wheel Upward-Velocity Progress Design

## Completed v6 evidence

The completed formal run
`phase_u_2kg_env256_dual_wheel_lift_v6_998400_20260813_seed721402`
consumed all 998,400 authorized PPO-training transitions.  All six fixed
checkpoint panels reached the legal jump window in 8/8 rollouts, but every
rollout remained grounded and ended at the unchanged
`takeoff_missed_liftoff_deadline`.  There were no Apex successes, candidate
snapshots, continuation probes, physical failures, roll/pitch violations,
illegal contacts, saturation, numerical faults, or identity faults.

All 157 checkpoint sidecars validate recursively.  All six outcome accounts
close, and all 48 MP4/48 timing-aligned NPZ failure artifacts exist and match
their declared hashes.  Representative videos at 0, 505,600, and 998,400
transitions visually confirm the same grounded behavior.

The v6 `minimum_wheel_terrain_clearance` term did not measure useful lift
progress while the wheels remained in contact.  Host MuJoCo forward
reconstruction of the saved fixed traces found post-window minimum-clearance
excursions only between roughly -0.00017 m and +0.00018 m.  These solver-scale
gaps generated a small positive v6 reward despite both wheels visibly
remaining supported.  The 505,600 checkpoint received the largest fixed
dual-wheel-height return, about 0.089, but neither deterministic nor frozen
stochastic evaluation produced confirmed liftoff.  The policy ultimately
selected safe ground travel, about +9 forward reward, followed by one -20
task-failure penalty.

## Alternatives

1. **Increase the height-progress weight.** Rejected because this magnifies
   solver-scale contact gaps without supplying the direction of the required
   action.
2. **Apply only a height deadband.** Rejected because height remains nearly
   constant until the contact constraint has already been broken, leaving the
   temporal-credit gap intact.
3. **Use MuJoCo body `cvel` directly.** Rejected after audit: the wheel-body
   spatial velocity is expressed at MuJoCo's body reference and includes wheel
   rotation; the two values were about -3 and +6 m/s on visibly grounded
   traces and did not equal wheel-support vertical velocity.
4. **Use consecutive pure-JAX wheel support geometry (selected).** The actual
   world-space support bounds already define each wheel's terrain clearance.
   Differencing each wheel separately across real control ticks gives its
   support-bound vertical velocity.  Taking the minimum requires both wheels
   to rise in the same tick.

## Single hypothesis

Phase U has encountered stochastic launch events but converges to the grounded
local optimum because the current dense lift component supplies no clean
pre-liftoff action-direction credit.  Replacing the height-only score with a
deadbanded, synchronized dual-wheel upward-velocity score will increase useful
low-rate liftoff coverage without rewarding ground-contact jitter.

This is one reward-semantics hypothesis.  No PPO, exploration, reset, model,
threshold, deadline, action, observation, or safety variable changes with it.

## Runtime and timing contract

The external two-phase adapter exposes a pure-JAX helper that returns the two
collision-wheel terrain clearances in stable manifest order.  On reset,
`PhaseExpertEnvAdapter` records the current two-value vector under its own
adapter-owned `phase_expert/*` info namespace.  On every step it computes:

```text
wheel_upward_velocity_i =
    (current_wheel_clearance_i - previous_wheel_clearance_i) / ctrl_dt

minimum_wheel_upward_velocity = min_i(wheel_upward_velocity_i)
```

The previous vector is then replaced by the current vector for the next
control tick.  This uses the real previous and current physics frames and may
not duplicate a frame or reconstruct history from CSV.  It remains compatible
with `jax.jit`, `jax.vmap`, and batched MJX state.  It does not modify
`dvgc/env.py`, `env.step`, observation, action mapping, XML, or the environment
event latch.

The adapter-owned previous-clearance value is timing state, not a deployable
model feature, success label, metadata feature, or new snapshot claim.  The
current `ApexBandSignals.minimum_wheel_terrain_clearance` remains available for
geometry diagnostics and future feature allowlisting but is no longer the
source of this reward component.

## Reward contract

Replace the v6 height target with two explicit hashed fields:

```text
dual_wheel_upward_velocity_deadband = 0.02 m/s
dual_wheel_upward_velocity_target = 0.20 m/s
```

The deadband is evidence-bounded: it is above the measured grounded-trace
maximum of 0.0106 m/s and is one tenth of the already authoritative
`takeoff_liftoff_vz = 0.20 m/s` physical threshold.  The component is:

```text
rate_quality = clip(
    1 - angular_speed /
        (max_apex_angular_speed * angular_rate_penalty_cap_ratio),
    0,
    1,
)

dual_wheel_lift_progress =
    dual_wheel_lift_progress_weight
    * window_entered
    * clip(
        (minimum_wheel_upward_velocity - deadband)
        / (target - deadband),
        0,
        1,
      )
    * rate_quality
```

The retained weight is 4.0.  At or below 0.02 m/s the component is exactly
zero; at 0.11 m/s it is 2.0; at or above 0.20 m/s it is 4.0 before rate
qualification.  A single wheel rising while the other remains supported earns
zero.  Before legal-window entry it remains exactly zero, including early
airborne.  It never implies liftoff, stable airborne, Apex membership, success,
done, or a continuation label.

## Preserved contracts

- authoritative XML path and 2 kg payload;
- +/-50 N m hip/knee limits and action order;
- natural Phase U reset and real three-frame FIFO;
- legal jump-window latch and position-based takeoff deadlines;
- pre-window task-progress reward exactly zero;
- early airborne nonterminal and nonsuccess semantics;
- full Apex transition-band success contract;
- roll/pitch/contact/backward/nonfinite physical failures;
- all PPO, network, optimizer, exploration, horizon, seed, and evaluation
  settings;
- candidate acquisition requirement of real held-out Apex success and at least
  eight independent successful parents.

## Red-green and run gate

RED must prove that v6 rewards solver-scale positive height gaps and has no
consecutive-tick dual-wheel velocity state.  GREEN must prove:

- exact zero before legal-window entry;
- exact zero at and below the 0.02 m/s deadband;
- proportional credit between 0.02 and 0.20 m/s;
- bounded full credit at and above 0.20 m/s;
- minimum-of-two-wheel synchronization;
- angular-rate qualification remains unchanged;
- reset initializes the exact current physical clearance vector;
- each step uses the immediately previous real control tick and updates it
  monotonically in timing order;
- no liftoff/success/done implication;
- JIT/VMAP/batched runtime compatibility;
- reward manifest/hash binding and exact-schema rejection.

After targeted tests, run compileall, the full suite,
`scripts/local_preflight.sh`, and a fresh 64+32 runtime gate.  Then run one
fresh 256-environment engineering smoke.  Only a clean smoke permits one new
fresh-initialization, run-bound formal authorization capped at 998,400 aligned
PPO-training transitions.  The completed v6 run is immutable provenance and
must not be resumed.

