# Phase U Dual-Wheel Lift Progress Design

## Evidence and root cause

The completed v5 run
`phase_u_2kg_env256_rate_ascent_998400_20260813_seed721302` consumed
998,400 PPO-training transitions without an Apex success.  All six held-out
panels reached the legal jump window, but every rollout remained grounded and
ended at the unchanged position-based missed-liftoff deadline.  Frozen-policy
stochastic audits at 505,600 and 896,000 transitions produced no confirmed
liftoff in 64 rollouts each.  At 896,000 only three of 64 trajectories reached
root `vz >= 0.2 m/s`; the largest excursion also reached 6.37 rad/s angular
speed.

The v5 rate qualification successfully removed the earlier high-rotation
reward exploit, but the formal adapter still exposes no dense reward that
distinguishes synchronized wheel lift from grounded root-height oscillation.
`ascent_progress` observes root vertical velocity.  The one-shot legal-liftoff
and stable-airborne bonuses arrive only after two physically valid airborne
confirmation ticks.  The unchanged deadline is correctly armed only after the
legal jump-window latch and triggers when the robot front reaches the approved
pre-obstacle position; changing it would not supply the missing control credit.

The old environment contains legacy dual-wheel takeoff shaping, but the Gate C
adapter owns the Phase U reward and deliberately discards the old route's
reward.  The missing quantity must therefore be reconstructed in the pure-JAX
two-phase runtime, not imported from the legacy reward or event matcher.

## Alternatives

1. Increase the one-shot liftoff/stable-airborne bonuses.  Rejected because the
   frozen policy almost never reaches those events, so their magnitude does not
   make the preceding action sequence learnable.
2. Move the missed-liftoff deadline.  Rejected because it changes an approved
   physical task boundary without identifying a timing defect and still gives
   no synchronized-wheel gradient.
3. Add a window-gated, rate-qualified minimum-wheel-clearance progress term.
   Selected because it supplies deployable, present-time physical credit for
   the missing transition while preserving every success and failure contract.

## Single hypothesis

The next run changes one scientific hypothesis only:

> Phase U fails to acquire a reproducible launch because the formal adapter has
> no dense credit for synchronized wheel lift between window entry and confirmed
> liftoff.  Rewarding bounded minimum wheel clearance in that interval will
> increase low-angular-rate liftoff coverage without rewarding grounded root
> bounce or high-rotation ejection.

## Runtime signal

Extend `ApexBandSignals` with:

```text
minimum_wheel_terrain_clearance
```

The external pure-JAX runtime computes collision support bounds for every
collision-relevant robot geom, selects the existing `wheel_mask`, computes each
wheel's clearance to the ground or immutable obstacle top beneath it, and
returns the minimum across all wheel geoms.  This is a deployable physical
geometry signal.  It does not use the legacy phase, matcher, reward, reference
time/index, result fields, or host `mj_geomDistance`.

## Reward contract

Add two explicit, hashed Phase U reward fields:

```text
dual_wheel_lift_progress_weight = 4.0
dual_wheel_lift_progress_target = 0.015 m
```

The component is:

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
    * clip(minimum_wheel_terrain_clearance /
           dual_wheel_lift_progress_target, 0, 1)
    * rate_quality
```

Properties:

- before legal window entry it is exactly zero, including early airborne;
- grounded or downward-penetrating wheel geometry receives zero;
- both wheels must rise because the minimum wheel clearance is used;
- it is bounded by 4.0 per control tick and uses no future information;
- high-angular-rate ejection is continuously suppressed by the same v5 rate
  quality used for ascent;
- it never sets liftoff, stable-airborne, Apex success, done, or an outcome;
- all roll/pitch/contact/nonfinite failures and the position deadline remain
  unchanged.

All other Phase U reward coefficients, PPO hyperparameters, reset, observation,
network, exploration prior, XML, actuator limits, action mapping, thresholds,
deadline positions, horizon, seed namespace, and evaluation protocol remain
fixed.

## Red-green and validation

RED must demonstrate that the current implementation has no component and no
runtime wheel-clearance signal.  GREEN must prove:

- exact zero before legal-window entry, even when wheel clearance is positive;
- exact zero for grounded/negative clearance;
- proportional credit at half the 0.015 m target;
- bounded full credit at and above the target;
- reduced credit at elevated angular rate and zero at the configured cap;
- no effect on liftoff, Apex success, done, or task/physical failure;
- JIT and VMAP compatibility of signal extraction and reward evaluation;
- stable reward manifest/hash and exact-schema rejection for missing fields.

After focused tests, run compileall, the full test suite,
`scripts/local_preflight.sh`, and a fresh managed runtime gate because the
runtime/reward fingerprint changes.  Only then run one 256-environment PPO
smoke.  A clean smoke permits one fresh-initialization run-bound experiment up
to 998,400 aligned training transitions.  The prior v5 checkpoint must not be
resumed.

Fixed evaluation remains the authority.  Snapshot acquisition still requires
real Apex successes from at least eight independent parents.  No result from
this iteration is a formal expert, feasibility model, or Soft Tube by itself.
