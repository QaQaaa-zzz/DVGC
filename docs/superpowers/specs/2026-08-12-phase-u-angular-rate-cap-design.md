# Phase U Angular-Rate Penalty Cap Design

## Evidence

The isolated `angular_rate_penalty_weight: 0.25 -> 1.0` run
`phase_u_2kg_angrate1_env512_998400_20260812_seed720102` entered a contractual
Gate Pause at 755,200 training transitions. Fixed evaluation produced no Apex
success or candidate state. The 256k, 512k, and 755.2k checkpoints each ended
8/8 rollouts at `pitch_limit`.

The timing-aligned traces identify an impulsive hip command as the immediate
mechanism. At 102.4k the hip target remains close to its -1.2 rad reset and the
peak angular speed is 0.26 rad/s. At 755.2k it moves from -1.2 to +0.43 rad in
two control ticks and immediately returns near -1.21 rad; peak angular speed is
24.25 rad/s.

The current reward clips `angular_speed / apex_max_angular_speed` at 1.0. The
Apex threshold is 1.2466 rad/s, so the reward cannot distinguish a small
threshold exceedance from the observed 19.45x exceedance. Increasing only the
weight preserved that loss of severity information.

Offline counterfactual scoring of one fixed trace per checkpoint shows that a
cap ratio of 8 leaves the stable 0 and 102.4k trajectories unchanged while
increasing the accumulated angular-rate cost by 49.86, 51.49, and 52.23 for
the 256k, 512k, and 755.2k pitch failures respectively.

## Single hypothesis

Expose an auditable bounded field:

```text
phase_u_reward.angular_rate_penalty_cap_ratio
```

and set it to 8.0 in the stable Phase U smoke and formal configurations. Keep
`angular_rate_penalty_weight` at 1.0. The component becomes:

```text
-weight * clip(angular_speed / apex_max_angular_speed, 0, cap_ratio)
```

The term remains bounded and observable. At high angular rate its maximum is
-8 per control tick, which exceeds the maximum combined +6 ascent and
clearance shaping available in a tick. At sub-threshold rates it is identical
to the current reward.

## Preserved contracts

This experiment does not change XML, the 2 kg payload, +/-50 N m actuator
limits, action mapping, observation, reset, jump-window semantics, early
airborne semantics, thresholds, deadline, PPO layout, exploration standard
deviation, optimizer, horizon, or any other reward component. Progress before
the legal jump window remains zero. Roll, pitch, contact, and nonfinite hard
failures remain active.

The total reward remains clipped to [-50, 50]. The new cap must be finite and
strictly positive and participates in the reward-contract hash.

## Rejected alternatives

- Another weight increase is rejected because it still treats all rates above
  the old cap equally.
- Increasing action smoothness is rejected for this round because it penalizes
  every action channel and does not directly preserve angular-rate severity.
- Gating all ascent reward on stable airborne is rejected because it would
  remove the dense learning signal before liftoff is established.
- Changing action mapping, hip limits, safety thresholds, XML, or reset is out
  of scope.

## Validation and stopping

Red-green tests must prove default compatibility, explicit cap behavior,
finite/positive validation, stable config selection, boundedness, and unchanged
pre-window progress gating. After full static/runtime requalification, run one
fresh 512-environment 12,800-transition smoke. Only a clean smoke permits a
fresh-initialization formal authorization up to 998,400 aligned transitions.
The same fixed checkpoint and Gate Pause protocol remains authoritative.

