# Phase U Rate-Qualified Window Ascent Single-Hypothesis Design

## Evidence

The weight-8 formal run
`phase_u_2kg_env256_window_ascent8_998400_20260813_seed721202` completed all
998,400 training transitions but produced no Apex success or candidate parent.
The first four held-out checkpoints remained grounded. At 755,200 and 998,400,
all eight held-out rollouts instead ended at `pitch_limit`; their mean summed
ascent reward was about +59.8/+59.9 and mean return became positive even though
physical-failure rate was 1.0. Peak angular speed was 24.39/23.37 rad/s.

The final trace is not a latch defect. Host MuJoCo forward audit shows rear
wheel support at tick 22, dual-wheel separation beginning at tick 23, and both
wheels clearly airborne at tick 24. By then angular speed is about 23.4 rad/s;
the second confirmation tick coincides with the retained pitch failure, so a
legal two-tick confirmed liftoff never occurs. The policy has learned a
high-rotation ejection that harvests integrated positive `com_vz`.

Existing fixed impulse evidence separates the lower-rate boundary: hip
actions +0.10/+0.15 produce only modest vertical velocity and maximum pitch
about 0.06--0.18 rad without physical failure, while confirmed-liftoff impulses
at +0.25 or above later hit pitch limit. Increasing ascent weight again,
weakening angular penalties, changing the deadline, or relaxing pitch safety
is therefore rejected.

## Single hypothesis

Keep `ascent_progress_weight=8` and multiply the existing window-gated
positive-vertical-velocity score by one bounded current-state quality:

```text
rate_quality = clip(
    1 - angular_speed /
        (max_abs_apex_angular_speed * angular_rate_penalty_cap_ratio),
    0,
    1,
)

ascent_progress =
    ascent_progress_weight
    * window_entered
    * clip(com_vz / target_vertical_velocity, 0, 1)
    * rate_quality
```

The existing cap ratio is 4.0, so quality reaches zero at four times the Apex
angular-rate threshold. This is a smooth reward qualification, not a new hard
failure or a relaxation of Apex membership. It uses only current deployable
physical signals and already hashed config/threshold fields.

## Preserved contracts

- Pre-window ascent reward remains exactly zero, including early airborne.
- Early airborne remains nonterminal and does not imply liftoff or success.
- Low-rate positive vertical motion in the legal window remains rewarded.
- High-rate motion earns zero ascent progress but is still handled by the
  unchanged roll/pitch/contact/nonfinite hard failures and angular penalty.
- Liftoff, stable-airborne, clearance, Apex, success, deadline, XML, 2 kg
  payload, +/-50 N m limits, action mapping, reset, observation, exploration,
  PPO, network, optimizer, horizon, and every reward coefficient are unchanged.
- Reward remains bounded, future-free, reference-time-free, and label-free.

## Validation and experimental gate

Red-green tests must first prove current weight-8 code incorrectly pays full
ascent credit at extreme angular speed. GREEN must prove full credit at zero
rate, partial credit below the cap, zero at/above the cap, and zero before the
window. Bump the reward semantics/hash because executable meaning changes.

Then run focused/full static validation, preflight, a fresh 64+32 runtime gate,
and one fresh 256-environment 6,400-transition smoke. Only a clean smoke permits
a separate fresh-initialization formal authorization. Candidate snapshots still
require real held-out Apex success and independent parent diversity.

