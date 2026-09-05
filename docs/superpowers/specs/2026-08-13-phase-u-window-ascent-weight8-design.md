# Phase U Window-Ascent Weight 8 Single-Hypothesis Design

## Evidence

The fresh v4 run
`phase_u_2kg_env256_window_ascent_998400_20260813_seed721102` completed its
full 998,400-transition authorization. All six held-out panels reached the
legal jump window in 8/8 rollouts, but none produced confirmed liftoff,
stable airborne, clearance, or Apex success. All 48 rollouts ended at the
unchanged missed-liftoff deadline without physical failure, roll/pitch
violation, illegal contact, or action saturation.

The v4 gate worked: held-out policies now receive bounded positive
`ascent_progress` inside the legal window before liftoff (0.115--0.599 per
episode across checkpoints). Stochastic PPO also sampled confirmed liftoff in
46 of 156 rollout blocks. Those event-bearing blocks averaged approximately
+0.74 liftoff, +1.37 stable-airborne, and +3.93 ascent credit, but -32.96
angular-rate cost and -54.65 total return. Blocks without liftoff averaged
-8.11 angular-rate cost and -19.28 return. The learned deterministic policy
therefore keeps the hip near -1.25 rad, the knee near 2.50 rad, and remains
grounded.

Previous cap-1 experiments produced high-rate pitch failures, while cap 4 and
cap 8 produced conservative no-liftoff policies. Reopening the angular-rate
cap bracket is therefore rejected. The new evidence instead isolates the
scale of the newly available, correctly gated ascent credit.

## Single hypothesis

Change only the stable Phase U value:

```text
phase_u_reward.ascent_progress_weight: 4.0 -> 8.0
```

At cap ratio 4, the angular-rate component can contribute -4 per control tick.
The old maximum +4 ascent term can merely cancel that term before attitude and
other costs. A maximum +8 remains bounded but can give genuine positive
vertical velocity a net-positive local ordering after legal-window entry.

## Preserved contracts

- `ascent_progress` remains exactly zero before legal-window entry.
- Early airborne remains nonterminal and neither liftoff nor success.
- Liftoff, stable-airborne, clearance, Apex, and success predicates are
  unchanged.
- Roll, pitch, contact, backward-motion, platform-edge, and nonfinite hard
  failures are unchanged.
- The reward still uses current deployable `com_vz`, is future-free and
  label-free, and remains clipped to [-50, 50].
- XML payload 2 kg, +/-50 N m limits, action mapping, reset, observation,
  thresholds, deadline, exploration, PPO, network, optimizer, horizon, and
  evaluation seeds are unchanged.

## Red-green and dynamic gates

Tests first require both stable Phase U configurations to resolve weight 8.0
and prove an in-window `com_vz=0.5 m/s` state receives +4 while the same state
before the window receives zero. RED must fail against the current weight 4.0.
GREEN changes only the two stable configuration values.

After targeted tests, compileall, full pytest, preflight, and a fresh runtime
gate, one fresh 256-environment 6,400-transition smoke may run. A clean smoke
permits one fresh-initialization formal authorization up to 998,400 aligned
training transitions. It must not resume an old checkpoint. Candidate
acquisition still requires real held-out Apex successes and independent parent
diversity. No formal V_up, Soft Tube, Phase D, or unified PPO follows merely
from this change.

