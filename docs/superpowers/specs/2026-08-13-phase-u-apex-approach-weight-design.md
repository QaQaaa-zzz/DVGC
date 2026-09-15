# Phase U Apex-Approach Weight Design

## Evidence

The stable-airborne bridge run
`phase_u_2kg_stable16_liftoff8_cap8_env512_998400_20260813_seed720402`
entered a closed Gate Pause at 256,000 training transitions. All 24 held-out
rollouts reached the legal jump window, but none lifted off or reached Apex;
all ended at the unchanged missed-liftoff deadline without a physical failure.
All 21 checkpoint sidecars validate, all 24 MP4/NPZ pairs are present, and the
648 fixed-evaluation transitions close exactly.

The stochastic training distribution nevertheless contained the intended
milestones at 256k: mean liftoff bonus was 5.36, mean stable-airborne bonus was
1.76 (about 11% of episodes), and mean `apex_approach` was 0.283. Apex success
remained zero. The same batch averaged -33.52 angular-rate cost, -7.4 illegal-
contact cost, and 14% physical failure. Increasing the existing liftoff or
stable-airborne bonuses would therefore reinforce events without requiring
progress toward the full transition band.

## Single hypothesis

Change only the stable Phase U configuration value:

```text
apex_approach_weight: 2.0 -> 8.0
```

The existing `apex_approach` term is active only after legal jump-window entry,
stable full-structure airborne, and positive vertical velocity. Its score is
the bounded mean of the deployable Apex-contract proximity terms: obstacle-
relative position, full-structure clearance, vertical-velocity band, roll,
pitch, angular rate, and forward velocity. Increasing this weight selects the
already observed stable ascending samples according to downstream physical
quality instead of adding another weaker event bonus.

## Preserved boundaries

- Early airborne remains nonterminal telemetry and earns no post-window task
  progress.
- Window entry, liftoff, or stable airborne alone earns no `apex_approach`.
- Apex success still requires the complete two-phase physical contract.
- The term uses current deployable physical signals only; it has no reference
  time/index, future outcome, success label, metadata, or trajectory tracking.
- The term remains finite, non-negative, per-tick bounded by 8.0, and the total
  reward remains clipped to [-50, 50].
- XML/payload, +/-50 N m limits, action mapping, reset, observation, thresholds,
  deadlines, PPO layout, optimizer, exploration, angular cap, bridge bonuses,
  and all safety termination contracts remain fixed.

## Validation and execution

Red-green tests bind both stable configs and the reward-contract hash to 8.0
while preserving the activation and pre-window regressions. Then run focused
tests, compileall, full pytest, local preflight, and a fresh managed runtime
gate. A clean 512-environment smoke may authorize one fresh run-bound formal
run up to 998,400 aligned training transitions. Monitoring remains sparse and
the fixed checkpoint Gate Pause protocol remains unchanged.
