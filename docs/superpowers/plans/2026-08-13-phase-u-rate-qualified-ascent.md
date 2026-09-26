# Phase U Rate-Qualified Window Ascent Implementation Plan

1. Add direct RED tests for zero-rate full ascent credit, below-cap partial
   credit, at/above-cap zero credit, and pre-window zero.
2. Implement only the bounded angular-rate quality multiplier in
   `phase_u_reward_components` and bump the reward semantics from v4 to v5.
3. Run focused Phase U/two-phase/deadline tests, compileall, full pytest, and
   local preflight.
4. Refresh one managed runtime gate and record its 64+32 engineering
   transitions.
5. Commit/push, run one exact 256-env 6,400-transition smoke, audit sidecars,
   accounting, metrics, and failure media.
6. If smoke integrity passes, issue one fresh run-bound 998,400-transition
   authorization and supervise only startup, checkpoint milestones, and exit.
7. On terminal state, audit physical outcomes. Candidate acquisition and
   continuation probing remain gated on real Apex parent coverage.

No step changes reward weights, safety thresholds, deadline, model, reset,
observation, exploration, PPO hyperparameters, or action mapping.

