# Phase U Window-Ascent Weight 8 Implementation Plan

**Goal:** Test whether increasing only the correctly window-gated bounded
Phase U ascent credit from 4 to 8 makes physically useful launch attempts rank
above the learned grounded policy.

1. Add RED tests requiring both stable configs to resolve
   `ascent_progress_weight == 8.0` and direct reward behavior of +4 at
   `com_vz=0.5` inside the window, zero before it.
2. Change only `configs/phase_expert_smoke.json` and
   `configs/phase_expert_phase_u.json` from 4.0 to 8.0.
3. Run focused reward/config regressions, Phase U/two-phase/deadline tests,
   compileall, full pytest, and `scripts/local_preflight.sh`.
4. Refresh the managed runtime gate once because the reward fingerprint
   changes; record the 64+32 engineering transitions.
5. Commit and push validated source, then issue one exact run-bound
   256-environment smoke authorization and run 6,400 transitions.
6. If smoke integrity passes, issue a separate fresh 998,400-transition formal
   authorization and supervise only sparse milestones/terminal state.
7. At completion audit physical metrics, checkpoint identities, closed
   accounting, and failure videos. Start candidate snapshots only after real
   Apex successes and independent parent coverage.

No step changes XML, force limits, action mapping, reset, observation,
threshold/deadline, safety failures, exploration, PPO hyperparameters, network,
optimizer, horizon, or another reward coefficient.

