# Phase U Angular-Rate Penalty Cap Implementation Plan

1. Add red tests for `angular_rate_penalty_cap_ratio`: explicit cap scaling,
   invalid values, stable smoke/formal config value 8.0, and reward hash drift.
2. Add the field to `PhaseURewardConfig`, validation, serialization, and the
   bounded angular-rate component. Change only the two stable Phase U config
   values required by the design.
3. Run the focused reward/config tests, Phase U adapter regressions, full
   pytest, `scripts/local_preflight.sh`, and a fresh managed runtime gate.
4. Commit and push the validated source and evidence without including ignored
   run artifacts.
5. Create a fresh run-specific 512-environment smoke configuration and
   run-bound authorization, execute one 12,800-transition block, and audit its
   checkpoint, accounting, runtime faults, and failure videos.
6. If smoke is clean, create a fresh 998,400-transition formal configuration
   and authorization, launch a persistent fresh-initialization process, record
   its control/resume contract, and supervise only fixed milestones or terminal
   state.

