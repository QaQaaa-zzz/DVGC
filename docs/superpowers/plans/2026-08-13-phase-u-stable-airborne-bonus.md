# Phase U Stable-Airborne Bonus Implementation Plan

1. Add red schema/hash tests and adapter tests for pre-window, liftoff-only,
   first stable-airborne, repeated latch, and no-success behavior.
2. Add `stable_airborne_bonus_weight` and `stable_airborne_bonus`; calculate
   its one-shot transition from previous/current monotonic event state. Set
   only stable Phase U configs to 16.0.
3. Run focused and Phase U tests, compileall, full pytest, local preflight, and
   a fresh managed runtime gate.
4. Commit/push source and evidence without ignored run artifacts.
5. Run a fresh 512-env 12,800-transition smoke and close checkpoint/accounting/
   video/runtime audits.
6. If clean, issue and launch a fresh 998,400-transition formal run and inspect
   only fixed milestones or terminal state.

