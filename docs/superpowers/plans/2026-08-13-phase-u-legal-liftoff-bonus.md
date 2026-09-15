# Phase U Legal-Liftoff Bonus Implementation Plan

1. Add red tests for the new weight/schema, hash binding, legal-window-only
   transition behavior, one-shot semantics, and no success implication.
2. Add `liftoff_bonus_weight` and `legal_liftoff_bonus` to the Phase U reward
   contract; compute the transition from previous/current monotonic event
   state in the adapter. Set only the stable Phase U configs to 8.0.
3. Run focused reward/event/config tests, all Phase U regressions, compileall,
   full pytest, local preflight, and a fresh managed runtime gate.
4. Commit and push validated source/evidence; exclude ignored run artifacts.
5. Execute one fresh run-bound 512-env smoke and audit update finiteness,
   checkpoint identity, accounting, and failure videos.
6. If clean, issue a fresh 998,400-transition formal authorization, launch a
   fresh-initialization persistent process, and inspect only fixed milestones
   or terminal state.

