# Phase U Apex-Approach Weight Implementation Plan

1. Close the stable-airborne run audit: status, interaction accounting,
   checkpoint sidecars, failure videos/state traces, and terminal log path.
2. Add red stable-config and reward-hash tests expecting
   `apex_approach_weight = 8.0`.
3. Change only the two stable Phase U config values from 2.0 to 8.0; keep the
   dataclass default and every runtime/PPO/physics contract unchanged.
4. Run focused and Phase U tests, compileall, full pytest, local preflight, and
   a fresh managed runtime gate.
5. Commit and push validated source/evidence without ignored run artifacts.
6. Run one fresh 512-env 12,800-transition smoke and close its checkpoint,
   accounting, video, and runtime audits.
7. If clean, create one new run-bound authorization, launch a fresh 998,400-
   transition formal run, and inspect only terminal state, a fixed checkpoint,
   or abnormal exit.
