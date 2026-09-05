# Phase U Angular-Rate Cap-4 Implementation Plan

1. Close the Apex-weight-8 Gate Pause audit, including accounting, checkpoint
   sidecars, MP4/NPZ evidence, reward decomposition, and deterministic control
   trace.
2. Add red stable-config tests expecting
   `angular_rate_penalty_cap_ratio = 4.0` while retaining weight 1.0 and all
   bridge/Apex weights.
3. Change only the two stable Phase U config cap values from 8.0 to 4.0.
4. Run focused tests, compileall, full pytest, local preflight, and a fresh
   managed runtime gate.
5. Commit and push validated source/evidence without ignored run artifacts.
6. Run one fresh 512-env 12,800-transition smoke and audit checkpoint,
   interaction accounting, runtime faults, and failure videos.
7. If clean, create a new run-bound authorization, launch a fresh 998,400-
   transition formal run, and inspect only fixed milestones or terminal state.
