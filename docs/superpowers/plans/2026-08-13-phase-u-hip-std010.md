# Phase U Hip Exploration Standard Deviation 0.10 Implementation Plan

1. Close the cap-4 Gate Pause audit and record its reward/policy/trace evidence.
2. Add a red stable-config test for ordered action standard deviation
   `[0.05, 0.05, 0.10, 0.05]`.
3. Change only the hip element in the two stable Phase U configs.
4. Run focused tests, compileall, full pytest, local preflight, and a fresh
   managed runtime gate.
5. Commit and push validated source/evidence without ignored run artifacts.
6. Run one fresh 512-env 12,800-transition smoke and audit checkpoint,
   accounting, failures, and videos.
7. If clean, create a fresh run-bound authorization, launch up to 998,400
   transitions, and inspect only fixed milestones or terminal state.
