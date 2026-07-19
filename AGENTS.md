# DVGC Repository Instructions

## Research Scope

- Target the concise RA-L core: event-anchored backward bootstrap, end-to-end
  Final-Recovery empirical tubes, and tube-guided reset-state initialization.
- Do not claim that a learned GRU estimator, Physical-Belief variants, or
  trigger-budgeted relabeling is implemented unless the code and experiments
  are added and validated.
- The only authoritative robot model is
  `assets/orange_bike_4kg_horizontal.xml` with a 4 kg payload and hip/knee
  force limits of +/-50 N m.

## Environment

- The configured training runtime is
  `/home/qy/mujoco_playground/.venv/bin/python` on the user's Ubuntu machine.
- Do not create, reinstall, upgrade, or otherwise reconfigure that environment
  unless the user explicitly requests it.
- Prefer invoking the environment's Python executable directly instead of
  depending on shell activation.

## Workflow

- Work only in the real repository root. Do not create temporary Git repos or
  version-suffixed source trees.
- After each run, inspect metrics and terminal causes before changing training
  logic.
- Keep each validated change in a focused Git commit.
- Never mix policies, banks, or results across XML hashes, action mappings, or
  policy versions.
- Continue autonomously through Landing -> Flight -> Takeoff -> Approach ->
  natural-start evaluation, then the minimum RA-L seed/baseline/ablation
  matrix.  Pause only for environment reconfiguration, destructive operations,
  a claim-changing research fork, or an external hardware/permission/data
  blocker.  Geometry mapping and candidate-validation repairs that stay within
  an existing gate and do not alter the paper method may proceed autonomously
  for up to three evidence-based rounds; pause only after the third fails.
- Use one main agent.  Do not browse, draft the paper, make figures, or launch
  the multi-seed matrix before the seed-0 end-to-end chain passes.
- Maintain `docs/EXPERIMENT_STATE.md`; after context recovery read only this
  file plus `AGENTS.md` and `PROJECT.md` before resuming the last valid marker.
- Runs must be resumable and provenance-keyed.  Skip a completed step only when
  its input hashes and output hashes still match; never overwrite a run or feed
  independent-audit labels back into training.
- Keep `certified_tube` and `proposal_support_bank` as distinct artifact roles.
  Only an independently audited frozen-policy Tube may call states safe or
  contribute Tube precision/coverage.  Legal boundary, provisional and mined
  states may drive proposal search, teacher data and RSI but never promote
  themselves to safe.  Intermediate expert discovery and full-jump data
  collection must not be blocked merely because an intermediate Tube has fewer
  than four certified-safe states.
- After the valid Cycle-5 gate, do not run more roll-targeted descent PPO.
  Continue handoff decomposition, Landing-only certifier calibration,
  proposal-support construction, full-jump teacher trajectory mining and final
  shared-policy Tube-RSI.  Keep the Landing policy and C_L matcher radius fixed;
  a C_L extension requires separately certified contact proposals and a new
  immutable version/hash.

## Verification

- Run pure unit tests and static compilation before dynamic MJX work.
- Before a long PPO run, require a model-load test, reset/step smoke test,
  snapshot round-trip test, deterministic inference test, and short PPO test.
- Report physical failures and timeouts separately.
- During PPO, use a persistent process and inspect only sparse milestones,
  completion, or abnormal exit.  Prefer compact JSON extraction and log tails;
  avoid repeated full-log reads, broad reviews, and unrelated refactors.
- Run targeted tests for ordinary controller/report changes.  Re-run the full
  runtime gate only when runtime source, configuration, XML, or its fingerprint
  changes.  Commit only validated source/config/script/research-document edits;
  keep checkpoints and run artifacts ignored.
