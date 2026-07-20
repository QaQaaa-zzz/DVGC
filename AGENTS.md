# DVGC Repository Instructions

## Research Scope

- Target the concise RA-L core: event-aligned next-stage reachability labels,
  phase-conditioned reachability estimation, independently certified
  phase-wise tubes, Tube-RSI, and final unified-policy Final-Recovery audit.
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
- The exhaustive H1/C_L A/B and roll-targeted shared-Actor retention route is
  superseded and must not auto-resume.  Ascent, Apex and Descent remain
  substages of the independently owned Flight expert; canonical expert
  handoffs remain active.  Intermediate labels measure valid next-stage entry,
  while only the frozen final unified policy is judged as the formal JEL by
  end-to-end Final-Recovery.
- Sequential shared-Actor Flight retention repair is closed.  Use independently
  owned bootstrap experts with irreversible canonical-entry handoffs, mark all
  composite-policy recoverability as `expert_conditioned_provisional_envelope`,
  then consolidate with phase-balanced distillation plus joint RSI PPO.  Only
  fresh independent recertification of the frozen final shared Actor may use
  the `final_shared_policy_jel` role or support formal JEL claims.
- Bootstrap objectives are strictly local: Takeoff->Ascent, Ascent->Apex,
  Apex->Descent, Descent->Landing/C_L, and Landing->Stable.  C_L and Full Chain
  must not gate Takeoff, Ascent, or Apex.  Local proposal support is not a
  Tube; only immutable expert-stack Final evidence may be called an
  expert-conditioned provisional envelope.
- Before any expensive acquisition write `cost_estimate.json`, run a 2--5%
  pilot, and use the 4 -> 8 -> 16/32 adaptive branch funnel.  A failed current
  controller bank means negative-under-current-controller-bank or unknown,
  never physical unreachability.

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
