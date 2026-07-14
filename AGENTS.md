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

## Verification

- Run pure unit tests and static compilation before dynamic MJX work.
- Before a long PPO run, require a model-load test, reset/step smoke test,
  snapshot round-trip test, deterministic inference test, and short PPO test.
- Report physical failures and timeouts separately.
