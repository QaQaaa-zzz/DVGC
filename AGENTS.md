# DVGC Repository Instructions

## Research scope

- The current research direction is the concise two-phase RA-L method defined
  in `PROJECT.md` and `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`.
- The phases are `Propulsion-Ascent` and `Descent-Recovery`; the Apex
  transition band connects them but is not a third expert.
- Phase experts generate snapshots and continuation labels. They must not use
  a learned feasibility network or Tube during expert training.
- Train/freeze experts first, collect and label snapshots second, train
  `V_up`/`V_down` third, construct learned soft Tubes fourth, then train one
  unified Tube-RSI PPO.
- A learned soft Tube is training guidance, not a certified safe Tube. Only an
  independent frozen-final-policy evaluation may support a JCE/JEL claim.
- Do not claim that two-phase expert semantics, `V_up`/`V_down`, learned soft
  Tubes, two-phase unified PPO, or a new pipeline CLI is implemented until the
  code and experiments exist and pass validation.

## Immutable runtime and model

- Work only in `/home/qy/DVGC`; do not create another Git repository or a
  version-suffixed source tree.
- Use `/home/qy/mujoco_playground/.venv/bin/python` directly. Never create,
  reinstall, upgrade, or reconfigure that environment without explicit user
  authorization.
- The only authoritative model is
  `assets/orange_bike_4kg_horizontal.xml` with a 2 kg payload, hip/knee force
  limits of +/-50 N m, and action order
  `[steer, rear-wheel drive, hip, knee]`.
- The `4kg` token in that single retained path is a historical filename, not
  the current payload contract. Do not create a second XML to correct the name.
- Do not change meshes, collision geometry, obstacle dimensions, matcher
  radii, environment physics, reward meaning, snapshot semantics, or action
  mapping during repository cleanup.

## Repository workflow

- Never auto-resume an old five-stage controller, sequential shared-Actor
  route, H1/C_L A/B route, roll-targeted retention route, or final-shared v1/v2
  route. Existing code is legacy migration source only.
- Do not impose 4 -> 8 -> 16/32 as a universal hard requirement for learned
  training Tubes. Evaluation budgets belong to a separately approved protocol.
- Maintain one stable entrypoint per capability. Put variants in config, run
  metadata, and Git history; never create version-suffixed production source.
- Put one-off diagnostics in `tools/diagnostics/` and remove them when the task
  ends unless they are generalized and tested.
- Before every run, record its purpose, inputs, interaction cost, stopping
  condition, and output directory. Run outputs belong under
  `runs/<method>/<run_id>/` and must remain ignored.
- Long-running processes must be persistent and resumable. Inspect only sparse
  milestones, completion, or abnormal exit; do not poll full logs repeatedly.
- Do not change the approved method definition, artifact semantics, or claim
  boundary without explicit user authorization.

## Git discipline

- Preserve user changes. Never reset, stash/restore, rebase, force-push, or
  merge to `main` without explicit authorization.
- Use focused commits after a logical phase passes validation. Explicitly
  stage paths; never use `git add .` or `git add -A` in a mixed repository.
- Do not commit `runs/`, `artifacts/`, checkpoints, policy parameters, large
  pickle/JSON files, logs, profiling output, caches, or local service state.
- Unvalidated source/config/script/research-document changes must not be
  committed.

## Cleanup and verification

- `docs/REPOSITORY_LAYOUT.md` is the only dependency/deletion ledger. Build a
  retained closure from stable roots before deleting anything.
- Legacy route tests do not retain production modules by themselves. Extract
  shared contracts into stable API tests before deleting route-only tests.
- Inspect real user systemd state before deleting any launcher, watchdog,
  service, or timer. Never stop an active service without permission.
- Run static compilation and targeted tests after each deletion batch. Run the
  full runtime gate only when its fingerprint changes or final verification
  requires it.
- The runtime gate's 64+32 timestep PPO is solely a compile/update/resume smoke
  test. This cleanup must not start formal PPO training or describe gate output
  as a learnability result.
- Report physical failures and timeouts separately and never weaken a contract
  merely to make tests pass.

After context recovery, read only `AGENTS.md`, `PROJECT.md`, and
`docs/EXPERIMENT_STATE.md` before resuming from the last validated marker.
