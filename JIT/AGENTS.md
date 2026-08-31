# JIT working rules

- `JIT/` is the active development area for this work.
- Treat the root `dvgc/`, `cli/`, `scripts/`, and `tests/` directories as
  read-only by default. Do not clean or rewrite the root repository from JIT.
- The only model XML is `assets/orange_bike_4kg_horizontal.xml`.
- Use only `/home/qy/mujoco_playground/.venv/bin/python`. Never change,
  reinstall, or reconfigure that environment.
- Do not create `JIT_v2`, `DVGC_new`, nested Git repositories, or
  version-suffixed production Python files.
- Write run outputs under `JIT/runs/<capability>/<run_id>/`. Run outputs are
  ignored and are not committed.
- Preserve frozen `pi_up_star`, `pi_down_star`, first-pass `V_up`/`V_down`, and
  the existing 222-entry TRAIN-only learned Soft Tube as immutable bootstrap
  provenance. Do not retrain or silently relabel those completed artifacts.
- The completed Round-1 unified transition-10,009,600 checkpoint is the current
  candidate `pi_0` for Tube-conditioned envelope-expansion work. It is not yet
  `pi_unified_star` and does not support a final JCE/JEL claim.
- The Round-1 canonical natural-start `yaw_limit` result is retained as a
  cold-start diagnostic. Under the current research scope, ordinary locomotion
  before Tube entry is not the final JIT deployment/evaluation domain.
- The preflighted `pi_unified_round2_natural50.json` experiment was superseded
  before launch. Preserve its config and handoff evidence; do not launch it
  unless the research question is explicitly changed back to cold-start
  locomotion.
- The active next stage is policy–Tube envelope iteration as defined in
  `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`: freeze `pi_0`, acquire boundary
  candidates through authoritative real dynamics, label continuation under the
  frozen unified policy, learn policy-conditioned continuation fields, build an
  expanded TRAIN-only Tube, and only then predeclare the next policy-improvement
  run.
- Boundary expansion must not directly mutate `qpos`/`qvel` or widen coordinate
  bounds and call the result capability evidence. Reuse/generalize the existing
  real-dynamics boundary machinery and provenance-complete snapshots.
- Expert-conditioned `V_up`/`V_down` are bootstrap `Tube_0` authorities only.
  Later unified-policy continuation fields must bind the exact frozen unified
  policy checkpoint and protocol.
- Keep expansion TRAIN, expansion validation, iteration audit, and final
  JCE/JEL evaluation disjoint. Final evaluation data may not influence Tube
  construction, policy training, checkpoint selection, threshold selection, or
  convergence stopping.
- Every learned Tube remains training guidance only, never a certified safe
  Tube or formal viability set.
- Do not automatically launch PPO, delete files, clean the root repository, or
  claim unimplemented work.
