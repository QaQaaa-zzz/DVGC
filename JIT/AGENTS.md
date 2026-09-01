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
  the existing 222-entry TRAIN-only learned Tube_0 as immutable bootstrap
  provenance. Do not retrain or silently relabel those completed artifacts.
- Tube_1 is completed with 3,119 TRAIN entries: the exact 222-entry Tube_0 core
  plus 2,897 policy-conditioned expansion states. It is training guidance only,
  not a certified safe Tube or viability kernel.
- The fresh iteration-1 unified run
  `pi_1_tube1_natural10_10009600_seed821101_20260901_retry01` completed exactly
  10,009,600 training transitions with all declared TRAIN panels, no TEST or
  validation data, no expert switching, and final-checkpoint restore verified.
  This completion does not itself establish capability-envelope expansion.
- Preserve the first pi_1 attempt as an `engineering_error` artifact. It reached
  1,024,000 training transitions and its completed panel report records 449
  diagnostic interactions even though terminal status accounting recorded zero
  diagnostic interactions. Do not edit that historical run to make accounting
  look cleaner.
- The active next stage is: freeze the exact completed pi_1 checkpoint, then run
  core-preservation and boundary-gain gates. Only both gates together can
  authorize an empirical envelope-expansion claim or the next Tube/policy
  iteration. Final TEST/JCE/JEL remains untouched.
- Boundary expansion must not directly mutate `qpos`/`qvel` or widen coordinate
  bounds and call the result capability evidence. Reuse/generalize the existing
  real-dynamics boundary machinery and provenance-complete snapshots.
- Expert-conditioned `V_up`/`V_down` are bootstrap Tube_0 authorities only.
  Later unified-policy continuation fields must bind the exact frozen unified
  policy checkpoint and protocol.
- Keep expansion TRAIN, expansion validation, iteration audit, and final
  JCE/JEL evaluation disjoint. Final evaluation data may not influence Tube
  construction, policy training, checkpoint selection, threshold selection, or
  convergence stopping.
- Every learned Tube remains training guidance only, never a certified safe
  Tube or formal viability set.
- Importable implementation belongs under `JIT/src/jit_dvgc`; executable entry
  points belong under `JIT/cli`; pytest coverage belongs under `JIT/tests`.
- New production code must use the categorized namespaces documented in
  `JIT/docs/CODE_ORGANIZATION.md` (`training`, `tube`, `snapshots`, `analysis`,
  `continuation`, `acquisition`) instead of adding another experiment-stage
  module at `JIT/src/jit_dvgc/` root. Legacy flat modules remain for import and
  frozen-artifact compatibility until an explicit compatibility migration.
- Keep CLI files thin: argument parsing and dispatch only. Put reusable
  scientific/runtime logic in `src` packages and tests in `tests`.
- Do not automatically launch PPO, delete files, clean the root repository, or
  claim unimplemented work.
