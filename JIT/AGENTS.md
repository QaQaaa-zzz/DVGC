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
- The current `pi_up_candidate` is `transition_9977856` from
  `JIT/runs/phase_u/phase_u_v4_pitch15penalty_9977856_seed820901_20260826`.
  Phase U training stops here. The eight natural-reset rollouts all reached
  Apex but later ended at `pitch_limit`; their similar reset states do not
  represent eight independent initial conditions.
- The next step is to freeze candidates and design the handoff snapshot bank.
  Phase D, `pi_down`, continuation labels, `V_up`/`V_down`, learned soft
  Tubes, unified PPO, and final JCE/JEL remain unimplemented.
- Do not automatically launch PPO, delete files, clean the root repository,
  or claim unimplemented work.
