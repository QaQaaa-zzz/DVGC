# Findings and Decisions

## Requirements

- Work only in `/home/qy/DVGC` and create a new top-level `JIT/` directory.
- Keep every generated file and output under `JIT/`.
- Build an independent implementation; existing `dvgc` code may be read for
  understanding but must not be imported or copied wholesale.
- Load `assets/orange_bike_4kg_horizontal.xml` directly and do not duplicate or
  modify it. Its historical filename contains `4kg`, but the authoritative
  payload is 2 kg.
- Use `/home/qy/mujoco_playground/.venv/bin/python` without changing it.
- Use MuJoCo, MJX Warp, MuJoCo Playground, JAX/GPU, and Brax PPO.
- Implement the first delivery only: environment, timing, Phase U reward and
  termination, truthful video capture, 1,024-env smoke, and one-block PPO smoke.
- After complete verification, explicitly stage and commit only `JIT/`.

## Research Findings

- The working tree already contains user changes in
  `dvgc/phase_u_launch_diagnostic.py`,
  `tests/test_phase_u_launch_diagnostic.py`, `.vscode/`, and the untracked
  rebuild guide. They are outside task scope.
- The authoritative XML SHA-256 is
  `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192`.
- `data/reference_jump.csv` has SHA-256
  `612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f`,
  821 data rows, 39 columns, and 0.002 s samples from 0.002 to 1.642 s.
- The CSV is a kinematic guideline, not a dynamically authoritative expert.
- Installed versions observed: JAX 0.6.2, MuJoCo 3.6.0, Brax 0.14.2.
- Outside device isolation, JAX sees the RTX 4090 D as `CudaDevice(id=0)`.
- MuJoCo Playground environments subclass `mjx_env.MjxEnv`, convert with
  `mjx.put_model(..., impl="warp")`, and call
  `mjx_env.step(model, data, ctrl, n_substeps)`.
- The `MjxEnv` base derives `n_substeps` as `round(ctrl_dt / sim_dt)`; the
  required values therefore produce exactly four substeps.
- The authoritative keyframe is `initial_state`; named joints are
  `floating_base_joint`, `rearwheel_joint`, `steering_joint`,
  `frontwheel_joint`, `hip_joint`, and `knee_joint`.
- Actuators are ordered `cmd_steering_v`, `cmd_rearwheel_f`, `cmd_hip_f`, and
  `cmd_knee_f`. Their control ranges are `[-0.8,0.8]`, `[-5,40]`,
  `[-1.3,0.5]`, and `[-1.5,2.5]` respectively; hip and knee force ranges are
  both `[-50,50]`.
- The guide's Actor fields sum to 27 values per frame and 81 values for the
  three-frame flattened history. The initial design's 23-value count was
  corrected during self-review.
- Host MuJoCo reports `nq=12`, `nv=11`, `nu=4`, one keyframe, 18 sensors, and
  26 geoms. Joint qpos addresses are rear wheel 7, steering 8, front wheel 9,
  hip 10, and knee 11; dof addresses are 6 through 10 respectively.
- The collision-relevant robot primitives are boxes, cylinders, and
  ellipsoids, so complete-structure support bounds can be computed analytically
  under JAX without Host MuJoCo queries during training.
- Active MJX contacts expose fixed-shape geom ids and `efc_address`; unused
  slots in the non-Warp implementation can be masked by negative addresses,
  but the installed MJX Warp `Data` deliberately exposes no `contact` field.
  The training path must therefore use geometry/IMU support and penetration
  estimates from deployable state, never contact-array access.
- The hip mapping is piecewise around the XML `initial_state` value `-1.2`:
  zero action holds the keyframe target, positive action spans to `0.5`, and
  negative action spans to `-1.3`.
- The obstacle box spans x `[3.6, 7.6]` with top z `0.16`; these values are
  derived from XML geom position/size, not duplicated as authoritative physics.
- Installed Brax PPO exposes asymmetric observation keys through
  `make_ppo_networks(policy_obs_key="state", value_obs_key="privileged_state")`
  and supports `gae_lambda`, deterministic evaluation, checkpoint paths, and
  checkpoint restore.
- The fixed first-delivery PPO layout is 1,024 environments, unroll 25,
  batch size 128, eight minibatches, and one update per batch: exactly 25,600
  training transitions per aligned block.
- Current approved two-phase semantics use Propulsion-Ascent and
  Descent-Recovery only; Apex is an interface band, not a third phase.

## Technical Decisions

| Decision | Rationale |
|---|---|
| Actor observation uses a real three-frame FIFO plus valid mask | Prevents fake copied history and preserves runtime measurability. |
| Critic receives a separate privileged observation | Allows asymmetric PPO without leaking success/reward/reference metadata to the Actor. |
| Reference analysis is an offline module disconnected from `env.step` and PPO | Enforces the weak-prior claim boundary. |
| Reward components are a fixed-key JAX pytree and individually recorded | Makes reward gating and failure analysis auditable. |
| Evaluation stops immediately on terminated or truncated | Prevents post-done stepping and repeated terminal frames. |
| Runs must have a frozen pre-run manifest | Records purpose, inputs, cost ceiling, stopping condition, and output path before interaction. |
| `JIT/.gitignore` will ignore `runs/`, caches, and generated artifacts | Keeps runtime output inside JIT without committing it. |
| Preserve all incoming environment `info` fields | Brax wrappers extend the pytree and require identical scan carry structure. |
| Expose `time_out` as a float mirror of `truncated` | Enables Brax timeout bootstrapping without conflating truncation with physical termination. |

## Issues Encountered

| Issue | Resolution |
|---|---|
| GPU is hidden inside the default command sandbox | Use an approved unsandboxed runtime command only for GPU checks and GPU smoke execution. |
| Repository guide suggests modifying root `scripts/local_preflight.sh` | Create `JIT/scripts/local_preflight.sh` instead, because the user requires all generated changes under JIT. |
| A self-contained JIT tree conflicts with the prohibition on duplicating the XML | Keep code self-contained but load the one retained XML by resolved absolute path. |
| Root pytest collects both root and JIT tests with duplicate basenames | Mark JIT tests as a distinct package and insert only `JIT/src` during test collection. |
| Root preflight retains one failure in user-modified launch-diagnostic files | Do not edit outside JIT; separately verify all other 1,030 tests and record the exact exclusion. |

## Resources

- `docs/TWO_PHASE_REBUILD_GUIDE.md`
- `PROJECT.md`
- `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`
- `docs/EXPERIMENT_STATE.md`
- `assets/orange_bike_4kg_horizontal.xml`
- `data/reference_jump.csv`
- `dvgc/action_mapping.py`
- `dvgc/two_phase_semantics.py`
- `dvgc/two_phase_runtime.py`
- `dvgc/snapshot_timing.py`
- `/home/qy/mujoco_playground/mujoco_playground/_src/mjx_env.py`

## Visual or Browser Findings

- No browser or image inspection was used. The user declined the optional visual companion.
