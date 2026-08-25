# Findings and Decisions

## Requirements

- On 2026-08-25 the user explicitly rejected mixed hip/knee semantics. Both
  joints must use the hip-style keyframe-centered piecewise absolute position
  map. With the authoritative knee keyframe at its upper limit, knee maps
  `-1 -> -1.5`, `0 -> 2.5`, and `+1 -> 2.5` radians.
- After verified modification, the user authorized one fresh approximately
  five-million-step run and prohibited loading any old checkpoint for its
  initial launch.

- On 2026-08-25 the user replaced the original Phase U reward/event contract:
  use the provided planar-jump reward as a formula reference, remove all target
  and deceleration semantics, use root-x jump zone `[2.5,3.1]`, and terminate at
  the first valid height/descent Apex.
- `jump_signal` is one only during the first visit inside the jump zone; after
  first exit it is zero permanently. It is one shared Actor/critic task scalar,
  appended outside the three-frame history.
- Remove approximate front/rear wheel-support booleans and platform-relative
  structure clearance from policy observations and Apex logic.
- Use 5% airborne training RSI centered at `x=2.8,z=2.0,vx=2.0`; add positive
  `vz` so a high reset cannot receive immediate success by falling.
- Held-out formal evaluation remains natural-reset only. No PPO training is
  authorized by this implementation task.

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

- The prior JIT layout collected 25,600 transitions per block but performed
  only eight optimizer steps. The user's 12-env, 2,048-step, 1,024-minibatch,
  eight-epoch SB3 layout collected 24,576 transitions and performed 192
  optimizer steps. Brax `384 env x 64 unroll`, `batch=16`, `24 minibatches`,
  and `8 updates` reproduces the aggregate 24,576/1,024/192 layout.
- The closest lower whole-block value to five million is 4,988,928 transitions
  (`203 * 24,576`).
- The previous v2 natural panels all failed by roll limit and the final policy
  exploration scale collapsed. The new learning rate therefore remains
  `1e-4` while entropy, clip, gamma, lambda, update count, and gradient clip
  move toward the user's prior successful PPO contract.
- The first v3 fixed-rate smoke completed all 24,576 transitions and kept
  policy means/stds bounded, although KL 341.1 included a before/after
  observation-normalizer coordinate change. Two isolated adaptive-KL smokes
  (8 passes and 1 pass) delayed normalizer warm-up and instead exploded policy
  outputs and KL to roughly 1.6-1.7 million. The rejected adaptive experiment
  was removed; fixed `1e-4`, 8 passes is the validated active profile.
- A new canonical config hash is sufficient to reject old checkpoints while
  preserving the checkpoint serialization format. The initial manifest must
  additionally prove `parent_checkpoint=null` and `starting_transition=0`.
- The user prohibition is interpreted strictly for the initial workflow: do
  not open old checkpoints even for diagnostics. Natural and forced-RSI panels
  inspect only parameters produced by the new run at declared milestones.
- The fresh v3 run completed 4,988,928 training transitions plus 192 natural
  evaluation and 88 forced-RSI diagnostic transitions. Strict provenance
  verification passed, including final checkpoint restore and artifact lineage.
- Every natural milestone was 0/8 Apex and 8/8 physical failure. The first
  milestone ended on pitch limits; every later milestone ended on illegal wheel
  contact. The final panel lasted exactly two transitions per rollout and had
  50% deterministic action saturation.
- Every forced-RSI milestone was 8/8 Apex, but the reset already supplied
  height and upward velocity. The final successful RSI transition incurred
  about 3,173 W joint power and an unclipped reward of -80.95, so the result is
  event-chain evidence rather than a learned stable-jump claim.
- The final natural terminal action saturated near `[+1,-1,+1,-1]`, commanding
  controls near `[+0.8,0,+0.5,-1.5]`. Joint energy, illegal contact, and physical
  failure summed to more than -107 before smaller positive terms; total reward
  was clipped from -103.607 to -50.
- The knee keyframe equals its +2.5 rad actuator upper limit. Hip and knee use
  the same absolute formula, but the knee positive half-axis is consequently
  degenerate while a negative action can request the full 4 rad span in one
  control step. This is a candidate mechanism requiring frozen-policy action
  intervention, not yet a proven sole cause.

- Current MJX-Warp `Data` exposes `actuator_force` but no geom-paired `contact`
  field. Exact hip/knee mechanical power is available; exact wheel contact pairs
  are not. The user rejected approximate support state, so no replacement
  support observation is needed.
- The authoritative root starts at `x=1.5,z=0.15`; the obstacle spans
  `[3.6,7.6]` with top `z=0.16`. Root-x jump zone `[2.5,3.1]` is therefore a
  0.6 m pre-platform trigger interval.
- Existing `structure_clearance` is the lowest robot collision point minus the
  platform top plane without horizontal-overlap gating. It is not general
  ground clearance and will not remain a v2 Actor/Apex feature.
- A reset at `z=2.0,vz=0` would satisfy height and begin descending under gravity
  on the first step. The v2 RSI must start with positive vertical velocity and
  Apex must require observed ascent before descent.
- Existing formal v1 verification reads the saved raw resolved config rather
  than calling `load_config`; schema-aware verification can therefore preserve
  old evidence while active configs move to v2.

- The installed Brax PPO callback exposes only observation normalizer, Actor,
  and critic parameters. It does not expose optimizer state, minibatch RNG, or
  the complete internal TrainingState.
- `num_evals=40` yields one initial callback plus 39 post-block callbacks for
  the exact `998,400 / 25,600 = 39` formal layout when Brax evaluation is
  disabled.
- Parameter-level warm resume is available through `restore_params`; it must be
  declared as optimizer-reset warm start, not exact continuation.
- The approved formal milestones are 0, 102,400, 256,000, 512,000, 742,400,
  and 998,400 transitions. Every nonzero milestone receives the same eight
  held-out deterministic seeds.

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
| Keep formal orchestration in `formal_training.py` | Preserves the verified smoke runner and gives formal evaluation/resume one focused owner. |
| Use one persistent 39-block process | Avoids resetting PPO optimizer between normal blocks while retaining warm recovery after abnormal exit. |
| Record warm resume as optimizer reset | Matches the actual installed Brax API and prevents false exact-resume claims. |
| Run eight held-out seeds at five nonzero milestones | Provides fixed evidence without using Brax's separate evaluation semantics. |

## Issues Encountered

| Issue | Resolution |
|---|---|
| GPU is hidden inside the default command sandbox | Use an approved unsandboxed runtime command only for GPU checks and GPU smoke execution. |
| Repository guide suggests modifying root `scripts/local_preflight.sh` | Create `JIT/scripts/local_preflight.sh` instead, because the user requires all generated changes under JIT. |
| A self-contained JIT tree conflicts with the prohibition on duplicating the XML | Keep code self-contained but load the one retained XML by resolved absolute path. |
| Root pytest collects both root and JIT tests with duplicate basenames | Mark JIT tests as a distinct package and insert only `JIT/src` during test collection. |
| Root preflight retains one failure in user-modified launch-diagnostic files | Do not edit outside JIT; separately verify all other 1,030 tests and record the exact exclusion. |
| Root compatibility initially deselected the older manifest test instead of the new user-modified relative-x test | Reproduced both focused tests: the older test passes and only `test_relative_x_manifest_binds_exact_onsets_and_ceiling` fails because the modified production signature lacks `mode`; leave both user paths untouched. |
| First formal launch failed after its first 25,600-transition block because Brax `EpisodeMetricsLogger` invoked the shared `progress_fn` inside the compiled epoch before `policy_params_fn` | Corrected the failed run ledger with an explicit ignored audit artifact; disable only Brax's in-epoch episode logger for formal mode while retaining the once-per-block loss/KL/SPS callback and fixed held-out terminal analysis. The transition-0-only checkpoint cannot support warm resume, so use a new fresh run id. |

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
# 2026-08-25 v4 Apex-continuation findings

- Host MuJoCo replay of the previous final natural state identified the only
  contact as the XML-authorized `floor/rearwheel_collision` pair at about
  `-0.014175 m`. The analytic JIT threshold had incorrectly converted ordinary
  compliant wheel support into an illegal-contact terminal and penalty.
- Wheel/terrain clearance and penetration remain useful telemetry, but are not
  trustworthy terminal classifiers. Prohibited chassis/body clearance remains
  the illegal-contact source.
- Apex must remain a one-shot reward/event boundary rather than a terminal so
  post-Apex behavior can be observed. The full trace is split with the Apex
  state shared by pre/post files, conserving transitions exactly.
- Brax's episode logger calls `progress_fn` before the ordered policy callback.
  v4 therefore routes `episode/*` statistics into an independent JSONL stream
  and retains checkpoint-aligned PPO loss progress in the existing stream.
- Episode means are explicitly the rolling last 100 completed episodes. A
  block with no completed episode writes no episode row; later horizon/failure
  completions supply nonempty evidence before final plotting.
- Real startup metrics exposed that the default cached auto-reset kept JIT
  `episode_step/events/timeout` info after done. The first episode was valid but
  later episodes collapsed to one step. Correct training requires
  `full_reset=True`; this is now config-bound and GPU-tested, and the invalid
  attempt was aborted without checkpoint reuse.
- Playground's full reset also overwrote the just-finished
  `episode_done/episode_metrics`, so physical reset correctness alone was not
  enough for the requested reward/length curves. JIT now preserves exactly
  those logging fields through reset and exposes them after reset; terminal
  length two and restarted length one are both asserted on GPU.
