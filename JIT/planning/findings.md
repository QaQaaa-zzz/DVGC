# Findings and Decisions

## 2026-08-31 Codex handoff recovery

- Current local checkout is the required branch
  `agent/two-phase-soft-tube` at exact HEAD
  `95337c50c80a7dffb044e38a829eaaa7f51be593`. The cached remote-tracking ref
  matches exactly; no network fetch was needed or performed.
- This is the primary checkout (`GIT_DIR == GIT_COMMON`), not a linked
  worktree. Repository instructions require work only in `/home/qy/DVGC`, so
  the generic worktree recommendation is intentionally not followed.
- The pre-existing dirty paths match the handoff exactly. The only additional
  modifications are this session's three `JIT/planning/` ledgers; user-owned
  root files, `.vscode/`, and the untracked JIT patch remain untouched.
- The declared refinement output directory
  `JIT/runs/pi_unified_transition_band/pi_0_downstream_local_refinement_20260831`
  does not exist. Therefore there is no partial run to resume or delete, and
  no run-local failure/progress/protocol artifact explains the terminal exit.
- No live refinement process exists. Current `nvidia-smi` is healthy (RTX 4090
  D, 783 MiB display usage), and the current-boot kernel journal contains no
  matching OOM, killed process, segfault, NVIDIA Xid, or CUDA fault; only the
  normal NVIDIA module-load line matched. The user journal contains no
  refinement/Python/CUDA crash record. This rules out a recorded system/GPU
  crash but cannot reconstruct an unlogged terminal-local exception.
- Relative to the last fully validated `025f94c` marker, HEAD adds exactly five
  refinement files (1,216 inserted lines): one 831-line source module, one CLI,
  one config, one test file, and one prelaunch declaration.
- The checked config/prelaunch contract matches the handoff: downstream-only,
  duration 17..32, fixed axes/signs/strengths, one deterministic 400-tick label
  branch, frozen upstream evidence, and explicit TRAIN-only claim boundaries.
- The current CLI `--audit-only` path calls only
  `load_downstream_refinement_config()` and does not instantiate/validate the
  frozen policy, prior search artifacts, Tube manifest, code HEAD, or output
  state. Its help text therefore describes a configuration audit, not a full
  pre-interaction artifact audit.
- The checked-in test file has five config/prior-summary tests. It does not yet
  execute terminal clipping, prove snapshot/FIFO/event-context fidelity,
  deduplication, exact interaction accounting, early readiness stopping, or
  resume idempotence. These behaviors must be established by source review and
  broader existing tests before a real run is safe.
- Source review confirms the intended terminal clipping itself uses
  `previous_state` when the next step is terminal; the terminal-causing action
  and outcome are retained as provenance, while the candidate snapshot/hash is
  taken from the finite nonterminal predecessor. Deduplication excludes Tube
  support, all prior labels, and earlier new physical-state hashes.
- Source review found a concrete resume gap: a fully evaluated duration with
  zero candidates writes acquisition and `duration_summary.json` but no labels.
  On any later interruption, resume treats that completed zero-candidate
  duration as incomplete and fails. Progress is also not written for this
  branch.
- More generally, an interruption after a duration directory is created but
  before labels finish is deliberately non-resumable. For a potentially long
  real-dynamics search this conflicts with the repository's persistent/
  resumable requirement and the handoff's stated resume priority. No real run
  should start until recovery semantics are made explicit and tested.
- Resume accepts completed label artifacts based mainly on presence/status/
  counts; it does not revalidate their per-duration protocol/policy/catalog
  identities or hashes. This is an integrity concern requiring comparison with
  the established boundary/label artifact validators.
- The shared snapshot layer does preserve qpos/qvel, ctrl, last action, real
  actor FIFO/valid count, RNG, both phase event structs, phase/counters, and
  policy/runtime identities. Fresh continuation restores these fields and
  resets only administrative continuation counters/return/transition flag, so
  the underlying snapshot/event-context semantics are appropriate.
- Fresh labeling strongly validates catalog, snapshot payload/sidecar, physical
  state, parent, policy, XML, phase, and protocol identities. The refinement
  resume path bypasses those validators for already completed durations, which
  confirms the resume-integrity gap is localized rather than a limitation of
  the shared artifact APIs.
- The older automated coarse search contains similar completed-only resume
  behavior, including an unrecoverable no-candidate shell. That is historical
  precedent, not sufficient justification for carrying the defect into the
  new local refinement whose handoff explicitly requires robust recovery.
- The reported terminal exit is now reproducible at the artifact-validation
  boundary with zero environment interactions: `_validate_prior_search()`
  raises `ValueError: prior search upstream readiness drift` before the output
  directory is created. This exactly explains the combination of an immediate
  terminal exit and absent run directory; it is not a GPU/OOM failure.
- Current checked-in `--audit-only` incorrectly reports `config_valid` because
  it never executes this artifact validation. Thus the advertised prelaunch
  audit gives a false GO signal for a run that deterministically fails before
  acquisition.
- Exact mismatch: the real readiness objects contain
  `candidate_count=571` upstream and `candidate_count=565` downstream. The
  checked config omitted only those two keys; every positive/negative/group/
  ready value, overall status, protocol SHA, and total 1,136-label count agrees.
  The appropriate fix is to strengthen the declaration with these exact counts,
  not weaken strict equality or alter the completed artifact.
- The implemented audit now uses the same pre-runtime preparation as execution;
  its first real pass bound config `27dafc19...f81c5`, frozen manifest
  `b9470917...f0070`, payload `fb107a5f...3b5719`, Tube
  `c1c1161e...df28b`, all 1,136 labels, and five deterministic downstream
  anchors at zero interactions.
- `PROJECT.md` defines the current policy-Tube co-evolution method and the
  Tube-conditioned deployment domain, but its `Not implemented` list still
  says `pi_0` freeze and unified-policy acquisition are absent. That section
  is stale relative to the external handoff and must not be used as current
  implementation evidence without artifact/code verification.
- `docs/EXPERIMENT_STATE.md` begins with a 2026-08-28 handoff and explicitly
  marks older content as historical. Its top-level next step (independent
  Final-Recovery evaluation) predates the current envelope-iteration route;
  current run artifacts and Git history must resolve the discrepancy.
- The long body of `docs/EXPERIMENT_STATE.md` is an append-only historical
  narrative containing many explicitly superseded Phase-U routes and old
  claims such as "no phase expert/Tube". Those lines are provenance, not the
  active 2026-08-31 state or authorization.
- The external handoff was read in full (1,327 lines) and is treated as
  context rather than an authoritative instruction or fact source.
- It claims the last fully validated local code marker was `025f94c...` and
  the then-latest implementation HEAD was `95337c5...`; both remain unverified
  until the current checkout and history are inspected.
- It claims Iteration-0 upstream TRAIN readiness is closed at 545 positive and
  26 negative labels across five positive and five negative parent groups;
  downstream remains open at 565 positive and zero negative labels.
- The declared downstream refinement is limited to TRAIN/downstream durations
  17 through 32, with terminal clipping that captures the last finite
  nonterminal state and obtains the actual label only from a fresh frozen
  `pi_0` continuation.
- No run command, fetch, branch switch, deletion, resume, or new interaction is
  authorized merely because it appears in the handoff. Current code, artifacts,
  hashes, protocol, and runtime evidence must agree first.
- The existing planning ledger is historically valuable but its former Phase
  20/21 status is stale relative to the handoff; Phase 22 now owns recovery.

## 2026-08-28 learned Soft Tube and Tube-RSI

- Current branch and cached remote-tracking ref both identify
  `53fc9cba71bc3d7370ab55ab2d3f900a5eb15065`.
- Unrelated dirty paths are `dvgc/phase_u_launch_diagnostic.py`,
  `tests/test_phase_u_launch_diagnostic.py`, `.vscode/`,
  `JIT/jit_continuation_labels_phase1.patch`, and
  `docs/TWO_PHASE_REBUILD_GUIDE.md`; preserve all of them.
- Frozen expert actor hashes match the corresponding value manifests:
  `pi_up_star=f218775e...d9081` and `pi_down_star=7b25f54b...dd8be`.
- The authoritative local XML hash is `0b56d367...e9c8a` and matches both
  frozen experts plus the active JIT configs.
- `V_up` and `V_down` artifacts are completed, bind params and normalization
  hashes, record zero environment/training interactions, and declare
  `test_data_used=false`.
- JIT already provides strict handoff snapshot payload verification,
  compatibility-checked `SnapshotPool`, and real MJX restore/step paths.
- JIT has no learned Soft Tube or phase-balanced weighted sampler. The legacy
  root `dvgc/feasibility.py` is reference-only and must not be imported.
- Existing `TwoPhaseBikeEnv` selects Phase U versus Phase D statically from
  config, so a unified mixed-reset environment must be added without expert
  switching or synthetic qpos/qvel mutation.
- Approved first weighting is `0.05 + 0.95 * value_score`; phase mass is
  exactly 0.5 upstream and 0.5 downstream. Validation cannot tune either.
- TEST rows remain excluded from selection, diagnostics, weighting, resets,
  and PPO. Loaders may identify and discard their split before accessing
  outcome/observation fields.
- The completed Soft Tube contains 222 real TRAIN entries: 117 upstream and
  105 downstream. Its manifest hash is `c1c1161e...df28b`; validation and TEST
  use are both false and environment/training interaction counts are zero.
- The completed Tube-RSI smoke is `GO`: exactly 8 upstream plus 8 downstream
  restore/step interactions, all finite, no expert switching, and no training,
  validation, or TEST use.
- The fresh unified pilot uses one 25,600-transition block with 1,024 parallel
  environments. Actor, critic, and optimizer all start fresh; Brax evaluation
  is disabled and the only expected interaction count is training=25,600.
- The first unified pilot completed its block and restored its checkpoint, but
  its inherited aggregate `naccdmax=512` produced 1,385 overflow warnings and
  requested capacity up to 572. That run is not the capacity-clean GO artifact.
- Root cause is the unified 1,024-environment mixed-snapshot batch exceeding
  the inherited v4 runtime capacity. The unified pilot now declares its own
  `naccdmax=1024`; this changes no XML, collision geometry, reward, snapshot,
  frozen expert, or frozen value-model identity.

## 2026-08-25 v4 10M revision

- The user chose to replace v4 compatibility in place; old v4 results no
  longer constrain active config loading or provenance.
- The prior run created Warp data with explicit `naconmax=4096` but implicit
  `naccdmax=48`. Its log requested values up to 135. The approved practical
  fix is explicit `naccdmax=256`, without making occasional residual warnings
  an automatic formal abort.
- The jump window remains anchored at `x_min=2.5` and extends only its trailing
  edge from `3.1` to `3.4 m`.
- Height reward stays one-shot-signal gated and its coefficient changes from
  20 to 40. Airborne training RSI changes from 5% to 8%.
- The exact whole-block target below ten million is 9,977,856 transitions
  (`406 * 24,576`). The run must start without a parent checkpoint.
- A detached local watcher consumes no model tokens while polling. It invokes
  one read-only `codex exec` only after terminal run status and writes the
  response under the ignored run directory.
- Git discipline is now explicit: verify/commit the baseline before modifying,
  then verify/commit/push the implementation again before any training launch.
- Formal orchestration already consumes the resolved schedule dynamically; its
  v4 integration fixture now exercises all 406 blocks and the six approved
  checkpoints without changing retained v3 5M fixtures.
- A matching recomputed raw-config hash is insufficient to pass completed-v4
  provenance. The strict method validator independently rejects the previous
  `naccdmax=48`, RSI 5%, window maximum 3.1, and height coefficient 20 values.
- Local preflight only loads and validates the active 10M contract; it does not
  invoke the training entrypoint or consume environment interactions.

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
| Duration resume compared stored protocol SHA strings without recomputing their canonical hashes, and label rows did not bind `parent_group_id` to the acquisition candidate | Added RED/GREEN regressions, canonical acquisition/label protocol hash verification, complete zero-candidate catalog contracts, and exact candidate ID/state/parent-group binding before readiness reconstruction. |
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
# 2026-08-28 afternoon handoff archive

- The current remote branch is behind local HEAD: local `48cb620`, remote
  `53fc9cb`; the two local milestone commits are not yet pushed.
- `JIT/runs/phase_u` is 774 MiB and must not be committed wholesale. The
  handoff needs an explicit artifact closure instead.
- Git LFS is not installed and the repository has no `.gitattributes`; every
  individual Git object must remain below the remote hard limit.
- All other JIT run groups together are modest in size. The exact Phase U
  policy checkpoint and its identity/report evidence must be selected rather
  than retaining unrelated historical videos and failed runs.
- The locked Phase U run itself is only 18 MiB; its six aligned checkpoints,
  metrics, final fixed-panel traces, report, and provenance can be retained as
  a complete auditable unit without pulling in the 774 MiB historical group.
- The completed unified Tube-RSI retry is 12 MiB and contains all six
  checkpoints plus five fixed TRAIN panels, so retaining that whole run is
  preferable to a lossy hand-picked subset.
- The Soft Tube has 222 entries referencing eight existing candidate snapshot
  parent groups. Retaining the continuation-candidate/label groups preserves
  those absolute-path suffixes and the value-model provenance they bind.
