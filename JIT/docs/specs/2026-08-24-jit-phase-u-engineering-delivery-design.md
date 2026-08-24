# JIT Independent Phase U Engineering Delivery Design

## Status and scope

This design covers one independent engineering delivery under `/home/qy/DVGC/JIT`:

- a minimal MuJoCo Playground `MjxEnv` for Propulsion-Ascent;
- 200 Hz simulation and 50 Hz control with four MJX substeps;
- one observable Actor contract with a real three-frame history;
- complete-structure geometry signals, jump-window latching, Apex success,
  physical failure, and timeout semantics;
- the bounded component-wise Phase U reward in the rebuild guide;
- deterministic fixed evaluation and truthful offline video capture;
- a 1,024-environment GPU smoke and one 25,600-transition Brax PPO smoke;
- identity-bound checkpoint save/restore and closed interaction accounting.

It does not include a formal 998,400-transition training run, a trained or
frozen expert claim, Phase D, continuation labels, feasibility models, learned
soft Tubes, unified PPO, or JCE/JEL evaluation.

## Isolation boundary

All generated source, tests, configuration, scripts, documents, planning files,
and run outputs live below `JIT/`. The Python package is named `jit_dvgc` to
avoid collision with the repository's existing `dvgc` package. It must not
import any `dvgc` module.

The implementation reads two retained external inputs:

1. `assets/orange_bike_4kg_horizontal.xml`, the single authoritative model.
   It is not copied or edited. The implementation verifies its expected
   SHA-256 before every runtime or PPO run.
2. `data/reference_jump.csv`, used only by an offline reference-analysis
   command to record broad kinematic ranges and provenance. Training code,
   Actor observations, resets, rewards, actions, and success predicates do not
   depend on its time index, actions, or pointwise states.

The fixed interpreter is `/home/qy/mujoco_playground/.venv/bin/python`. The
project is run with `PYTHONPATH=/home/qy/DVGC/JIT/src`; no package installation
or environment mutation is required.

## Architecture

`constants.py` defines timing, action order, Actor frame order, reward keys,
end codes, and immutable schema names. `config.py` validates JSON configuration
and produces a canonical hash. `model.py` resolves the retained XML, verifies
identity, loads Host MuJoCo, sets `model.opt.timestep = 0.005` before conversion,
audits required names/ranges, and creates an MJX Warp model.

`action_mapping.py` maps normalized actions in the declared order
`[steer, rear-wheel drive, hip, knee]` to actuator controls. The knee target is
incremental: positive action decreases the XML knee angle, negative action
increases it, and zero holds the current position. Every target is clipped to
the XML control or joint range; normalized actions are never treated as torque.

`observation.py` constructs one 27-value observable frame, maintains a genuine
three-frame FIFO, and exposes an 81-value flattened Actor observation with history-valid
masking. It separately constructs the privileged critic observation. Reward,
success, done fields, end codes, reference indices, policy identity, and future
outcomes are prohibited from the Actor field list.

The privileged observation has 114 values: the 81-value Actor history, 12
qpos values, 11 qvel values, exact roll/pitch/yaw, exact root linear velocity,
full-structure clearance, and three current support/contact flags.

`geometry.py` builds a frozen collision-geometry manifest from the XML and uses
JAX-compatible support bounds to compute the robot's frontmost point,
`obstacle_relative_x`, full-structure clearance, wheel support, and prohibited
penetration estimates. The installed MJX Warp `Data` has no contact array, so
the training path deliberately uses deployable geometry/IMU estimates and
performs no Host MuJoCo collision query.

`semantics.py` advances fixed-shape JAX event state. Jump-window entry is
monotonic. Stable airborne and ascending events are observable. Propulsion-
Ascent succeeds only on first full Apex-band membership. Nonfinite state,
attitude limits, illegal structure/wheel contact, backward exit, and platform
overrun are physical failures. Horizon exhaustion without success or failure
is `truncated=true`, `terminated=false`.

`rewards.py` computes every guide-defined component as a fixed-field JAX struct.
All jump-related positive terms are exactly zero until legal-window entry;
early airborne is telemetry only. Total reward is finite and clipped to
`[-50, 50]`. `env.py` owns physical reset/step, observation history, event
state, reward state, and fixed-shape info/metrics pytrees.

`evaluation.py` runs held-out deterministic seeds and derives pass/fail from
saved states and event metrics, never from video. `video.py` captures the
initial state and exactly one frame after every real environment transition,
stops at done, and renders saved states without environment interaction or
frame duplication.

`ppo.py` adapts the environment to Brax PPO, validates the exact batch layout,
runs one aligned block, and separates training, Brax evaluation, fixed
evaluation, diagnostic, and rendering counts. `checkpoint.py` stores Actor,
critic, observation normalizer, transition count, config/XML hashes, and field
orders. `provenance.py` freezes the run manifest before interaction and closes
the status and accounting after completion or abnormal exit.

## Data flow

Natural reset loads the XML keyframe, applies configured initial velocity,
performs MJX forward, creates empty history with a false valid mask, clears all
events and counters, and emits Actor/critic observations with a stable pytree.

For each control tick:

```text
normalized action
  -> clipped actuator targets
  -> one constant ctrl across four 0.005 s MJX Warp substeps
  -> next physical state at +0.020 s
  -> observable geometry and motion signals
  -> event-latch transition and terminal classification
  -> component reward and bounded total
  -> real FIFO update
  -> Actor/critic observations plus fixed-key info and metrics
```

PPO consumes only the Actor `state` key for the policy and the
`privileged_state` key for the value network. At declared checkpoints, a fixed
held-out panel uses deterministic actions, saves machine-readable traces, and
optionally renders a small representative subset. Rendering consumes zero
environment transitions.

## Reference use

The offline reference command validates CSV identity and reports declared
column ranges relevant to broad timing, position, velocity, posture, and joint
motion. The resolved Phase U config records the reference hash and the source
of initial Apex thresholds. No environment method imports or calls the
reference-analysis module, and a static dependency test enforces this boundary.

## Error handling and stopping

Configuration, model identity, object name/range, timing, action order,
observation order, or PPO layout mismatch fails before environment interaction.
Runtime nonfinite state produces a closed physical-failure outcome when it can
be represented safely; compilation, CUDA, OOM, checkpoint, or identity failure
closes the run as an engineering error and must not be converted into an
episode timeout or task failure.

Every run predeclares purpose, input hashes, requested and aligned interaction
cost, stopping conditions, seeds, and output directory. The PPO smoke stops
after one block or immediately on NaN, Inf, OOM, hash drift, checkpoint failure,
or abnormal process exit. No automatic fallback changes parallel environment
count; if 1,024 environments fail, the result is recorded and a reduced-env
retry requires a separately declared config and recalculated block size.

## Test and verification strategy

Implementation follows red-green-refactor. Pure tests cover timing arithmetic,
configuration identity, action order/ranges/knee sign, observation order and
FIFO behavior, event monotonicity, Apex gates, reward gating/bounds, terminal
classification, video frame accounting, and interaction accounting.

Host tests load the retained XML, verify the 2 kg payload and +/-50 N m hip/knee
force limits, confirm required joint/actuator/geom names, prove the in-memory
timestep override is 0.005, and validate offline rendering from a saved trace.

GPU tests verify single reset/step under `jax.jit`, batched reset/step under
`jax.vmap`, finite 1,024-environment short rollout state, and one PPO update.
Checkpoint restore is accepted only when field orders and config/XML hashes
match. The final JIT-local preflight runs static compilation, all JIT tests,
reference-boundary checks, GPU runtime smoke evidence checks, and PPO smoke
artifact validation.

## Outputs and claim boundary

Generated run data lives under `JIT/runs/phase_u/<run_id>/` and is ignored by
Git. A run contains the frozen manifest, resolved config, status, JSONL metrics,
checkpoint, fixed evaluation, state traces, optional videos, and an exact
resume command. Run outputs, policy parameters, logs, videos, and caches are
never committed.

The final report may claim only that named engineering contracts passed or
failed. A finite one-block PPO update is not evidence that Phase U is learnable,
that an Apex-capable policy exists, or that any Tube or safety property exists.
