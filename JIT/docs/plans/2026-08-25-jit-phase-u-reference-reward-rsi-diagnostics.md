# JIT Phase U Reference Reward, RSI, and Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Phase U shaping contract with the approved target-free reference reward, one-shot jump signal, bounded airborne RSI, height/descent Apex termination, and synchronized numeric/PNG/video diagnostics.

**Architecture:** Pure JAX modules own configuration, event state, reward math, observations, and reset selection. The environment exposes one mixed training reset and one forced-natural evaluation reset; Host-only evaluation and video modules serialize and render already captured traces without stepping physics. A v2 config/checkpoint identity makes new training incompatible with v1 parameters, while provenance verification continues to understand retained v1 runs.

**Tech Stack:** Python 3.11, JAX/Flax, MuJoCo MJX-Warp, Brax PPO, NumPy, Matplotlib Agg, Mediapy, pytest.

## Global Constraints

- Work only under `/home/qy/DVGC/JIT`; do not modify the authoritative XML, reference CSV, or `/home/qy/mujoco_playground/.venv`.
- Keep one `Propulsion-Ascent` task; first valid Apex descent is terminal success.
- Jump zone is inclusive root x `[2.5, 3.1] m`; its current signal is `0 -> 1 -> 0` and never reopens after first exit.
- Training reset uses exactly `0.05` airborne RSI probability centered at `x=2.8 m`, `z=2.0 m`, `vx=2.0 m/s`, `vz=1.0 m/s`; formal held-out evaluation forces natural reset.
- Remove target-position/direction/deceleration rewards and approximate wheel-support observations.
- New Actor/critic dimensions are `76/106`; old checkpoints must be rejected.
- Do not launch PPO training in this implementation task.
- Do not commit intermediate tasks; create at most one explicitly staged JIT-only commit after complete validation if the user requests delivery.

---

### Task 1: Versioned configuration, event, and reward contracts

**Files:**
- Modify: `JIT/configs/phase_u_smoke.json`
- Modify: `JIT/configs/phase_u_formal.json`
- Modify: `JIT/src/jit_dvgc/config.py`
- Modify: `JIT/src/jit_dvgc/constants.py`
- Modify: `JIT/src/jit_dvgc/semantics.py`
- Modify: `JIT/src/jit_dvgc/rewards.py`
- Modify: `JIT/tests/test_contracts.py`
- Modify: `JIT/tests/test_formal_config.py`
- Modify: `JIT/tests/test_semantics.py`
- Modify: `JIT/tests/test_rewards.py`

**Interfaces:**
- Produces: `ResetConfig` with `airborne_rsi_probability` and exact min/max state bounds.
- Produces: `EventConfig(jump_zone_x_min, jump_zone_x_max, min_ascent_velocity, apex_height, min_descent_velocity)`.
- Produces: `initial_event_state(root_x) -> EventState` and `advance_events(previous, signals, config) -> EventState` with `jump_signal`, `jump_zone_seen`, `jump_zone_consumed`, `ascending_seen`, `height_seen`, and `apex_seen`.
- Produces: `phase_u_reward(inputs, config, physical_limits) -> RewardResult` with fixed component keys and `unclipped_total`.

- [x] **Step 1: Write literal contract and event tests**

Add tests that independently assert v2 schema loading, jump bounds `2.5/3.1`, RSI probability `0.05`, reward weights, 25 frame fields, and the exact reset/transition table:

```python
assert initial_event_state(jp.array(2.4), cfg).jump_signal == 0
assert initial_event_state(jp.array(2.8), cfg).jump_signal == 1
assert initial_event_state(jp.array(3.2), cfg).jump_zone_consumed
entered = advance_events(initial_event_state(jp.array(2.4), cfg), signals(x=2.8), cfg)
left = advance_events(entered, signals(x=3.2), cfg)
reentered = advance_events(left, signals(x=2.8), cfg)
assert (entered.jump_signal, left.jump_signal, reentered.jump_signal) == (1, 0, 0)
```

Name the caught mutations: wrong inclusive boundary, reopening after exit, pre-marking RSI ascent, accepting old component names, and retaining target fields.

- [x] **Step 2: Run Task 1 contract tests and verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_contracts.py JIT/tests/test_formal_config.py JIT/tests/test_semantics.py -q
```

Expected: failures on v1 schemas, old relative-x event fields, old 27-value frame, and missing one-shot event members.

- [x] **Step 3: Write literal reward tests**

Use hand-derived values for roll/pitch/yaw piecewise boundaries, speed Gaussian, height curve, action costs, pitch-rate cost, joint mechanical energy, terminal costs, and total clipping. Include:

```python
assert reward(z=0.50, jump_signal=False).components.height == 0.0
assert reward(z=0.50, jump_signal=True).components.height == pytest.approx(30.0)
assert reward(z=0.80, jump_signal=True).components.height == pytest.approx(12.0)
assert reward(z=0.81, jump_signal=True).components.height == pytest.approx(8.0)
```

Assert no `target`, `drive`, `liftoff`, `stable_airborne`, `clearance`, or `apex_progress` component exists.

- [x] **Step 4: Run reward tests and verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_rewards.py -q
```

Expected: failures because old reward fields and formulas remain.

- [x] **Step 5: Implement minimal v2 config, events, and reward**

Introduce v2 schemas, exact dataclasses and validation, new component keys, one-shot root-x signal state, height/ascent/descent Apex state, and the approved target-free reward. Convert Euler radians to degrees only inside reward math; use measured hip/knee actuator forces and velocities supplied in `RewardState`. Compute component sum before clipping and expose it as `RewardResult.unclipped_total`.

- [x] **Step 6: Run Task 1 tests and verify GREEN**

Run both commands from Steps 2 and 4. Expected: all selected tests pass with no warnings.

### Task 2: Mixed training reset, forced-natural evaluation, and 76/106 observations

**Files:**
- Modify: `JIT/src/jit_dvgc/observation.py`
- Modify: `JIT/src/jit_dvgc/env.py`
- Modify: `JIT/src/jit_dvgc/checkpoint.py`
- Modify: `JIT/src/jit_dvgc/ppo.py`
- Modify: `JIT/src/jit_dvgc/formal_training.py`
- Modify: `JIT/tests/test_observation.py`
- Modify: `JIT/tests/test_env_host.py`
- Modify: `JIT/tests/test_env_gpu.py`
- Modify: `JIT/tests/test_checkpoint.py`
- Modify: `JIT/tests/test_formal_training.py`

**Interfaces:**
- Produces: `actor_observation(history, jump_signal) -> jax.Array[(76,)]`.
- Produces: `privileged_observation(data, actor_obs, geometry) -> jax.Array[(106,)]` without wheel-support booleans or platform-relative clearance.
- Produces: `TwoPhaseBikeEnv.reset(rng)` for mixed training resets and `TwoPhaseBikeEnv.reset_natural(rng)` for evaluation.
- Produces: checkpoint identity field `actor_task_fields=("jump_signal",)`.

- [x] **Step 1: Write observation and reset behavior tests**

Assert exact 25-field frame order with `root_height` replacing structure clearance, current jump signal appended once outside history, dimensions `76/106`, and no wheel-support field. Add deterministic reset tests using known PRNG keys to observe both reset sources, exact bounds, signal state, and `reset/source_airborne_rsi`. Assert `reset_natural` always returns the keyframe state and signal zero.

- [x] **Step 2: Run observation/Host tests and verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_observation.py JIT/tests/test_env_host.py JIT/tests/test_checkpoint.py -q
```

Expected: old `81/114` shapes, missing `reset_natural`, and old identity fields fail.

- [x] **Step 3: Implement reset and observation changes**

Build one `_reset(rng, force_natural)` path so natural and mixed reset produce identical fixed pytrees. Sample RSI x/z/vx/vz from the approved closed ranges, retain keyframe orientation/joints, initialize one-shot events from root x, place reset source and jump signal in metrics/info, and make both formal and smoke fixed diagnostics use `reset_natural`. Pass `data.actuator_force[2:4]` and hip/knee qvel into reward state.

- [x] **Step 4: Verify Host GREEN, then GPU RED/GREEN**

Run the Step 2 command, then:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_env_gpu.py -m gpu -q
```

Expected: JIT reset/step pytrees match, 1,024-environment state is finite, and shapes are `(1024,76)` and `(1024,106)`.

### Task 3: Trace metrics, diagnostic PNG/data, and synchronized video

**Files:**
- Create: `JIT/src/jit_dvgc/diagnostics.py`
- Modify: `JIT/src/jit_dvgc/evaluation.py`
- Modify: `JIT/src/jit_dvgc/video.py`
- Modify: `JIT/src/jit_dvgc/formal_training.py`
- Modify: `JIT/src/jit_dvgc/ppo.py`
- Modify: `JIT/tests/test_evaluation.py`
- Create: `JIT/tests/test_diagnostics.py`
- Modify: `JIT/tests/test_video.py`

**Interfaces:**
- Produces: `trace_series(trace) -> dict[str, np.ndarray]` with aligned reward, state, pose, velocity, action/control, energy, event, and terminal arrays.
- Produces: `save_trace_dashboard(trace, path, reward_scaling) -> DiagnosticReport`.
- Extends: `render_trace(...) -> VideoReport` with diagnostic PNG/data paths and hashes.

- [x] **Step 1: Write failing diagnostic artifact tests**

Create a four-state real `EpisodeTrace` fixture with literal rewards and metrics. Assert the data bundle contains every component plus `reward_unclipped`, clipped reward, PPO-scaled reward, x/y/z, roll/pitch/yaw, angular/linear velocity, action/control, hip/knee force/velocity/power, jump signal, reset source, height/ascent/Apex events, and terminal code. Assert all arrays have four samples and the PNG has nonzero size.

- [x] **Step 2: Run diagnostic/evaluation tests and verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_evaluation.py JIT/tests/test_diagnostics.py JIT/tests/test_video.py -q
```

Expected: missing diagnostics module/report fields and old evaluation metric names.

- [x] **Step 3: Implement trace serialization and plots**

Use Matplotlib's noninteractive Agg backend. Save a multi-panel full trajectory PNG and compressed aligned NPZ. Extend video frames with a right-side telemetry panel showing current component bars, total/unclipped reward, root position, Euler pose, velocity, jump signal, RSI source, and terminal reason. Render only captured states; never call `env.step`.

- [x] **Step 4: Run diagnostic/evaluation/video tests and verify GREEN**

Run the Step 2 command. Also decode the produced PNG and MP4 with Mediapy,
assert the PNG has nonzero width/height, and assert the encoded video contains
exactly one frame per captured state.

### Task 4: Backward provenance, documentation, and old-run audit

**Files:**
- Modify: `JIT/src/jit_dvgc/provenance.py`
- Modify: `JIT/tests/test_provenance_verify.py`
- Modify: `JIT/tests/test_formal_provenance.py`
- Modify: `JIT/README.md`
- Modify: `JIT/docs/VERIFICATION.md`
- Modify: `JIT/planning/task_plan.md`
- Modify: `JIT/planning/findings.md`
- Modify: `JIT/planning/progress.md`

**Interfaces:**
- `verify_run` accepts retained formal v1 evidence and new formal v2 evidence, validating only artifacts declared by that schema.
- New run reports require diagnostic PNG/data hashes; old v1 reports retain their original video/state requirements.

- [x] **Step 1: Write backward-verification tests**

Use synthetic v1 and v2 closed-run fixtures. Assert v1 remains verifiable without new plot fields, v2 requires PNG/data artifacts and hashes, and a v1 checkpoint cannot warm-start a v2 config because config/observation identity differs.

- [x] **Step 2: Run provenance tests and verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_provenance_verify.py JIT/tests/test_formal_provenance.py JIT/tests/test_formal_training.py -q
```

Expected: v2 formal schema and new artifact requirements are unsupported.

- [x] **Step 3: Implement schema-aware verification and update docs**

Branch verifier requirements by resolved schema, preserve v1 semantics exactly, and add v2 diagnostic requirements. Document the new formulas, dimensions, reset mixture, natural-only claim boundary, checkpoint incompatibility, and the fact that no new PPO training has been run.

- [x] **Step 4: Run provenance tests and retained old-run verification**

Run the Step 2 command and:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m jit_dvgc.provenance verify-run JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824_retry1
```

Expected: tests pass and the retained v1 completed run still verifies.

### Task 5: Complete verification and staged-content audit

**Files:**
- Modify only as required by failures proven within JIT scope.

- [x] **Step 1: Run static and focused complete JIT tests**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall -q JIT/src JIT/cli JIT/tests
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests -m "not gpu" -q
/home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests -m gpu -q
```

- [x] **Step 2: Run JIT local preflight**

```bash
bash JIT/scripts/local_preflight.sh
```

Expected: exit zero without PPO training or run creation.

- [ ] **Step 3: Inspect behavior and repository scope**

Run `git diff --check -- JIT`, inspect reward/observation/config diffs, and confirm no files outside JIT changed by this task. Verify existing user dirty paths remain untouched and no `JIT/runs`, checkpoints, logs, videos, caches, or generated diagnostics are staged.

- [ ] **Step 4: Perform final self-review**

Map every design requirement to a passing test, scan for stale v1 reward names and target/deceleration semantics in active v2 source/config, and verify all new public functions have behavior tests. Record exact commands/results in `JIT/planning/progress.md` and `JIT/docs/VERIFICATION.md`.

- [ ] **Step 5: Commit only after user-authorized delivery**

Explicitly stage the validated JIT source/config/test/docs/planning paths, inspect `git diff --cached --name-only` and `git diff --cached --check`, then create one focused commit. Do not push unless separately requested.
