# JIT Phase U Engineering Delivery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Multi-agent delegation is not used for this repository task.

**Goal:** Build and verify an independent, auditable Phase U engineering-smoke stack entirely under `JIT/`.

**Architecture:** A unique `jit_dvgc` package separates pure contracts, Host model audit, JAX geometry/semantics/reward, the MJX-Warp environment, evaluation/video, and Brax PPO orchestration. The only external data dependencies are the retained authoritative XML and weak-prior reference CSV, both read-only and identity checked.

**Tech Stack:** Python 3.12, JAX 0.6.2, MuJoCo 3.6.0, MuJoCo MJX Warp, MuJoCo Playground `MjxEnv`, Brax 0.14.2 PPO, pytest, NumPy, MediaPy.

## Global Constraints

- Create or modify files only below `/home/qy/DVGC/JIT`.
- Do not import any module whose top-level package is `dvgc`.
- Use `/home/qy/mujoco_playground/.venv/bin/python` directly; do not install or reconfigure anything.
- Load `/home/qy/DVGC/assets/orange_bike_4kg_horizontal.xml` directly; do not copy or edit it.
- Preserve the 2 kg payload, hip/knee force ranges `[-50, 50]`, and action order `[steer, rear-wheel drive, hip, knee]`.
- Use `SIM_DT=0.005`, `CTRL_DT=0.020`, and `N_SUBSTEPS=4`.
- Use `data/reference_jump.csv` only for offline weak-prior analysis and provenance.
- Implement Propulsion-Ascent only; do not add placeholder Phase D, Tube, unified PPO, or JCE/JEL code.
- Do not launch the 998,400-transition formal run.
- Run outputs stay under `JIT/runs/` and remain ignored by Git.
- Do not commit between tasks. After complete verification, explicitly stage only `JIT/` and make one focused commit.

---

### Task 1: Project contracts, configuration, and provenance

**Files:**

- Create: `JIT/.gitignore`
- Create: `JIT/README.md`
- Create: `JIT/configs/phase_u_smoke.json`
- Create: `JIT/src/jit_dvgc/__init__.py`
- Create: `JIT/src/jit_dvgc/constants.py`
- Create: `JIT/src/jit_dvgc/config.py`
- Create: `JIT/src/jit_dvgc/provenance.py`
- Create: `JIT/src/jit_dvgc/reference_analysis.py`
- Create: `JIT/tests/conftest.py`
- Create: `JIT/tests/test_contracts.py`
- Create: `JIT/tests/test_reference_boundary.py`

**Interfaces:**

- Produces: `load_config(path: Path) -> ResolvedConfig`
- Produces: `canonical_sha256(value: Mapping[str, Any]) -> str`
- Produces: `file_sha256(path: Path) -> str`
- Produces: `predeclare_run(spec: RunDeclaration) -> Path`
- Produces: `close_run(run_dir: Path, status: str, accounting: InteractionAccounting, reason: str) -> None`
- Produces: `analyze_reference(path: Path) -> dict[str, Any]`

- [x] **Step 1: Write failing contract and reference-boundary tests**

```python
def test_fixed_orders_and_timing():
    assert SIM_DT == 0.005
    assert CTRL_DT == 0.020
    assert N_SUBSTEPS == 4
    assert ACTION_ORDER == ("steer", "rear_wheel_drive", "hip", "knee")
    assert len(ACTOR_FRAME_FIELDS) == 27

def test_smoke_layout_is_one_exact_block():
    cfg = load_config(JIT_ROOT / "configs/phase_u_smoke.json")
    assert cfg.ppo.num_parallel_envs == 1024
    assert cfg.ppo.block_transitions == 25_600
    assert cfg.ppo.requested_transitions == 25_600

def test_training_package_has_no_dvgc_imports():
    forbidden = []
    for path in (JIT_ROOT / "src/jit_dvgc").glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden.extend(n.name for n in node.names if n.name == "dvgc" or n.name.startswith("dvgc."))
            if isinstance(node, ast.ImportFrom) and node.module and (node.module == "dvgc" or node.module.startswith("dvgc.")):
                forbidden.append(node.module)
    assert forbidden == []

def test_reference_module_is_absent_from_runtime_import_closure():
    for name in ("env.py", "ppo.py", "rewards.py", "semantics.py"):
        assert "reference_analysis" not in (JIT_ROOT / "src/jit_dvgc" / name).read_text()
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_contracts.py JIT/tests/test_reference_boundary.py -q
```

Expected: collection fails because `jit_dvgc.constants` and `jit_dvgc.config` do not exist.

- [x] **Step 3: Implement immutable constants and validated config**

Use frozen dataclasses `PPOConfig`, `ApexConfig`, `RewardConfig`, and
`ResolvedConfig`. Validate positive finite values, exact timing ratio, exact
action order, `batch_size * num_minibatches % num_parallel_envs == 0`, and
`requested_transitions % block_transitions == 0`. Compute:

```python
block_transitions = unroll_length * batch_size * num_minibatches
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
config_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
```

The smoke JSON fixes 1,024 envs, horizon 200, unroll 25, batch 128, eight
minibatches, one update, one block, training seed 820001, and held-out seeds
920001 through 920008. It records the guide's Apex thresholds and reward
weights exactly.

- [x] **Step 4: Implement closed run provenance and offline reference analysis**

`predeclare_run` creates a new directory only if absent and atomically writes
`run_manifest.json`, `resolved_config.json`, `status.json`, and
`resume_command.txt` before interaction. `close_run` requires nonnegative
training/Brax/fixed/diagnostic counts and writes one terminal status.
`analyze_reference` validates the expected CSV hash, parses numeric columns,
and reports row count, time interval, and min/max for declared kinematic fields.

- [x] **Step 5: Verify GREEN and update planning records**

Run the Task 1 test command and expect all tests to pass. Record the red and
green outputs in `JIT/planning/progress.md`.

---

### Task 2: Authoritative model audit and action mapping

**Files:**

- Create: `JIT/src/jit_dvgc/model.py`
- Create: `JIT/src/jit_dvgc/action_mapping.py`
- Create: `JIT/tests/test_model.py`
- Create: `JIT/tests/test_action_mapping.py`

**Interfaces:**

- Consumes: `ResolvedConfig`
- Produces: `load_host_model(config: ResolvedConfig) -> ModelBundle`
- Produces: `put_warp_model(bundle: ModelBundle) -> ModelBundle`
- Produces: `map_action(action: jax.Array, knee_position: jax.Array, mapping: ActionMapping) -> jax.Array`

- [x] **Step 1: Write failing Host model and mapping tests**

```python
def test_authoritative_model_contract():
    bundle = load_host_model(load_smoke_config())
    assert bundle.xml_sha256 == EXPECTED_XML_SHA256
    assert bundle.mj_model.opt.timestep == pytest.approx(0.005)
    assert bundle.actuator_names == ("cmd_steering_v", "cmd_rearwheel_f", "cmd_hip_f", "cmd_knee_f")
    np.testing.assert_array_equal(bundle.mj_model.actuator_forcerange[2:], [[-50, 50], [-50, 50]])
    assert float(bundle.mj_model.geom("load").mass) == pytest.approx(2.0)

@pytest.mark.parametrize("knee_action,expected", [(1.0, 2.3), (-1.0, 2.5), (0.0, 2.5)])
def test_knee_sign_and_clipping(knee_action, expected):
    ctrl = map_action(jp.array([0.0, 0.0, 0.0, knee_action]), jp.array(2.5), mapping_fixture())
    assert float(ctrl[3]) == pytest.approx(expected)

def test_mapping_is_jittable_and_batched():
    actions = jp.zeros((16, 4))
    ctrl = jax.jit(jax.vmap(map_action, in_axes=(0, 0, None)))(actions, jp.full((16,), 2.5), mapping_fixture())
    assert ctrl.shape == (16, 4)
    assert bool(jp.isfinite(ctrl).all())
```

- [x] **Step 2: Verify RED**

Run the two Task 2 test files. Expected: import failure because model and mapping modules are absent.

- [x] **Step 3: Implement model identity/audit and mapping**

Resolve repository root from `Path(__file__).resolve().parents[3]`. Verify XML
hash before loading. Set `mj_model.opt.timestep = SIM_DT` before
`mjx.put_model(mj_model, impl="warp")`. Resolve all joint, actuator, sensor,
keyframe, and collision geom ids by name. Derive control and joint ranges from
the model. Map:

```python
steer = clip(action[0], -1, 1) * steer_limit
rear = clip(base_rear_speed + clip(action[1], -1, 1) * rear_speed_delta, rear_min, rear_max)
hip = hip_min + 0.5 * (clip(action[2], -1, 1) + 1.0) * (hip_max - hip_min)
knee = clip(knee_position - clip(action[3], -1, 1) * knee_delta, knee_min, knee_max)
```

- [x] **Step 4: Verify GREEN and record results**

Run Task 2 tests and record exact counts and output.

---

### Task 3: Observable frame and real three-frame history

**Files:**

- Create: `JIT/src/jit_dvgc/observation.py`
- Create: `JIT/tests/test_observation.py`

**Interfaces:**

- Consumes: MJX data, `ModelIndex`, geometry signals, previous action
- Produces: `observable_frame(data: mjx.Data, model_index: ModelIndex, geometry: GeometrySignals, last_action: jax.Array, history_valid: jax.Array) -> jax.Array` with shape `(27,)`
- Produces: `initial_history() -> HistoryState`
- Produces: `advance_history(history: HistoryState, frame: jax.Array) -> HistoryState`
- Produces: `actor_observation(history: HistoryState) -> jax.Array` with shape `(81,)`
- Produces: `privileged_observation(data: mjx.Data, frame: jax.Array) -> jax.Array`

- [x] **Step 1: Write failing observation tests**

```python
def test_fifo_contains_three_consecutive_frames():
    history = initial_history()
    for value in (1.0, 2.0, 3.0, 4.0):
        history = advance_history(history, jp.full((27,), value))
    obs = actor_observation(history).reshape(3, 27)
    np.testing.assert_array_equal(obs[:, 0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(obs[:, -1], [1.0, 1.0, 1.0])

def test_reset_history_is_empty_and_not_fake_repeated_frames():
    obs = actor_observation(initial_history()).reshape(3, 27)
    np.testing.assert_array_equal(obs, np.zeros((3, 27)))

def test_actor_fields_exclude_privileged_or_outcome_data():
    forbidden = {"reward", "success", "terminated", "truncated", "end_code", "reference_index", "policy_hash"}
    assert forbidden.isdisjoint(ACTOR_FRAME_FIELDS)
```

- [x] **Step 2: Verify RED**

Run `JIT/tests/test_observation.py`; expect missing-module failure.

- [x] **Step 3: Implement JAX history and observations**

Use a `flax.struct.dataclass` with `frames: (3,27)` and `valid_count: int32`.
Shift with `jp.concatenate([frames[1:], frame[None]], axis=0)`, replace each
frame's final mask value from slot validity, and cap valid count at three.
Construct the exact declared field order from sensors and current measurable
signals. The critic concatenates Actor observation, qpos, qvel, exact root
velocity/attitude, clearance, and contact/support booleans.

- [x] **Step 4: Verify GREEN and record results**

Run Task 3 tests, including `jax.jit(advance_history)` and batched vmap coverage.

---

### Task 4: Complete-structure geometry and Phase U semantics

**Files:**

- Create: `JIT/src/jit_dvgc/geometry.py`
- Create: `JIT/src/jit_dvgc/semantics.py`
- Create: `JIT/tests/test_geometry.py`
- Create: `JIT/tests/test_semantics.py`

**Interfaces:**

- Consumes: Host model audit and MJX data
- Produces: `build_geometry_contract(mj_model) -> GeometryContract`
- Produces: `extract_geometry(data, contract) -> GeometrySignals`
- Produces: `initial_event_state() -> EventState`
- Produces: `advance_events(previous, signals, config) -> EventState`
- Produces: `classify_terminal(inputs: TerminalInputs, config: ResolvedConfig) -> TerminalState`

- [x] **Step 1: Write failing analytic geometry tests**

```python
def test_box_support_bound_is_orientation_aware():
    bounds = collision_support_bounds(jp.array([[1., 0., 2.]]), jp.eye(3)[None], jp.array([GEOM_BOX]), jp.array([[.2, .3, .4]]))
    assert float(bounds.max_x[0]) == pytest.approx(1.2)
    assert float(bounds.min_z[0]) == pytest.approx(1.6)

def test_relative_x_uses_frontmost_collision_structure():
    positions = jp.array([[3.0, 0.0, 0.5], [3.4, 0.0, 0.5]])
    rotations = jp.broadcast_to(jp.eye(3), (2, 3, 3))
    types = jp.array([GEOM_BOX, GEOM_BOX])
    sizes = jp.array([[0.1, 0.1, 0.1], [0.2, 0.1, 0.1]])
    signals = full_structure_metrics(
        positions, rotations, types, sizes,
        obstacle_front_x=3.6, obstacle_top_z=0.16,
    )
    assert float(signals.robot_frontmost_x) == pytest.approx(3.6)
    assert float(signals.obstacle_relative_x) == pytest.approx(0.0)
    assert float(signals.full_structure_clearance) == pytest.approx(0.24)

def test_authoritative_manifest_covers_all_collision_robot_geoms():
    contract = build_geometry_contract(load_smoke_model().mj_model)
    assert set(contract.robot_geom_names) == {"base_collision", "rearwheel_collision", "steer_collision", "frontwheel_collision", "downarm_collision", "knee_motor_collision", "uparm_collision"}
```

- [x] **Step 2: Write failing semantic tests**

```python
def test_window_latch_is_monotonic():
    event = initial_event_state()
    event = advance_events(event, signal_fixture(obstacle_relative_x=1.0), cfg)
    assert bool(event.window_latched)
    later = advance_events(event, signal_fixture(obstacle_relative_x=2.0), cfg)
    assert bool(later.window_latched)

def test_apex_requires_every_gate():
    assert bool(apex_membership(valid_apex_fixture(), cfg.apex))
    for field in APEX_GATE_FIELDS:
        assert not bool(apex_membership(invalidate(valid_apex_fixture(), field), cfg.apex))

def test_horizon_is_truncated_not_terminated():
    inputs = terminal_fixture(
        episode_step=199, success=False, physical_failure=False,
    )
    terminal = classify_terminal(inputs, cfg)
    assert not bool(terminal.terminated)
    assert bool(terminal.truncated)
    assert int(terminal.end_code) == END_TIMEOUT
```

- [x] **Step 3: Verify RED**

Run Task 4 tests; expect missing geometry and semantics modules.

- [x] **Step 4: Implement analytic support, contacts, events, and terminal precedence**

Support formulas cover sphere, capsule, ellipsoid, cylinder, and box.
Collision robot ids derive from contype/conaffinity and non-world body id.
Contact masks use `efc_address >= 0` and symmetric geom pairs. Terminal
precedence is nonfinite, roll, pitch, prohibited body contact, illegal wheel
contact, backward exit, platform overrun, Apex success, then horizon timeout.
Success and physical failure set `terminated`; only horizon sets `truncated`.

- [x] **Step 5: Verify GREEN and record results**

Run Task 4 tests under eager, JIT, and vmap paths.

---

### Task 5: Component-wise Phase U reward

**Files:**

- Create: `JIT/src/jit_dvgc/rewards.py`
- Create: `JIT/tests/test_rewards.py`

**Interfaces:**

- Consumes: `RewardInputs`, previous/current events, `RewardConfig`, `ApexConfig`
- Produces: `phase_u_reward(inputs, config, apex) -> RewardResult`
- `RewardResult.total` is scalar; `RewardResult.components` has exactly `REWARD_COMPONENT_KEYS`.

- [x] **Step 1: Write failing reward tests**

```python
def test_every_jump_positive_is_zero_before_window():
    result = phase_u_reward(positive_progress_fixture(window_latched=False), reward_cfg, apex_cfg)
    for key in ("liftoff", "stable_airborne", "ascent", "clearance", "apex_progress", "apex_success"):
        assert float(result.components[key]) == 0.0

def test_early_airborne_is_not_penalty_success_or_done():
    result = phase_u_reward(early_airborne_fixture(), reward_cfg, apex_cfg)
    assert float(result.components["physical_failure"]) == 0.0
    assert float(result.components["apex_success"]) == 0.0

def test_high_rotation_suppresses_motion_credit_and_adds_rate_penalty():
    result = phase_u_reward(ascent_fixture(angular_speed=20.0), reward_cfg, apex_cfg)
    assert float(result.components["ascent"]) < 1e-6
    assert float(result.components["rate"]) < 0.0

def test_total_is_finite_and_bounded():
    batched = jax.vmap(phase_u_reward, in_axes=(0, None, None))(extreme_batch(), reward_cfg, apex_cfg)
    assert bool(jp.isfinite(batched.total).all())
    assert bool((batched.total >= -50).all() & (batched.total <= 50).all())
```

- [x] **Step 2: Verify RED**

Run Task 5 tests; expect missing reward module.

- [x] **Step 3: Implement the guide's exact equations**

Compute pose/rate/motion quality, drive progress, one-time window/liftoff/
stable-airborne bonuses, gated ascent/clearance, seven-component mean Apex
score positive improvement, first Apex success, attitude/rate/action/contact/
failure/timeout penalties, and final clip. Use `jp.where` for event gates and
never Host boolean conversion.

- [x] **Step 4: Verify GREEN and record results**

Run Task 5 tests and assert component key stability between nominal and terminal inputs.

---

### Task 6: MJX-Warp environment integration

**Files:**

- Create: `JIT/src/jit_dvgc/env.py`
- Create: `JIT/tests/test_env_host.py`
- Create: `JIT/tests/test_env_gpu.py`

**Interfaces:**

- Consumes: Tasks 1–5
- Produces: `TwoPhaseBikeEnv(mjx_env.MjxEnv)`
- Produces: `reset(rng) -> mjx_env.State`
- Produces: `step(state, action) -> mjx_env.State`

- [x] **Step 1: Write failing Host integration tests**

```python
def test_environment_timing_and_shapes():
    env = TwoPhaseBikeEnv(load_smoke_config(), convert_model=False)
    assert env.sim_dt == pytest.approx(.005)
    assert env.dt == pytest.approx(.020)
    assert env.n_substeps == 4
    assert env.action_size == 4

def test_reset_and_step_have_identical_pytree_structure():
    env = gpu_env()
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    next_state = jax.jit(env.step)(state, jp.zeros(4))
    assert jax.tree.structure(state) == jax.tree.structure(next_state)
    assert float(next_state.data.time - state.data.time) == pytest.approx(.020, abs=1e-6)
```

- [x] **Step 2: Verify RED**

Host test fails because `TwoPhaseBikeEnv` is absent. GPU test is marked
`@pytest.mark.gpu` and is not run during this RED command.

- [x] **Step 3: Implement reset and step**

Reset uses `initial_state` keyframe qpos/qvel, applies configured initial x
velocity, creates MJX data with Warp capacity settings, calls `mjx.forward`,
and initializes stable metrics/info keys. Step maps action, advances exactly
four substeps with unchanged ctrl, computes signals/events/terminal/reward,
advances history once, and returns the same pytree structure. `done` is
`terminated | truncated` as float for Brax while booleans remain in `info`.

- [x] **Step 4: Verify Host GREEN**

Run pure/Host tests and static `jax.eval_shape` checks.

- [x] **Step 5: Verify GPU JIT and 1,024-env vmap**

Run outside device isolation with `XLA_PYTHON_CLIENT_PREALLOCATE=false`:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_env_gpu.py -m gpu -q
```

Tests compile one reset/step, 50 control ticks (time approximately 1.0 s), and
a four-tick rollout over 1,024 vmapped environments. They require finite qpos,
qvel, observations, reward, and metrics.

---

### Task 7: Evaluation, truthful video, and checkpoint contracts

**Files:**

- Create: `JIT/src/jit_dvgc/evaluation.py`
- Create: `JIT/src/jit_dvgc/video.py`
- Create: `JIT/src/jit_dvgc/checkpoint.py`
- Create: `JIT/tests/test_evaluation.py`
- Create: `JIT/tests/test_video.py`
- Create: `JIT/tests/test_checkpoint.py`

**Interfaces:**

- Produces: `capture_episode(env, policy, seed, horizon) -> EpisodeTrace`
- Produces: `summarize_phase_u(traces) -> dict[str, Any]`
- Produces: `render_trace(env, trace, path, fps=50) -> VideoReport`
- Produces: `save_checkpoint(path, payload: CheckpointPayload) -> None`
- Produces: `load_checkpoint(path, expected: CheckpointIdentity) -> CheckpointPayload`

- [x] **Step 1: Write failing evaluation/video tests**

```python
def test_capture_stops_at_done_and_counts_initial_state_once():
    trace = capture_episode(fake_done_after(3), zero_policy, seed=1, horizon=200)
    assert trace.environment_transitions == 3
    assert len(trace.frames) == 4
    assert trace.frames[-1].terminated or trace.frames[-1].truncated

def test_renderer_never_steps_and_never_duplicates_frames(tmp_path):
    trace = four_distinct_frames()
    report = render_trace(env_that_raises_on_step(), trace, tmp_path / "trace.mp4", fps=50)
    assert report.captured_state_count == 4
    assert report.encoded_frame_count == 4

def test_checkpoint_rejects_identity_mismatch(tmp_path):
    save_checkpoint(tmp_path / "checkpoint", valid_payload())
    with pytest.raises(ValueError, match="config_sha256"):
        load_checkpoint(tmp_path / "checkpoint", expected_identity(config_sha256="0" * 64))
```

- [x] **Step 2: Verify RED**

Run Task 7 tests; expect missing modules.

- [x] **Step 3: Implement capture, summary, render, and checkpoint identity**

Capture initial qpos/qvel/ctrl/action/reward components/info, append one state
per real step, and break immediately on done. Summary reports every guide-
required Phase U rate, extrema, end-reason count, component sum, and action
saturation fraction. Rendering reconstructs Host `MjData` from saved qpos,
qvel, ctrl and calls `mujoco.mj_forward`; it never calls `env.step`.
Checkpoint data stores normalizer, Actor, critic, transition number, config/XML
hash, Actor field order, and action order with an atomic sidecar identity hash.

- [x] **Step 4: Verify GREEN and record results**

Run Task 7 tests and confirm the encoded MP4 contains exactly the captured frame count.

---

### Task 8: Brax PPO runner and stable CLI

**Files:**

- Create: `JIT/src/jit_dvgc/ppo.py`
- Create: `JIT/cli/train_phase_expert.py`
- Create: `JIT/tests/test_ppo_contract.py`
- Create: `JIT/tests/test_cli.py`

**Interfaces:**

- Produces: `make_network_factory() -> Callable`
- Produces: `run_phase_u_smoke(config_path: Path, run_id: str) -> dict[str, Any]`
- CLI: `train_phase_expert.py --phase propulsion_ascent --config JIT/configs/phase_u_smoke.json --run-id phase_u_1024_one_block_20260824_seed820001 --smoke`

- [x] **Step 1: Write failing PPO/CLI tests**

```python
def test_network_factory_separates_actor_and_critic_keys():
    factory = make_network_factory()
    networks = factory({"state": (81,), "privileged_state": (PRIVILEGED_OBS_SIZE,)}, 4, identity)
    assert networks.policy_network is not None
    assert networks.value_network is not None

def test_cli_rejects_unimplemented_phase_before_creating_run(tmp_path):
    result = run_cli("--phase", "descent_recovery", "--run-id", "forbidden", env={"JIT_RUN_ROOT": str(tmp_path)})
    assert result.returncode != 0
    assert not (tmp_path / "forbidden").exists()

def test_interaction_accounting_closes_one_block():
    report = validate_smoke_report(smoke_report_fixture(training=25_600, fixed=0, diagnostic=0))
    assert report.total_environment_transitions == 25_600
```

- [x] **Step 2: Verify RED**

Run Task 8 contract tests; expect missing PPO and CLI implementations.

- [x] **Step 3: Implement asymmetric PPO and CLI orchestration**

Use `networks.make_ppo_networks` with policy key `state`, value key
`privileged_state`, tanh-normal actions, and hidden layers `(256,256,256)`.
Invoke `ppo.train` with the exact frozen smoke layout, normalized observations,
deterministic eval support, and one update. The CLI validates phase/config/GPU,
predeclares the run, saves transition-0 identity, runs PPO, saves final
checkpoint, reloads it, writes metrics/status, and closes accounting on success
or engineering failure.

- [x] **Step 4: Verify contract GREEN**

Run Task 8 non-GPU tests and record output.

- [x] **Step 5: Predeclare and run the one-block GPU PPO smoke**

Use run id `phase_u_1024_one_block_20260824_seed820001`. The manifest declares
purpose `compile_update_checkpoint_restore_engineering_smoke`, training ceiling
25,600, no formal-training claim, stop after one block, and output
`JIT/runs/phase_u/<run_id>/`.

Run outside device isolation:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py --phase propulsion_ascent --config JIT/configs/phase_u_smoke.json --run-id phase_u_1024_one_block_20260824_seed820001 --smoke
```

Inspect only completion/abnormal exit, final metrics, checkpoint identity, and
closed terminal causes. Do not interpret return or success rate as learnability.

---

### Task 9: JIT-local preflight, final audit, and single Git commit

**Files:**

- Create: `JIT/scripts/local_preflight.sh`
- Create: `JIT/docs/VERIFICATION.md`
- Modify: `JIT/planning/task_plan.md`
- Modify: `JIT/planning/findings.md`
- Modify: `JIT/planning/progress.md`

**Interfaces:**

- Produces: one noninteractive preflight command and a machine-readable evidence summary.

- [x] **Step 1: Implement preflight and its contract test**

The shell script uses the fixed interpreter and `PYTHONPATH=JIT/src`, then runs:

```bash
python -m compileall -q JIT/src JIT/cli
python -m pytest JIT/tests -q -m "not gpu"
python -m pytest JIT/tests/test_env_gpu.py -q -m gpu
python -m jit_dvgc.reference_analysis --input data/reference_jump.csv --output JIT/runs/reference_analysis.json
python -m jit_dvgc.provenance verify-run JIT/runs/phase_u/phase_u_1024_one_block_20260824_seed820001
```

It checks timing constants and absence of `dvgc` imports before dynamic tests.

- [x] **Step 2: Run complete fresh verification**

Run:

```bash
bash JIT/scripts/local_preflight.sh
```

Require exit code 0, complete output, zero failed tests, finite GPU smoke, and a
closed PPO run. Record exact test counts, elapsed time, hashes, interaction
counts, and terminal-cause panel in `JIT/docs/VERIFICATION.md`.

- [x] **Step 3: Audit requirements and repository isolation**

Re-read the design and this plan. Check every requirement has evidence. Run
`git status --short`, `git diff -- JIT`, and a source scan for placeholders,
forbidden imports, version-suffixed production files, committed run outputs,
and files outside `JIT/`. Preserve the user's pre-existing dirty paths.

- [x] **Step 4: Explicitly stage only JIT and inspect staged content**

```bash
git add JIT
git status --short
git diff --cached --stat
git diff --cached --check
```

Confirm no `JIT/runs/`, checkpoint, video, log, cache, or unrelated path is staged.

- [x] **Step 5: Create the single focused commit**

```bash
git commit -m "feat(jit): add independent phase U engineering smoke"
```

After commit, verify `git status --short` shows only the user's original
outside-JIT changes and report the commit id plus verification evidence.
