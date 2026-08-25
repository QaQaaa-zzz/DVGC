# JIT Absolute Joint Targets and Fresh 5M Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the active knee incremental command with the same keyframe-centered absolute target rule as hip, verify a new identity-isolated Brax PPO contract, and launch one fresh aligned 4,988,928-transition Phase U run.

**Architecture:** Keep the stable JIT entrypoint and historical v1/v2 verifiers. Add an explicit action-semantics field and a new formal config whose canonical hash isolates checkpoints, generalize formal scheduling to values supplied by an exactly validated config, and preserve natural-only promotion panels. Launch only after one JIT-only commit is pushed.

**Tech Stack:** Python 3.12, JAX 0.6.2, MuJoCo 3.6.0, MJX Warp, MuJoCo Playground, Brax 0.14.2 PPO, pytest, JSON/JSONL, NumPy NPZ, Git.

## Global Constraints

- Work only in `/home/qy/DVGC`; all generated task content stays under `JIT/`.
- Use `/home/qy/mujoco_playground/.venv/bin/python` directly and do not modify that environment.
- Do not modify the authoritative XML, meshes, collision geometry, physics timing, reward formula, reset distribution, observations, jump signal, Apex predicate, or action order.
- Preserve unrelated user changes outside `JIT/`.
- The initial 4,988,928-transition launch must be fresh and must not restore any old checkpoint.
- Explicitly stage only validated `JIT/` paths; do not use `git add .` or `git add -A`.
- Do not commit `JIT/runs/`, checkpoints, logs, videos, images, NPZ data, caches, or PID files.

---

### Task 1: Absolute hip/knee action contract

**Files:**
- Modify: `JIT/tests/test_action_mapping.py`
- Modify: `JIT/src/jit_dvgc/action_mapping.py`
- Modify: `JIT/src/jit_dvgc/model.py`
- Modify: `JIT/src/jit_dvgc/config.py`
- Create: `JIT/configs/phase_u_absolute_smoke.json`

**Interfaces:**
- Consumes: authoritative XML keyframe hip/knee positions and actuator control ranges.
- Produces: `ActionMapping.joint_target_semantics`, `hip_initial`, `knee_initial`, and absolute controls from `map_action(action, knee_position, mapping) -> jax.Array[(4,)]`. The retained `knee_position` argument remains API-compatible but is ignored for the new semantics.

- [x] **Step 1: Write the failing absolute-knee tests**

Replace incremental expectations with real endpoint and interior assertions:

```python
def test_absolute_hip_and_knee_targets_are_centered_on_keyframe(mapping):
    zero = map_action(jp.zeros(4), jp.asarray(-0.25), mapping)
    np.testing.assert_allclose(zero, [0.0, 12.0, -1.2, 2.5], atol=1e-6)

    negative = map_action(jp.array([0.0, 0.0, -1.0, -1.0]), jp.asarray(-0.25), mapping)
    np.testing.assert_allclose(negative[2:], [-1.3, -1.5], atol=1e-6)

    half_negative = map_action(jp.array([0.0, 0.0, -0.5, -0.5]), jp.asarray(1.0), mapping)
    np.testing.assert_allclose(half_negative[2:], [-1.25, 0.5], atol=1e-6)

    positive = map_action(jp.array([0.0, 0.0, 1.0, 1.0]), jp.asarray(-1.5), mapping)
    np.testing.assert_allclose(positive[2:], [0.5, 2.5], atol=1e-6)
```

Also assert that two different live knee positions produce the same knee
target for the same normalized action.

- [x] **Step 2: Run the focused test and verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  -m pytest JIT/tests/test_action_mapping.py -q
```

Expected: failure because the current knee target depends on live knee position.

- [x] **Step 3: Implement one shared piecewise absolute helper**

Implement:

```python
def _piecewise_absolute_target(action, initial, lower, upper):
    return jp.where(
        action >= 0.0,
        initial + action * (upper - initial),
        initial + action * (initial - lower),
    )
```

Use it for both hip and knee. Construct `knee_initial` from the authoritative
XML keyframe in `load_host_model`. Keep v2 incremental parsing and its retained
JSON configs only for historical evidence. Require the new explicit semantics
token in `phase_u_absolute_smoke.json` and every active absolute config.

- [x] **Step 4: Run focused mapping/model/config tests and verify GREEN**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  -m pytest JIT/tests/test_action_mapping.py JIT/tests/test_model.py \
  JIT/tests/test_contracts.py JIT/tests/test_formal_config.py -q
```

Expected: all selected tests pass, the new smoke config parses with absolute
semantics, and retained v2 configs still parse with their historical semantics.

### Task 2: New exact five-million PPO configuration

**Files:**
- Create: `JIT/configs/phase_u_absolute_5m.json`
- Modify: `JIT/tests/test_formal_config.py`
- Modify: `JIT/src/jit_dvgc/config.py`

**Interfaces:**
- Consumes: `PPOConfig.block_transitions` and the approved v3 action/reward/reset/model contracts.
- Produces: one `ResolvedConfig` with target `4_988_928`, block size `24_576`, seed `820201`, held-out keys `930001..930008`, and six exact checkpoint milestones.

- [x] **Step 1: Write failing exact-config tests**

Assert all exact values:

```python
config = load_config(jit_root / "configs" / "phase_u_absolute_5m.json")
assert config.ppo.block_transitions == 24_576
assert config.ppo.requested_transitions == 4_988_928
assert config.ppo.num_parallel_envs == 384
assert config.ppo.unroll_length == 64
assert config.ppo.batch_size == 16
assert config.ppo.num_minibatches == 24
assert config.ppo.num_updates_per_batch == 8
assert config.ppo.entropy_cost == 0.01
assert config.formal.checkpoint_transitions[-1] == 4_988_928
```

Mutation tests must reject a changed action semantic, target, learning rate,
update count, seed, model capacity, reward coefficient, reset bound, or
milestone.

- [x] **Step 2: Run the new config tests and verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  -m pytest JIT/tests/test_formal_config.py -q
```

Expected: failure because the new schema/config does not yet exist.

- [x] **Step 3: Add the exact config and schema validation**

Create `phase_u_absolute_5m.json` with the approved values and generalize
`FormalTrainingConfig.formal_blocks` to derive
`requested_transitions // block_transitions` through `ResolvedConfig` rather
than hard-coding 39. Preserve exact v2 validation and add a separate exact
approved contract for the new schema.

- [x] **Step 4: Run config tests and verify GREEN**

Run the focused command from Step 2. Expected: all tests pass.

### Task 3: Dynamic formal schedule and fresh-start identity

**Files:**
- Modify: `JIT/tests/test_formal_training.py`
- Modify: `JIT/tests/test_formal_provenance.py`
- Modify: `JIT/src/jit_dvgc/formal_training.py`
- Modify: `JIT/src/jit_dvgc/provenance.py`
- Modify: `JIT/cli/train_phase_expert.py`

**Interfaces:**
- Consumes: exact formal target, block, checkpoint, and evaluation schedules from `ResolvedConfig`.
- Produces: one uninterrupted fresh formal segment, identity-bound checkpoint/evaluation evidence, strict completed-run verification, and same-config-only warm recovery after abnormal exit.

- [x] **Step 1: Write failing 5M controller/runner/provenance tests**

Use an injected fake trainer to assert:

```python
assert kwargs["num_timesteps"] == 4_988_928
assert kwargs["num_envs"] == 384
assert kwargs["unroll_length"] == 64
assert kwargs["batch_size"] == 16
assert kwargs["num_minibatches"] == 24
assert kwargs["num_updates_per_batch"] == 8
assert kwargs["restore_params"] is None
```

Assert the manifest has no parent checkpoint, begins at zero, and closes only
after the exact configured checkpoints/evaluations. Assert verifier acceptance
for the new schema and continued acceptance of retained completed v1/v2 runs.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  -m pytest JIT/tests/test_formal_training.py \
  JIT/tests/test_formal_provenance.py -q
```

Expected: failures from the hard-coded 998,400 target/schedules and unsupported
new schema.

- [x] **Step 3: Generalize from exact validated config values**

Remove hard-coded v2 target and milestone assumptions from report validation,
run declarations, and completed-run provenance. Do not weaken validation:
`resolve_config_payload` must first prove the schema is one of the exact
approved contracts, after which the runner/verifier consume its frozen target
and schedules. Keep `--restore-checkpoint` available only for an abnormal
same-config recovery; the initial launch command omits it.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass, including old
formal verification fixtures.

### Task 4: Evaluation evidence and documentation

**Files:**
- Modify: `JIT/tests/test_env_gpu.py`
- Modify: `JIT/tests/test_evaluation.py`
- Modify: `JIT/src/jit_dvgc/env.py`
- Modify: `JIT/src/jit_dvgc/formal_training.py`
- Modify: `JIT/scripts/local_preflight.sh`
- Modify: `JIT/README.md`
- Modify: `JIT/docs/VERIFICATION.md`

**Interfaces:**
- Consumes: new absolute action config and formal milestones.
- Produces: forced-airborne-RSI reset for diagnostic panels, separately named natural/RSI summaries, and current preflight/documentation without changing promotion semantics.

- [x] **Step 1: Write failing forced-RSI and evidence-separation tests**

Assert `reset_airborne_rsi(key)` always yields bounded configured RSI state,
`jump_signal=1`, and `reset/source_airborne_rsi=1`. Assert every new-run
milestone writes natural promotion summaries and forced-RSI diagnostic
summaries to different paths and totals without opening an old checkpoint.

- [x] **Step 2: Run focused tests and verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  -m pytest JIT/tests/test_env_gpu.py JIT/tests/test_evaluation.py -q
```

Expected: failure because a public forced-RSI reset/panel does not yet exist.

- [x] **Step 3: Implement the minimum separated diagnostic path**

Expose the existing bounded RSI reset as `reset_airborne_rsi`. Save forced-RSI
diagnostics for the current callback parameters outside natural promotion panel
directories and account their transitions as diagnostics, never fixed
evaluation. Update preflight to parse both active absolute configs and keep
retained v1/v2 run verification.

- [x] **Step 4: Run focused tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

### Task 5: Full verification, review, commit, and launch

**Files:**
- Modify after execution: `JIT/planning/task_plan.md`
- Modify after execution: `JIT/planning/findings.md`
- Modify after execution: `JIT/planning/progress.md`
- Create after completion: `JIT/docs/experiments/phase_u_absolute_5m_seed820201_20260825/REPORT.md`

**Interfaces:**
- Consumes: all validated source/config/tests/docs from Tasks 1-4.
- Produces: one pushed JIT-only source commit, one predeclared fresh persistent run, strict evidence verification, and a complete experiment report.

- [x] **Step 1: Run static and complete non-GPU verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall -q JIT/src JIT/cli
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  -m pytest JIT/tests -q -m "not gpu"
```

- [x] **Step 2: Run GPU environment and one-block PPO verification**

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python \
  -m pytest JIT/tests/test_env_gpu.py -q -m gpu
bash JIT/scripts/local_preflight.sh
```

- [x] **Step 3: Review the complete JIT diff and request code review**

Confirm no path outside `JIT/` is staged or modified by this task. Resolve all
Critical and Important review findings, then rerun affected tests.

- [x] **Step 4: Create one explicit JIT-only commit and push**

```bash
git add JIT/src JIT/tests JIT/configs JIT/cli JIT/scripts \
  JIT/README.md JIT/docs JIT/planning
git diff --cached --check
git diff --cached --name-only
git commit -m "feat(jit): use absolute joint targets for fresh Phase U training"
git push origin agent/two-phase-soft-tube
```

- [x] **Step 5: Predeclare and launch the fresh aligned run**

Use unique run ID `phase_u_absolute_4988928_seed820201_20260825` and no
`--restore-checkpoint` argument:

```bash
mkdir -p JIT/runs/phase_u
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_absolute_5m.json \
  --run-id phase_u_absolute_4988928_seed820201_20260825 --formal \
  > JIT/runs/phase_u/phase_u_absolute_4988928_seed820201_20260825.launch.log 2>&1 \
  < /dev/null &
```

Record the PID under the same run namespace. Inspect startup once, then only
the declared milestones, completion, or abnormal exit.

- [x] **Step 6: Verify and analyze completed evidence**

Run strict provenance verification, inspect all natural and forced-RSI panels,
plot/video/NPZ artifacts, KL/std/action trends, returns and terminal causes,
then write the complete report without promoting a failed policy.
