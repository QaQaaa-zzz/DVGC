# JIT Phase U v4 10M Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the active Phase U v4 contract in place, remove the CCD-capacity bottleneck, add one-shot post-run Codex analysis, verify the complete JIT tree, and launch one fresh 9,977,856-transition run.

**Architecture:** Keep the existing environment, PPO runner, formal evidence, and CLI boundaries. Extend the strict v4 configuration with explicit Warp CCD capacity and the approved event/reward/reset values, then add one independent read-only watcher CLI whose only external side effect is one `codex exec` invocation after terminal run status. Preserve v1-v3 behavior while intentionally replacing v4 compatibility.

**Tech Stack:** Python 3.12, JAX, MuJoCo MJX Warp, Brax PPO, pytest, Bash launch wrappers, Codex CLI.

## Global Constraints

- Work only in `/home/qy/DVGC`; every generated file and output stays under `JIT/`.
- Use `/home/qy/mujoco_playground/.venv/bin/python` without changing that environment.
- Do not edit the authoritative XML, collision geometry, timing, observations, action order, action mapping, networks, termination limits, or PPO optimizer layout.
- Active v4 exact values are `jump_zone_x_min=2.5`, `jump_zone_x_max=3.4`, `height_coeff=40.0`, `airborne_rsi_probability=0.08`, `naconmax=4096`, `naccdmax=256`, and `njmax=256`.
- The fresh formal target is 9,977,856 transitions with no parent checkpoint and no restore option.
- Preserve all user-owned changes outside `JIT/`.
- Do not make per-task commits; create and push one validated JIT implementation
  commit before every PPO interaction in this round, including the bounded smoke.

---

### Task 1: Replace the active v4 configuration contract

**Files:**
- Modify: `JIT/src/jit_dvgc/config.py`
- Modify: `JIT/configs/phase_u_continuation_smoke.json`
- Delete: `JIT/configs/phase_u_continuation_5m.json`
- Create: `JIT/configs/phase_u_continuation_10m.json`
- Modify: `JIT/tests/test_formal_config.py`
- Modify: `JIT/tests/test_contracts.py`

**Interfaces:**
- Consumes: `resolve_config_payload(payload) -> ResolvedConfig` and `PPOConfig.block_transitions`.
- Produces: v4 `ResolvedConfig.model["naccdmax"] == 256`, 406 aligned PPO blocks, the exact six-checkpoint schedule, and the approved reward/reset/event values.

- [x] **Step 1: Write failing exact-contract tests**

Add literal assertions for the new smoke/formal paths and exact values:

```python
V4_10M_CHECKPOINTS = (
    0, 491_520, 1_990_656, 4_988_928, 7_987_200, 9_977_856,
)

assert formal.ppo.requested_transitions == 9_977_856
assert formal.formal.formal_blocks == 406
assert formal.ppo.num_evals == 407
assert formal.events.jump_zone_x_max == pytest.approx(3.4)
assert formal.reward.height_coeff == pytest.approx(40.0)
assert formal.reset.airborne_rsi_probability == pytest.approx(0.08)
assert formal.model["naccdmax"] == 256
```

Add mutation cases for `naccdmax=48`, `jump_zone_x_max=3.1`,
`height_coeff=20.0`, and `airborne_rsi_probability=0.05`.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_formal_config.py JIT/tests/test_contracts.py -q
```

Expected: failures identify the missing `phase_u_continuation_10m.json` and old v4 values.

- [x] **Step 3: Implement the exact v4 replacement**

Change only the v4 branch of `_validate_formal` and
`_validate_approved_absolute_method`. Use fresh v4 seeds `820400` for smoke,
`820401` for formal, and held-out seeds `950001..950008`. Add `naccdmax` only to
the v4 expected-model dictionary; retain v1-v3 dictionaries unchanged.

- [x] **Step 4: Verify GREEN**

Run the focused command from Step 2. Expected: all selected tests pass.

### Task 2: Bind the real Warp CCD capacity

**Files:**
- Modify: `JIT/src/jit_dvgc/env.py`
- Modify: `JIT/tests/test_env_gpu.py`
- Modify: `JIT/tests/test_model.py`

**Interfaces:**
- Consumes: `ResolvedConfig.model["naccdmax"]`.
- Produces: every v4 reset creates MJX Warp data with aggregate CCD capacity 256.

- [x] **Step 1: Write a failing GPU capacity regression**

Construct the real v4 smoke environment and assert the observable runtime
capacity rather than source text:

```python
state = env.reset(jax.random.PRNGKey(0))
capacity = int(state.data._impl.naccdmax)
assert capacity == 256
```

Also add a boundary test proving `naccdmax > naconmax` is rejected by config
validation.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_model.py -q -m "not gpu"
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_env_gpu.py -q -m gpu
```

Expected: the real data reports the old implicit capacity 48 or configuration lacks the field.

- [x] **Step 3: Pass `naccdmax` to Warp data construction**

Add the explicit argument next to `naconmax` in `TwoPhaseBikeEnv.reset`:

```python
naccdmax=int(self._resolved_config.model["naccdmax"]),
```

Keep model conversion, collision geometry, and physics unchanged.

- [x] **Step 4: Verify GREEN**

Run both focused commands from Step 2. Expected: selected Host and GPU tests pass.

### Task 3: Lock event, reward, and RSI behavior

**Files:**
- Modify: `JIT/tests/test_semantics.py`
- Modify: `JIT/tests/test_rewards.py`
- Modify: `JIT/tests/test_env_gpu.py`

**Interfaces:**
- Consumes: existing `advance_events`, `phase_u_reward`, and `TwoPhaseBikeEnv.reset`.
- Produces: behavioral proof that the configuration changes affect the real semantics without changing their structure.

- [x] **Step 1: Add hand-derived behavior tests**

Use literal positions to prove signal behavior:

```python
assert signal_at_x_3_25 is True
assert signal_after_x_3_41 is False
assert signal_after_return_to_x_3_0 is False
```

For one fixed reward state above `jump_reward_min_height`, assert that changing
only `jump_signal` changes the height component from `0.0` to the exact
`40.0 * _height_raw` result. In a 4,096-reset GPU batch, assert the observed RSI
fraction lies within a fixed statistical tolerance around 0.08 and every RSI
sample starts with the jump signal enabled.

- [x] **Step 2: Verify RED against the old contract where applicable**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_semantics.py JIT/tests/test_rewards.py -q
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_env_gpu.py -q -m gpu
```

Expected: literals tied to 3.4/40/0.08 fail before Task 1 configuration is applied; after Task 1 they pass without extra production logic.

- [x] **Step 3: Refactor tests only if needed**

Keep expected arithmetic literal and avoid reproducing the production formula
through production helpers. Do not add a second reward path.

- [x] **Step 4: Verify GREEN**

Run both commands from Step 2. Expected: all selected tests pass.

### Task 4: Add one-shot terminal-run Codex analysis

**Files:**
- Create: `JIT/cli/watch_training_and_analyze.py`
- Create: `JIT/tests/test_watch_training_and_analyze.py`
- Modify: `JIT/README.md`

**Interfaces:**
- Consumes: `--run-dir`, `--pid-file`, `--launch-log`, `--poll-seconds`, terminal `status.json`, and a `codex` executable on `PATH`.
- Produces: `AUTO_ANALYSIS.md`, `codex_exec.log`, `codex_analysis.started.json`, and `codex_analysis.completed.json` under the exact ignored run directory.

- [x] **Step 1: Write failing CLI integration tests**

Run the CLI against a temporary completed run and a fake executable named
`codex` that records arguments and writes the requested `--output-last-message`
path. Assert:

```python
assert invocation_count == 1
assert "--sandbox" in argv and "read-only" in argv
assert auto_analysis.read_text()
assert completed_marker["returncode"] == 0
```

Run the CLI a second time and assert invocation count remains one. Add a
running-status test that changes `status.json` to completed after one short
poll, proving no Codex invocation occurs before terminal state.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_watch_training_and_analyze.py -q
```

Expected: import or CLI path fails because the watcher does not exist.

- [x] **Step 3: Implement the minimal watcher**

Use atomic `os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)`
for the started marker. Poll local
files without invoking Codex. At terminal state execute exactly:

```python
[
    codex, "exec", "--sandbox", "read-only", "--cd", str(repo_root),
    "--output-last-message", str(run_dir / "AUTO_ANALYSIS.md"), prompt,
]
```

The prompt binds the exact run path, prohibits source edits/training/resume,
requests provenance verification for completed runs, compares all natural/RSI
panels, summarizes curves and terminations, counts `CCD overflow` lines in the
declared launch log, and reports evidence paths. Capture stdout/stderr in
`codex_exec.log` and always write the completed marker with return code and UTC timestamp.

- [x] **Step 4: Verify GREEN**

Run the focused command from Step 2. Expected: all watcher tests pass without network or real model calls.

### Task 5: Integrate the 10M contract with formal evidence and preflight

**Files:**
- Modify: `JIT/tests/test_formal_training.py`
- Modify: `JIT/tests/test_provenance_verify.py`
- Modify: `JIT/scripts/local_preflight.sh`
- Modify: `JIT/docs/VERIFICATION.md`
- Modify: `JIT/planning/task_plan.md`
- Modify: `JIT/planning/findings.md`
- Modify: `JIT/planning/progress.md`

**Interfaces:**
- Consumes: active v4 config path and exact checkpoint schedule.
- Produces: dynamic formal orchestration/provenance tests and a preflight that loads the active 10M contract without launching it.

- [x] **Step 1: Update formal fixtures and write failure-first assertions**

Replace v4-only 4,988,928 literals with the six approved 10M checkpoints while
leaving v3 fixtures unchanged. Assert manifest fields remain:

```python
assert manifest["parent_checkpoint"] is None
assert manifest["starting_training_transition"] == 0
assert manifest["training_transition_ceiling"] == 9_977_856
```

Require provenance to reject a completed v4 run whose raw config reports
`naccdmax=48`, RSI 0.05, window max 3.1, or height coefficient 20.

- [x] **Step 2: Verify RED**

Run:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_formal_training.py JIT/tests/test_provenance_verify.py -q -m "not gpu"
```

Expected: current v4 fixtures and predeclared schedules disagree with the new approved contract.

- [x] **Step 3: Generalize only v4 test/evidence constants**

Keep production formal orchestration schedule-driven. Update preflight to load
`JIT/configs/phase_u_continuation_10m.json`, require 406 blocks, and never launch training.

- [x] **Step 4: Verify GREEN**

Run the focused command from Step 2. Expected: selected tests pass.

### Task 6: Full verification, one implementation commit, push, smoke, and launch

**Files:**
- Modify: `JIT/docs/plans/2026-08-25-jit-v4-10m-ccd-auto-analysis.md`
- Runtime only: `JIT/runs/phase_u/<run_id>/`, `<run_id>.launch.log`, `<run_id>.pid`, and `<run_id>.watcher.*`

**Interfaces:**
- Consumes: all preceding implementation and tests.
- Produces: one pushed source commit, one closed smoke, and one fresh persistent formal training plus watcher.

- [ ] **Step 1: Run static and complete JIT verification**

Re-run this gate after the final review corrections; the earlier successful
pass predates the completed-v4 fresh-start provenance fix.

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall -q JIT/src JIT/cli
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests -q -m "not gpu"
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests -q -m gpu
JIT/scripts/local_preflight.sh
```

Record exact pass counts and failures. Preserve the known user-owned root test conflict rather than editing outside JIT.

- [ ] **Step 2: Perform the final pre-training Git gate**

Run `git diff --check`, inspect `git status --short`, explicitly stage only the
validated JIT source/config/test/doc/script paths, inspect `git diff --cached`,
commit once, push normally, and confirm `git rev-parse HEAD` equals the remote
branch ref. Do not stage `JIT/runs`, logs, checkpoints, caches, or outside-JIT
paths. This gate precedes every new PPO interaction, including the bounded
smoke.

- [ ] **Step 3: Run one fresh bounded v4 GPU smoke**

Launch `phase_u_v4_ccd256_window34_height40_rsi8_smoke_24576_seed820400_20260825`
with `phase_u_continuation_smoke.json`, no restore option, and a run-local log.
Verify status, finite PPO metrics, transition accounting, checkpoint identity,
actual `naccdmax=256`, and count `CCD overflow` lines without treating an
occasional residual line as an automatic method failure.

- [ ] **Step 4: Launch the fresh formal process**

Use run id
`phase_u_v4_ccd256_window34_height40_rsi8_9977856_seed820401_20260825` and run:

```bash
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python \
  JIT/cli/train_phase_expert.py --phase propulsion_ascent \
  --config JIT/configs/phase_u_continuation_10m.json \
  --run-id phase_u_v4_ccd256_window34_height40_rsi8_9977856_seed820401_20260825 \
  --formal \
  > JIT/runs/phase_u/phase_u_v4_ccd256_window34_height40_rsi8_9977856_seed820401_20260825.launch.log \
  2>&1 < /dev/null &
```

Persist the PID. Do not pass `--restore-checkpoint`.

- [ ] **Step 5: Launch the one-shot watcher**

Start `JIT/cli/watch_training_and_analyze.py` in a second detached session bound
to the exact run directory, PID file, and launch log. Persist its PID/log beside
the training launcher. Verify both PIDs are live and no analysis marker exists
while training status is running.

- [ ] **Step 6: Perform one startup inspection and hand off**

Inspect GPU identity, backend manifest, transition-zero checkpoint identity,
fresh manifest fields, and startup status once. Report run ID, training PID,
watcher PID, GPU, effective transition target, and ETA. End the interactive
session without polling the training process.
