# JIT Apex Continuation and Training Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct normal wheel-ground contact handling, continue Phase U beyond Apex, save independently verifiable pre/post-Apex evidence, add step-aligned training curves, and launch one fresh 4,988,928-transition v4 run.

**Architecture:** Introduce an exact v4 config identity while retaining v1-v3 artifact verification. Keep wheel penetration as telemetry rather than failure, make Apex a monotonic nonterminal event, split saved traces at the first Apex boundary, and route Brax's asynchronous episode logger separately from ordered checkpoint progress before producing hashed training plots.

**Tech Stack:** Python 3.12, JAX 0.6.2, MuJoCo 3.6.0, MJX Warp, MuJoCo Playground, Brax 0.14.2 PPO, NumPy NPZ, Matplotlib, ImageIO/FFmpeg, pytest, JSON/JSONL, Git.

## Global Constraints

- Work only in `/home/qy/DVGC`; all task-generated content stays under `JIT/`.
- Use `/home/qy/mujoco_playground/.venv/bin/python` without modifying that environment.
- Do not edit the authoritative XML, meshes, collision geometry, solver, friction, control timing, observation dimensions, action order, or joint mapping.
- Preserve unrelated user changes outside `JIT/`.
- The new formal run starts from random initialization and must not load any previous checkpoint.
- Create one complete JIT-only implementation commit after validation and push it before training.
- Never commit `JIT/runs/`, checkpoints, logs, videos, images, NPZ data, or PID files.

---

### Task 1: Correct wheel-contact and Apex semantics

**Files:**
- Modify: `JIT/tests/test_geometry.py`
- Modify: `JIT/tests/test_semantics.py`
- Modify: `JIT/tests/test_rewards.py`
- Modify: `JIT/tests/test_env_gpu.py`
- Modify: `JIT/src/jit_dvgc/geometry.py`
- Modify: `JIT/src/jit_dvgc/semantics.py`
- Modify: `JIT/src/jit_dvgc/env.py`

**Interfaces:**
- Consumes: analytic collision bounds, `EventState`, `TerminalInputs`, and existing reward inputs.
- Produces: diagnostic wheel clearances/penetration, nonterminal first-Apex event, and terminal states owned only by retained physical failures or horizon.

- [x] **Step 1: Write failing wheel-contact regression tests**

Require the authoritative keyframe and a Host-confirmed `floor/rearwheel_collision`
state with `-0.014175 m` distance to remain nonterminal and receive no illegal-
contact reward. Require prohibited body contact to remain a physical failure.

- [x] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_geometry.py JIT/tests/test_semantics.py JIT/tests/test_rewards.py -q
```

Expected: failures because wheel penetration currently maps to terminal and
penalty inputs.

- [x] **Step 3: Write failing nonterminal-Apex tests**

Assert first Apex sets `event.apex_seen`, pays the bonus once, leaves
`terminated=false`, and a later retained body/pitch/roll failure still
terminates. Assert horizon after Apex remains a truncation and Apex is counted
from the event rather than `terminal.success`.

- [x] **Step 4: Implement the minimum semantic correction**

Replace the misleading boolean wheel failure with raw wheel clearance and
penetration telemetry. Remove it from `TerminalInputs` and reward
`illegal_contact`. Make terminal classification ignore Apex while leaving
`first_apex` available to reward and metrics.

- [x] **Step 5: Run Host and GPU tests and verify GREEN**

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_geometry.py JIT/tests/test_semantics.py JIT/tests/test_rewards.py \
  JIT/tests/test_env_host.py -q
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_env_gpu.py -q -m gpu
```

### Task 2: Split and summarize Apex-continuation traces

**Files:**
- Modify: `JIT/tests/test_evaluation.py`
- Modify: `JIT/tests/test_diagnostics.py`
- Modify: `JIT/tests/test_video.py`
- Modify: `JIT/src/jit_dvgc/evaluation.py`
- Modify: `JIT/src/jit_dvgc/diagnostics.py`
- Modify: `JIT/src/jit_dvgc/video.py`

**Interfaces:**
- Produces: `ApexTraceSplit`, `split_trace_at_apex(trace)`, segment masks and files, post-Apex panel metrics, and dashboards/videos with an Apex boundary.

- [x] **Step 1: Write failing trace-split tests**

Create synthetic no-Apex and first-Apex-at-index-2 traces. Require shared Apex
boundary state, transition conservation, empty shape-valid post segment when
absent, and separate NPZ hashes/metadata.

- [x] **Step 2: Run focused tests and verify RED**

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_evaluation.py JIT/tests/test_diagnostics.py JIT/tests/test_video.py -q
```

- [x] **Step 3: Implement split data and summary metrics**

Add the boundary index/time and masks to diagnostic arrays; save
`representative_pre_apex.npz` and `representative_post_apex.npz`; add post-Apex
transitions/failure/horizon rates to `summarize_phase_u`.

- [x] **Step 4: Mark Apex in PNG and video telemetry**

Draw a vertical boundary on time-series panels and display pre/Apex/post state
in the synchronized telemetry panel without stepping the environment during
rendering.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the Task 2 command and require exact frame/sample/segment counts.

### Task 3: Record and plot training episode metrics

**Files:**
- Create: `JIT/src/jit_dvgc/training_curves.py`
- Create: `JIT/tests/test_training_curves.py`
- Modify: `JIT/tests/test_formal_training.py`
- Modify: `JIT/src/jit_dvgc/formal_training.py`

**Interfaces:**
- Produces: `FormalRunController.on_episode_progress(step, metrics)`, a progress router, `episode_metrics.jsonl`, and `save_training_curves(run_dir) -> TrainingCurveReport`.

- [x] **Step 1: Write failing callback-order tests**

Call episode progress before policy parameters and require it to append a
monotonic absolute-step episode row without touching ordered PPO progress.
Require block PPO progress to retain its current callback ordering checks.

- [x] **Step 2: Write failing plot/artifact tests**

Provide synthetic PPO and episode JSONL rows and require finite aligned NPZ
series for reward, length, KL, policy/value/total loss, policy std, and SPS,
plus a decodable PNG and hashed JSON report.

- [x] **Step 3: Run focused tests and verify RED**

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_formal_training.py JIT/tests/test_training_curves.py -q
```

- [x] **Step 4: Implement the separated episode callback**

Enable Brax `log_training_metrics=True` and set
`training_metrics_steps=block_transitions`. Route mappings whose keys start
with `episode/` to `on_episode_progress`; route checkpoint-aligned PPO mappings
to `on_progress`. Add the segment starting offset for warm recovery while the
initial v4 run remains fresh.

- [x] **Step 5: Implement and integrate training curves**

Generate PNG/NPZ/JSON after the exact target and before closing the run. Reject
missing required episode reward/length or PPO series rather than creating an
empty graph.

- [x] **Step 6: Run focused tests and verify GREEN**

Run the Task 3 command and require callback-order safety plus decoded plot
evidence.

### Task 4: Exact v4 config, formal artifacts, and provenance

**Files:**
- Create: `JIT/configs/phase_u_continuation_smoke.json`
- Create: `JIT/configs/phase_u_continuation_5m.json`
- Modify: `JIT/tests/test_formal_config.py`
- Modify: `JIT/tests/test_formal_provenance.py`
- Modify: `JIT/src/jit_dvgc/config.py`
- Modify: `JIT/src/jit_dvgc/formal_training.py`
- Modify: `JIT/src/jit_dvgc/provenance.py`
- Modify: `JIT/scripts/local_preflight.sh`

**Interfaces:**
- Produces: exact `jit_phase_u_*_v4` contracts, final segment/curve artifact reports, and strict completed-v4 verification while retaining v1-v3 verification.

- [x] **Step 1: Write failing exact-v4 and mutation tests**

Require target 4,988,928, seed 820301, held-out 940001..940008, existing PPO
layout, and rejection of drift in model/action/reset/event/reward/PPO/formal
values.

- [x] **Step 2: Write failing artifact-forgery tests**

Require verifier rejection for swapped pre/post files, missing Apex boundary,
nonconserved transition counts, missing reward/length curves, altered hashes,
wrong extensions, nonmonotonic training steps, or reused v3 checkpoint identity.

- [x] **Step 3: Implement exact config and formal artifact writing**

Use a new canonical config hash and unique run identity. Write segment and
training-curve reports into the final natural/RSI and run-root evidence.

- [x] **Step 4: Extend provenance without weakening retained versions**

Dispatch v4-only segment/curve verification after exact config validation.
Continue verifying retained completed v1-v3 runs with their historical rules.

- [x] **Step 5: Run focused tests and verify GREEN**

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests/test_formal_config.py JIT/tests/test_formal_training.py \
  JIT/tests/test_formal_provenance.py JIT/tests/test_training_curves.py -q
```

### Task 5: Documentation, full verification, one commit, and fresh launch

**Files:**
- Modify: `JIT/README.md`
- Modify: `JIT/docs/VERIFICATION.md`
- Modify: `JIT/planning/task_plan.md`
- Modify: `JIT/planning/findings.md`
- Modify: `JIT/planning/progress.md`

- [x] **Step 1: Document corrected semantics and exact launch**

Record that wheel contact is normal telemetry, Apex is nonterminal, training
curves are rolling completed-episode means, and the new run is fresh.

- [x] **Step 2: Run complete verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall -q JIT/src JIT/cli
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests -q -m "not gpu"
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest \
  JIT/tests -q -m gpu
bash JIT/scripts/local_preflight.sh
```

- [x] **Step 3: Review and resolve all Critical/Important findings**

Audit the entire JIT diff, old evidence compatibility, fresh-start boundary,
segment/curve provenance, and staged path list. Rerun affected tests after any
fix.

Startup audit additionally requires a real-wrapper regression showing that a
done episode resets JIT counters/events and that the next episode continues
beyond one tick. Bind `training_wrapper.full_reset=true` into the v4 config;
abort and prohibit any checkpoint produced before this contract.

- [ ] **Step 4: Create and push one explicit JIT-only commit**

Explicitly stage only the modified JIT source/config/test/script/docs/planning
paths. Require `git diff --cached --check`, no `JIT/runs/` path, no outside-JIT
path, and remote ref equality after push.

- [ ] **Step 5: Launch a fresh persistent v4 run**

Use run ID `phase_u_continuation_4988928_seed820301_20260825_retry1` and deliberately
omit `--restore-checkpoint`:

```bash
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_continuation_5m.json \
  --run-id phase_u_continuation_4988928_seed820301_20260825_retry1 --formal \
  > JIT/runs/phase_u/phase_u_continuation_4988928_seed820301_20260825_retry1.launch.log 2>&1 \
  < /dev/null &
```

- [ ] **Step 6: Inspect startup once and hand off**

Confirm GPU backend, running status, transition-zero checkpoint, null parent,
zero start, fresh semantics, and no restore argument. Report PID, run path,
log path, and an approximately 3-6 minute completion estimate. Do not
continuously supervise the run.
