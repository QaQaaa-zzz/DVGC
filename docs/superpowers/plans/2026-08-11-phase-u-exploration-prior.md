# Phase U Exploration-Prior Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Phase U enough initial stochastic action coverage to observe physically valid liftoff while preserving every approved task and physics contract.

**Architecture:** Parameterize the existing neutral tanh actor's initial standard deviation without changing its zero deterministic mode or its default 0.05 Landing behavior. Resolve an explicit 0.25 value from Phase U training configuration and pass the same value to PPO construction and checkpoint network metadata. Re-align the next formal run to the remaining cumulative expert-training budget.

**Tech Stack:** Python 3.12, JAX, Flax, Brax PPO, MuJoCo MJX Warp, pytest, JSON manifests.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Change one scientific hypothesis only: Phase-U initial action standard deviation 0.05 to 0.25.
- Keep reward, reset, optimizer, network layers, observation, horizon, XML, 4 kg payload, +/-50 N m limits, action mapping, event semantics, and fixed evaluation seeds unchanged.
- Keep the runtime/default actor prior at 0.05 for Landing, legacy, and unified callers.
- Keep cumulative formal Phase U expert training at or below 1,000,000 transitions; the new aligned ceiling is 448,000 and cumulative total is 995,200.
- Preserve failure videos and separate smoke, diagnostic, formal training, evaluation, candidate, and continuation interaction counts.

---

### Task 1: Parameterize the neutral actor prior

**Files:**
- Modify: `dvgc/runtime.py`
- Modify: `tests/test_optional_runtime.py`

**Interfaces:**
- Consumes: `POLICY_INITIAL_ACTION_STD = 0.05`.
- Produces: `make_dvgc_ppo_networks(..., initial_action_std: float = POLICY_INITIAL_ACTION_STD)`, `build_network_factory(*, initial_action_std: float = POLICY_INITIAL_ACTION_STD)`, and `make_ppo_train_fn(..., initial_action_std: float = POLICY_INITIAL_ACTION_STD)`.

- [x] **Step 1: Write failing distribution tests**

Add a test that constructs the real network with `initial_action_std=0.25` and
asserts literal zero location and literal 0.25 scale. Retain the existing test
that asserts the default scale is 0.05. Add invalid-value cases for 0.001,
1.0, NaN, and infinity.

- [x] **Step 2: Verify RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_optional_runtime.py::test_policy_network_accepts_phase_specific_initial_action_std
```

Expected: `TypeError` because the factory does not yet accept the keyword.

- [x] **Step 3: Implement the minimal runtime parameter**

Validate `0.001 < initial_action_std < 1.0`, use it to initialize the tanh
normal scale parameter, return a `functools.partial` from
`build_network_factory`, and pass the value through `make_ppo_train_fn`.

- [x] **Step 4: Verify GREEN**

Run the new tests and the existing PPO factory tests in
`tests/test_optional_runtime.py`.

### Task 2: Bind Phase U configuration, checkpoints, and remaining budget

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `configs/phase_expert_smoke.json`
- Modify: `configs/phase_expert_phase_u.json`
- Modify: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: `build_network_factory(initial_action_std=...)` and `make_ppo_train_fn(initial_action_std=...)`.
- Produces: `resolve_policy_initial_action_std(training_config) -> float`, consistent PPO/checkpoint factories, and a 448,000-transition formal budget.

- [x] **Step 1: Write failing configuration and forwarding tests**

Assert that the smoke and formal configs resolve to 0.25; missing, boolean,
nonfinite, 0.001, and 1.0 values are rejected. Capture the checkpoint network
factory and PPO factory arguments and assert both receive 0.25. Update formal
budget literals to 35 blocks, 448,000 training, 6,400 fixed evaluation,
454,400 combined, 51,200 candidate, 51,200 continuation, and 556,800 total
environment transitions.

- [x] **Step 2: Verify RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py
```

Expected: failure because the resolver and configuration field do not exist
and the stable formal budget still permits 998,400 transitions.

- [x] **Step 3: Implement the Phase U binding**

Resolve the explicit configuration value, pass it to both network construction
paths, add `policy_initial_action_std: 0.25` to smoke/formal configs, and set
the formal schedule to `[0, 100000, 250000, 448000]` with effective checkpoints
`[0, 102400, 256000, 448000]`, `num_evals=36`, and four fixed evaluations.

- [x] **Step 4: Verify GREEN**

Run `tests/test_phase_expert_training.py`, the relevant runtime tests, then the
full pytest suite.

### Task 3: Validate, commit, and run one bounded smoke

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored authorization and output under `runs/two_phase/`

**Interfaces:**
- Consumes: validated source/config/XML/threshold hashes.
- Produces: one 12,800-transition smoke with fixed evaluation, checkpoint, metrics, accounting, and failure videos.

- [x] **Step 1: Run repository verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate
```

The full runtime gate is required because `dvgc/runtime.py` changes.

- [x] **Step 2: Commit and push explicit source paths**

Commit the runtime, Phase U adapter, configurations, tests, plan, and experiment
state on `agent/two-phase-soft-tube`; push without force.

- [x] **Step 3: Authorize and run one smoke block**

Use a fresh run ID and seed, request 12,800 training transitions, and require
finite PPO update, checkpoint/network metadata, fixed evaluation, closed
accounting, and saved videos for every unsuccessful held-out rollout.

- [x] **Step 4: Gate the formal run**

Authorize formal training only if smoke has no numerical, checkpoint,
accounting, history, hash, broadphase, or rendering-contract failure. Smoke
liftoff or Apex success is not required.

### Task 4: Launch the remaining formal Phase U budget

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored authorization, output, log, and control record under `runs/two_phase/`

**Interfaces:**
- Consumes: smoke-qualified committed HEAD and stable 448,000-transition config.
- Produces: a detached, resumable formal run with fixed physical gates.

- [x] **Step 1: Run formal preflight and create a run-bound authorization**

Bind 448,000 training transitions, four fixed evaluations, source/config/XML
hashes, a fresh seed, cumulative Phase U accounting, and all existing Gate
Pause reasons.

- [x] **Step 2: Launch and check startup once**

Launch with `setsid`; once only, verify process liveness, `status=running`, the
absolute transition-0 checkpoint, and absence of broadphase/OOM/NaN/Inf/error
patterns.

- [x] **Step 3: Record and stop polling**

Commit/push the startup marker, preserve `.vscode/`, and leave the persistent
goal paused until completion or Gate Pause.
