# Phase U Channel-Specific Exploration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve safe exploration on steering, drive, and knee while giving Phase U hip actions enough initial stochastic coverage to discover liftoff.

**Architecture:** Extend the existing neutral tanh policy initializer from a scalar-only standard deviation to a validated scalar-or-vector contract. Resolve an explicit four-channel tuple from Phase U configuration and forward exactly the same tuple to PPO construction, checkpoint metadata, and run manifests. Keep all physical and optimization contracts fixed and use only the remaining aligned formal budget.

**Tech Stack:** Python 3.12, JAX, Flax, Brax PPO, MuJoCo MJX Warp, pytest, JSON manifests.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Change one scientific hypothesis only: Phase U initial action std becomes `[0.05, 0.05, 0.50, 0.05]` in action order `[steer, rear-wheel drive, hip, knee]`.
- Keep reward, reset, optimizer, network layers, observation, horizon, XML, 4 kg payload, +/-50 N m limits, action mapping, thresholds, and fixed evaluation seeds unchanged.
- Keep scalar `0.05` backward compatibility for Landing, legacy, and unified callers.
- Limit the next formal invocation to 192,000 expert-training transitions; program cumulative training remains 995,200.
- Preserve failure videos and separate smoke, formal training, evaluation, candidate, and continuation transition counts.

---

### Task 1: Add scalar-or-vector runtime initialization

**Files:**
- Modify: `tests/test_optional_runtime.py`
- Modify: `dvgc/runtime.py`

**Interfaces:**
- Consumes: `initial_action_std: float | Sequence[float]` and runtime `action_size`.
- Produces: an exact `(action_size,)` initial scale vector used by `make_dvgc_ppo_networks`, `build_network_factory`, and `make_ppo_train_fn`.

- [x] **Step 1: Write failing vector distribution tests**

Add real-network assertions that `initial_action_std=(0.05, 0.05, 0.5, 0.05)` creates zero location and that exact scale. Add rejection cases for an empty vector, length three, length five, nested values, boolean values, NaN, infinity, `0.001`, and `1.0`.

- [x] **Step 2: Run RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_optional_runtime.py -k 'initial_action_std or custom_initial_std'
```

Expected: vector construction fails because the current implementation coerces the tuple to one float.

- [x] **Step 3: Implement minimal scalar-or-vector normalization**

Add one private normalizer that broadcasts a scalar to `action_size`, validates a one-dimensional vector of exact length, validates every value against `0.001 < value < 1.0`, and returns an immutable tuple. Initialize the policy scale with the normalized vector while keeping the location exactly zero.

- [x] **Step 4: Run GREEN**

Run the command from Step 2 and then all of `tests/test_optional_runtime.py`.

- [x] **Step 5: Commit the runtime unit**

Explicitly stage only `dvgc/runtime.py` and `tests/test_optional_runtime.py` and commit `feat: support channel-specific policy exploration`.

### Task 2: Bind Phase U config, manifest, checkpoint, and remaining budget

**Files:**
- Modify: `tests/test_phase_expert_training.py`
- Modify: `dvgc/phase_expert_training.py`
- Modify: `configs/phase_expert_smoke.json`
- Modify: `configs/phase_expert_phase_u.json`

**Interfaces:**
- Consumes: `policy_initial_action_std` as a JSON number or flat list.
- Produces: `resolve_policy_initial_action_std(...) -> tuple[float, ...]`, exact forwarding to PPO/checkpoint factories, JSON manifest vector, and a 192,000-transition formal ceiling.

- [x] **Step 1: Write failing config and forwarding tests**

Assert that stable Phase U configs resolve to `(0.05, 0.05, 0.5, 0.05)`, invalid/missing shapes are rejected, PPO and checkpoint factories receive that tuple, and the manifest records the ordered list. Update budget assertions to 15 rollout blocks, 192,000 training, 4,800 fixed evaluation, 196,800 combined, 38,400 candidate, 38,400 continuation, and 273,600 total environment transitions.

- [x] **Step 2: Run RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py
```

Expected: tuple resolution and the reduced stable budget assertions fail against the scalar-0.25 implementation.

- [x] **Step 3: Implement the Phase U binding**

Resolve exactly four action-channel values, pass the tuple unchanged to both network construction paths, serialize it as a list in the run manifest, change both stable configs to `[0.05, 0.05, 0.5, 0.05]`, and set the formal schedule to `[0, 100000, 192000]` with `num_evals=16` and three checkpoint evaluations.

- [x] **Step 4: Run GREEN**

Run `tests/test_phase_expert_training.py` and the complete optional runtime test file.

- [x] **Step 5: Commit the Phase U binding**

Explicitly stage the two configs, adapter, and test, then commit `feat: focus phase u exploration on hip control`.

### Task 3: Verify source and qualify one 512-environment smoke

**Files:**
- Modify: `docs/RUNTIME_GATE.json`
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored run config, authorization, log, status, checkpoint, metrics, and videos under `runs/two_phase/`.

**Interfaces:**
- Consumes: committed source/config/XML/threshold hashes.
- Produces: complete repository evidence and one 12,800-transition engineering smoke.

- [x] **Step 1: Run fresh source verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate \
  --work-dir runs/two_phase/runtime_gate/phase_u_channel_std_20260812
```

- [x] **Step 2: Commit and push verified source**

Explicitly stage `docs/RUNTIME_GATE.json` and any verified documentation change, commit, and push `agent/two-phase-soft-tube` without force.

- [x] **Step 3: Preflight and authorize one 512-environment block**

Use a fresh run ID and seed, bind 12,800 training transitions plus the existing smoke evaluation ceilings, record purpose/inputs/stopping condition/output, and require source/config/XML/threshold hashes to match.

- [x] **Step 4: Execute and inspect smoke once**

Require a finite update, policy mean std vector matching the configured channel order, checkpoint, closed accounting, eight held-out results, and videos for every failure. Reject broadphase overflow, NaN/Inf, OOM, traceback, timing/history mismatch, or hash drift. Liftoff/Apex success is not required from one block.

### Task 4: Launch the remaining formal budget and stop polling

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored run authorization, output, log, and control evidence under `runs/two_phase/`.

**Interfaces:**
- Consumes: smoke-qualified committed HEAD and `configs/phase_expert_phase_u.json`.
- Produces: one persistent 192,000-transition Phase U run with fixed checkpoints at effective 0/102,400/192,000.

- [x] **Step 1: Create run-bound formal authorization**

Bind 192,000 training, 4,800 fixed evaluation, 38,400 candidate acquisition, 38,400 continuation diagnostics, source/config/XML/threshold hashes, fresh seed, and program cumulative range 803,200 to 995,200.

- [x] **Step 2: Launch persistently and check startup once**

Use `setsid`; verify only process liveness, `status=running`, transition-0 checkpoint, and absence of broadphase/OOM/NaN/Inf/error patterns.

- [x] **Step 3: Record, commit, push, and stop polling**

Record run ID, PID, status/metrics/log paths, budgets, checkpoints, resume contract, and current formal `V_up`/Tube status. Preserve `.vscode/`; do not inspect the process again until completion or Gate Pause.
