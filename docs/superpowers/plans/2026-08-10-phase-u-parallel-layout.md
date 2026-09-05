# Phase U Parallel Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the slow 64-environment formal Phase U layout with the highest tested collision-valid layout and launch one recoverable rerun.

**Architecture:** Keep the approved Phase U environment, reward, optimizer, and evaluation contracts unchanged. Change only the stable PPO batch layout to 512 environments, align the authorized budget and checkpoints to its 12,800-transition block, validate the whole repository, then launch one run-bound detached process with a single startup check.

**Tech Stack:** Python 3.11, pytest, Brax PPO, JAX/MJX Warp, JSON run manifests, Git.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Do not change XML, collision capacity, reward, reset, network, optimizer, episode horizon, action mapping, or physical limits.
- Reject 1,024 environments because its benchmark overflowed broadphase capacity.
- Keep formal training at or below 1,000,000 total environment transitions.
- Do not poll the detached process after the one startup health check.

---

### Task 1: Lock the aligned 512-environment budget

**Files:**
- Modify: `tests/test_phase_expert_training.py`
- Modify: `configs/phase_expert_phase_u.json`

**Interfaces:**
- Consumes: `validate_phase_expert_run_spec()` and the stable Phase U JSON schema.
- Produces: a 12,800-transition PPO block and a 998,400-transition formal ceiling.

- [x] **Step 1: Write the failing budget test**

Change the formal-config test to request 998,400 transitions and assert the
literal layout, block count 78, fixed-evaluation ceiling 9,600, combined
ceiling 1,008,000, candidate and continuation ceilings 76,800 each, and total
ceiling 1,161,600.

- [x] **Step 2: Verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py::test_phase_u_training_level_uses_aligned_cap_six_fixed_evaluations_and_no_hidden_brax_eval
```

Expected: failure because the stable config still declares 64 environments
and a 1,600-transition block.

- [x] **Step 3: Apply the minimal stable-config change**

Set `num_parallel_envs=512`, `num_minibatches=32`, `num_evals=79`,
`training_seed_count=512`, `checkpoint_cadence_transitions=12800`, maximum
training transitions 998,400, and the requested final checkpoint 998,400.
Add the already-implemented three-window plateau rule to the declared stopping
conditions.

- [x] **Step 4: Verify GREEN**

Run the named test, then all of `tests/test_phase_expert_training.py`.

### Task 2: Verify and commit the stable layout

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Retain ignored audit: `runs/two_phase/phase_experts/gate_c1_phase_u_env512_benchmark_20260810_seed710003/benchmark_audit.json`

**Interfaces:**
- Consumes: benchmark log and stable configuration.
- Produces: repository evidence and a focused commit.

- [x] **Step 1: Record both benchmark decisions**

Document the rejected 1,024 layout and accepted 512 layout with throughput,
interaction accounting, memory, and broadphase evidence.

- [x] **Step 2: Run fresh verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate --check-only
```

- [x] **Step 3: Commit and push explicit paths**

Stage only the stable config, its test, the design/plan, and experiment-state
document; push `agent/two-phase-soft-tube` without force.

### Task 3: Launch the recoverable formal rerun

**Files:**
- Create ignored run authorization under `runs/two_phase/authorizations/`
- Create ignored output under `runs/two_phase/phase_experts/`
- Create ignored persistent log/control record under `runs/two_phase/process_logs/`

**Interfaces:**
- Consumes: committed HEAD, clean tracked tree, stable 512 config, seed 710004.
- Produces: detached Phase U process, `status.json`, `metrics.jsonl`, checkpoints, and resume command.

- [x] **Step 1: Run formal preflight**

Request exactly 998,400 training transitions and confirm all separate
interaction ceilings.

- [x] **Step 2: Write a run-bound authorization**

Bind run ID, source tree, config/XML hashes, seed, budget, stopping conditions,
and a previously absent output directory.

- [x] **Step 3: Launch once and perform one startup check**

Start the stable CLI with `setsid`, redirect stdout/stderr to the persistent
log, then check liveness, running status, transition-0 checkpoint, and fatal
warning patterns once.

- [x] **Step 4: Stop interactive observation**

Record PID, paths, and resume command. Do not poll again; rely on the persistent
goal to resume on completion or Gate Pause.
