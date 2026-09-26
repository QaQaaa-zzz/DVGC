# Phase U Window-Active Ascent Credit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Phase U bounded positive-vertical-velocity credit after legal jump-window entry while preserving every confirmed-liftoff, clearance, Apex, safety, and provenance contract.

**Architecture:** Keep the existing `phase_u_reward_components` interface and change only which existing event latch gates `ascent_progress`. Preserve confirmed liftoff as the gate for clearance and downstream events, bump the hashed reward semantics, and prove the boundary through direct JAX reward tests and the real adapter regression suite.

**Tech Stack:** Python, JAX, pytest, MuJoCo/MJX, Brax PPO, JSON run manifests.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Keep `assets/orange_bike_4kg_horizontal.xml` at the authorized 2 kg payload and unchanged SHA-256.
- Keep +/-50 N m hip/knee limits and action order `[steer, rear-wheel drive, hip, knee]`.
- Keep 256 parallel environments for smoke/formal runs.
- Change one scientific hypothesis only: the `ascent_progress` activation gate.
- Do not move the jump window or deadline, weaken safety, alter reset/observation/history, or change PPO hyperparameters.
- Do not resume a checkpoint from a different reward-contract hash.
- Do not declare `pi_up`, `V_up`, or a Soft Tube without the corresponding evidence.

---

### Task 1: Encode the reward-boundary regression

**Files:**
- Modify: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: `phase_u_reward_components(...) -> dict[str, Any]` and `PhaseURewardConfig`.
- Produces: executable contract tests for window-active ascent and retained downstream gates.

- [ ] **Step 1: Write failing tests**

Add direct reward tests using positive `com_vz` that assert:

```python
assert float(before_window["ascent_progress"]) == 0.0
assert float(in_window_before_liftoff["ascent_progress"]) > 0.0
assert float(in_window_before_liftoff["clearance_progress"]) == 0.0
assert float(in_window_before_liftoff["apex_approach"]) == 0.0
assert float(in_window_nonascending["ascent_progress"]) == 0.0
```

Retain assertions that early airborne alone does not set success or done.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py -k "window_active_ascent or early_airborne"
```

Expected: the in-window/pre-liftoff ascent assertion fails because the current implementation requires confirmed liftoff.

### Task 2: Implement the minimal gate change

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: existing monotonic `event.jump_window_entered` and confirmed `event.liftoff_seen`.
- Produces: `ascent_progress` gated by legal window; clearance and Apex retain their existing gates.

- [ ] **Step 1: Change only the ascent gate**

In `phase_u_reward_components`, change only the first argument to the
`jp.where` that defines `ascent`: use the existing local `window` boolean
instead of `airborne_progress`. Keep `airborne_progress` unchanged as the gate
for `clearance` and keep `apex_eligible` unchanged as the downstream Apex gate.
Do not change the function signature or infer any gate from reward outcomes.

- [ ] **Step 2: Bump hashed semantics**

Change:

```python
PHASE_U_REWARD_SEMANTICS = "phase_u.window_active_ascent_credit.v4"
```

Update exact hash expectations only after calculating the new canonical hash
through the production function.

- [ ] **Step 3: Verify GREEN**

Run the exact RED command, then:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py \
  tests/test_two_phase_semantics.py \
  tests/test_takeoff_reward.py
```

Expected: all tests pass and retained deadline/safety/event tests remain green.

### Task 3: Requalify static and runtime contracts

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Runtime output: `runs/two_phase/runtime_gate/<fresh_run_id>/`

**Interfaces:**
- Consumes: the new reward-contract hash and unchanged runtime/model identities.
- Produces: fresh static and managed runtime evidence for the new source tree.

- [ ] **Step 1: Run static and full tests**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
```

- [ ] **Step 2: Run a fresh managed runtime gate**

Create a run manifest with purpose, exact inputs/hashes, 96-transition
engineering ceiling, stopping condition, and output path. Run the existing
stable `python -m cli.runtime_gate` entrypoint and verify its update/resume and
timing-explicit snapshot contracts.

- [ ] **Step 3: Record and commit qualification**

Explicitly stage source, tests, design/plan, and experiment-state paths. Do not
stage `runs/` or `.vscode/`. Commit and push the focused change.

### Task 4: Smoke and conditional formal execution

**Files:**
- Runtime config: `runs/two_phase/configs/<fresh_smoke>.json`
- Authorization: `runs/two_phase/authorizations/<fresh_smoke>.json`
- Runtime output: `runs/two_phase/phase_experts/<fresh_smoke>/`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: qualified source/tree/model/threshold/reward identities.
- Produces: engineering smoke evidence and, only if valid, one exact run-bound formal authorization.

- [ ] **Step 1: Run one 256-environment 1-block smoke**

Use 6,400 PPO-training transitions. Verify finite compile/update, checkpoint,
recursive sidecar, fixed evaluation, closed accounting, media hashes, and
separate transition totals. Treat physical performance as diagnostic only.

- [ ] **Step 2: Decide authorization from integrity evidence**

Authorize only if there is no NaN/Inf, optimizer/runtime failure,
state/timing/history corruption, identity mismatch, severe action saturation,
or immediate reward hacking. Low one-block success alone does not block the
formal run.

- [ ] **Step 3: Launch one fresh formal run if authorized**

Use 256 environments, fresh initialization, at most 998,400 PPO-training
transitions, checkpoints `0/102400/256000/505600/755200/998400`, fixed held-out
evaluation, candidate/continuation gates, persistent status/metrics/checkpoint
artifacts, and a sparse local completion watcher. Do not poll full logs.

- [ ] **Step 4: Audit before the next hypothesis**

At completion or Gate Pause validate all sidecars, outcome accounting, media
hashes, physical metrics, parent diversity, stochastic event coverage, and
representative timing-aligned traces. Begin snapshot/continuation only after
the existing independent-parent gate is genuinely satisfied.
