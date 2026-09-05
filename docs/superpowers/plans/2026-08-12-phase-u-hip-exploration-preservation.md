# Phase U Hip Exploration Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encode the single approved Phase U exploration hypothesis by reducing only the hip initial action standard deviation from 0.50 to 0.25 while preserving neutral early-airborne reward semantics.

**Architecture:** Keep the existing scalar-or-vector PPO runtime and Phase U reward adapter unchanged. Change only the stable smoke/formal JSON configuration values, strengthen stable-config tests, and use the existing pre-window reward regression as evidence that early airborne remains neither rewarded nor specially penalized.

**Tech Stack:** JSON configuration, Python 3.12, pytest, JAX/Brax runtime contract tests.

## Global Constraints

- Phase U action order remains `[steer, rear-wheel drive, hip, knee]`.
- The exact initial standard-deviation vector is `[0.05, 0.05, 0.25, 0.05]`.
- Early airborne is not terminal, not success, receives no pre-window task-progress reward, and receives no new early-airborne penalty.
- Do not modify reward code, reset, observation, event latch, optimizer, network, normalizer, horizon, XML, action mapping, force limits, snapshots, feasibility, or Tube contracts.
- Do not execute PPO without a fresh run-bound interaction authorization.

---

### Task 1: Stable Configuration Contract

**Files:**
- Modify: `tests/test_phase_expert_training.py`
- Modify: `configs/phase_expert_smoke.json`
- Modify: `configs/phase_expert_phase_u.json`

**Interfaces:**
- Consumes: `resolve_policy_initial_action_std(training_config) -> tuple[float, float, float, float]`.
- Produces: both stable configuration templates resolve to `(0.05, 0.05, 0.25, 0.05)`.

- [ ] **Step 1: Change only stable-config expectations to the approved vector**

Update the test that loads `configs/phase_expert_smoke.json` and
`configs/phase_expert_phase_u.json`:

```python
assert resolved == (0.05, 0.05, 0.25, 0.05)
assert module._jsonable(resolved) == [0.05, 0.05, 0.25, 0.05]
```

Do not change generic runtime-vector tests that intentionally use 0.50 as an
arbitrary valid per-channel value.

- [ ] **Step 2: Run the stable-config test and verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py::test_phase_u_configs_select_explicit_exploration_without_changing_default_runtime_prior
```

Expected: both current stable files resolve hip std 0.50, so the assertions
requiring 0.25 fail.

- [ ] **Step 3: Change only the two JSON vectors**

In each stable config, replace:

```json
"policy_initial_action_std": [0.05, 0.05, 0.5, 0.05]
```

with:

```json
"policy_initial_action_std": [0.05, 0.05, 0.25, 0.05]
```

- [ ] **Step 4: Run the stable-config and early-airborne reward tests**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py::test_phase_u_configs_select_explicit_exploration_without_changing_default_runtime_prior \
  tests/test_phase_expert_training.py::test_pre_window_takeoff_and_apex_progress_rewards_are_zero_even_when_airborne_and_rising \
  tests/test_prelaunch_continuation.py
```

Expected: PASS. The second and third targets prove the configuration-only
change did not alter early-airborne reward or termination semantics.

- [ ] **Step 5: Run affected suites**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py \
  tests/test_optional_runtime.py \
  tests/test_prelaunch_continuation.py
```

Expected: all pass.

### Task 2: Static Validation and State Marker

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: the validated stable configuration and the exhausted 995,200-transition program accounting.
- Produces: a recoverable marker that the 0.25 hypothesis is implemented but not dynamically tested or authorized for PPO.

- [ ] **Step 1: Update the experiment marker**

Record:

```text
implemented hypothesis: [0.05, 0.05, 0.25, 0.05]
reward/reset/observation changes: none
new training transitions: 0
smoke/formal PPO authorization: false
```

- [ ] **Step 2: Run complete static verification**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
```

Expected: compilation, full pytest, and preflight pass. Do not run runtime PPO
gate because this plan has no environment-interaction authorization.

- [ ] **Step 3: Commit focused implementation paths**

```bash
git add \
  configs/phase_expert_smoke.json \
  configs/phase_expert_phase_u.json \
  tests/test_phase_expert_training.py \
  docs/EXPERIMENT_STATE.md \
  docs/superpowers/plans/2026-08-12-phase-u-hip-exploration-preservation.md
git commit -m "config: preserve phase u timing exploration"
```

- [ ] **Step 4: Push the current branch**

```bash
git push origin agent/two-phase-soft-tube
```

Expected: local HEAD equals `origin/agent/two-phase-soft-tube`; `.vscode/`
remains untracked and untouched.
