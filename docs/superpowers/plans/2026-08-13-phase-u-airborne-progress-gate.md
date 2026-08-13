# Phase U Airborne Progress Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent a wheel-supported Phase U policy from collecting airborne task shaping after merely entering the jump window.

**Architecture:** Reuse the existing monotonic `TwoPhaseEventState.liftoff_seen` latch. Pass a derived `airborne_progress_enabled` boolean into the reward function and gate ascent, clearance, and Apex approach with it.

**Tech Stack:** Python 3.12, JAX, MJX/Brax, pytest.

## Global Constraints

- Change one experimental hypothesis only: airborne progress gating.
- Do not modify reward weights, reset, thresholds, XML, action mapping, PPO layout, network, optimizer, or episode horizon.
- Preserve early-airborne nontermination and all physical safety failures.
- Do not resume the paused run; smoke and formal retries require new run-bound authorizations.

---

### Task 1: Red-green reward gate

**Files:**
- Modify: `tests/test_phase_expert_training.py`
- Modify: `dvgc/phase_expert_training.py`

**Interfaces:**
- Consumes: `TwoPhaseEventState.jump_window_entered`, `TwoPhaseEventState.liftoff_seen`
- Produces: `phase_u_reward_components(..., airborne_progress_enabled, ...)`

- [ ] **Step 1: Write failing tests**

Add direct component and adapter tests proving window-only and early-airborne
states receive zero ascent/clearance/Apex shaping, while post-window legal
liftoff enables it.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py -k 'airborne_progress'
```

Expected: assertion failure because window entry alone currently enables
ascent and clearance.

- [ ] **Step 3: Implement the minimum gate**

Add the explicit boolean argument to `phase_u_reward_components`; use it for
ascent, clearance, and Apex approach. In `PhaseExpertEnvAdapter.step`, derive
it as:

```python
airborne_progress_enabled = (
    jp.asarray(event.jump_window_entered) & jp.asarray(event.liftoff_seen)
)
```

- [ ] **Step 4: Verify GREEN and regressions**

Run the focused tests, the entire phase-expert test file, full pytest, static
compile, and `scripts/local_preflight.sh`.

- [ ] **Step 5: Commit the validated source round**

Explicitly stage only the design, plan, implementation, tests, runtime gate,
and experiment-state record.

### Task 2: Dynamic qualification and fresh formal run

**Files:**
- Update: `docs/EXPERIMENT_STATE.md`
- Create under ignored `runs/two_phase/`: smoke/formal config, authorization,
  status, checkpoints, metrics, and videos

**Interfaces:**
- Consumes: validated reward-contract hash and current source hashes
- Produces: one bounded smoke and, only after it passes, one new 256-env formal run

- [ ] **Step 1: Re-run the managed GPU runtime gate if its source fingerprint changes**
- [ ] **Step 2: Preflight and run one 6,400-transition formal-path smoke**
- [ ] **Step 3: Audit checkpoint identity, accounting, numerical state, and videos**
- [ ] **Step 4: If smoke passes, create a new exact run-bound authorization**
- [ ] **Step 5: Launch persistent training and use a local PID watcher without model polling**
- [ ] **Step 6: At completion/Gate Pause, audit physical metrics and advance only if Apex parent coverage exists**
