# Phase U Confirmed-Airborne Liftoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent one-wheel or momentary support loss from being rewarded as legal Phase U liftoff.

**Architecture:** Keep the external pure-JAX event adapter and its monotonic latches. Replace support-loss liftoff admission with the existing temporally confirmed deployable `ApexBandSignals.stable_airborne` signal; all downstream rewards continue to consume the event latch.

**Tech Stack:** Python 3.12, JAX, MJX/Brax, pytest.

## Global Constraints

- Change one experimental hypothesis only: legal liftoff admission.
- Do not change reward weights, XML, action mapping, thresholds, reset, observation, PPO, network, optimizer, horizon, or safety failures.
- Early or one-wheel airborne remains nonterminal and unpenalized.
- Do not resume or relabel the completed 998,400-transition run.
- A fresh smoke and run-bound authorization are mandatory before more formal PPO.

---

### Task 1: Red-green confirmed-airborne event contract

**Files:**
- Modify: `tests/test_two_phase_runtime.py`
- Modify: `tests/test_phase_expert_training.py`
- Modify: `dvgc/two_phase_runtime.py`

**Interfaces:**
- Consumes: `ApexBandSignals.stable_airborne`, `TwoPhaseEventState.jump_window_entered`
- Produces: monotonic `TwoPhaseEventState.liftoff_seen`

- [ ] **Step 1: Add failing runtime tests**

Add a post-window transition with `stable_wheel_support=False` and
`stable_airborne=False`; assert `liftoff_seen=False`.  Then provide
`stable_airborne=True`; assert liftoff latches and remains true after the
signal falls.

- [ ] **Step 2: Add a failing adapter reward regression**

Use the existing fake adapter to show that one-wheel/support-loss telemetry
without confirmed airborne earns zero `legal_liftoff_bonus`,
`ascent_progress`, `clearance_progress`, and `apex_approach`.

- [ ] **Step 3: Verify RED**

Run the named tests and confirm they fail because support loss currently sets
`liftoff_seen`.

- [ ] **Step 4: Implement the minimum event predicate**

In `advance_two_phase_events`, require
`previous.jump_window_entered & apex.stable_airborne & no_failure` for new
liftoff admission.  Preserve the monotonic OR with `previous.liftoff_seen`.

- [ ] **Step 5: Update event-order fixtures and verify GREEN**

Make the first-occurrence sequence explicitly provide confirmed airborne
before liftoff, then a following tick for stable-airborne and ascending
admission.  Run both affected test files.

### Task 2: Bind identity and fully requalify

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `tests/test_phase_expert_training.py`
- Modify: `docs/EXPERIMENT_STATE.md`
- Modify: `docs/RUNTIME_GATE.json`

**Interfaces:**
- Produces: a new reward/event contract identity incompatible with shortcut checkpoints

- [ ] **Step 1: Add a failing identity regression**

Assert that the Phase U contract identity includes a stable semantic version
for confirmed-airborne liftoff.

- [ ] **Step 2: Verify RED, add the version, and verify GREEN**

Add the semantic version to the hashed reward/event contract payload without
changing numeric reward weights.

- [ ] **Step 3: Run complete verification**

Run targeted tests, compileall, full pytest, `scripts/local_preflight.sh`, and
a fresh managed GPU runtime gate because the runtime fingerprint changes.

- [ ] **Step 4: Commit and push the validated source round**

Stage only source, tests, design, plan, runtime-gate record, and experiment
state.  Preserve ignored run evidence and user `.vscode/` files.

### Task 3: Dynamic qualification and fresh formal retry

**Files:**
- Create ignored run-local smoke/formal configs and authorizations under `runs/two_phase/`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Produces: one 6,400-transition smoke, then at most one new 998,400-transition run

- [ ] **Step 1: Preflight and run the exact 256-environment formal-path smoke**
- [ ] **Step 2: Audit checkpoint identity, accounting, numerical state, physical outcomes, and videos**
- [ ] **Step 3: If smoke passes, create one exact run-bound formal authorization**
- [ ] **Step 4: Launch persistent training with a detached exit marker and sparse supervision**
- [ ] **Step 5: At completion, audit Apex parent coverage before any snapshot or continuation work**
