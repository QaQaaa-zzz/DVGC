# Prelaunch Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow early liftoff to continue into the legal jump window without termination, while preserving window-gated reward, existing roll/pitch limits, other physical failures, and the complete Gate B audit boundary.

**Architecture:** Make the smallest semantic change in `OrangeBikeDVGC.step`: remove prelaunch airborne from hard failure/end-code selection and remove wheel support from the window-entry latch. Keep the legacy counter as telemetry. Update fixed failure-video assumptions into continuation/outcome audits, then rerun the authoritative Gate B builder and its timing-explicit round trip.

**Tech Stack:** Python 3.12, JAX/MJX-Warp, MuJoCo 3.6, pytest, Pillow, mediapy H.264.

## Global Constraints

- Work only in `/home/qy/DVGC` using `/home/qy/mujoco_playground/.venv/bin/python` directly.
- Do not modify XML, geometry, action mapping, observation layout, reset timing, matcher, virtual environment, or PPO algorithms.
- Keep `max_roll_deg=35.0` and `max_pitch_deg=75.0` unchanged.
- Preserve prohibited-contact, invalid-wheel, backward, platform-back-edge, takeoff-task, nonfinite, timeout, recovery, chain, and stage-entry terminal semantics.
- `prelaunch_airborne_count` remains snapshot-compatible diagnostic state but cannot terminate.
- Window-gated takeoff/ascent progress is zero before the legal jump window.
- Formal training transitions remain zero; stop after Gate B.
- Preserve `.vscode/` and all unrelated user changes.

---

### Task 1: Early-airborne continuation and window latch

**Files:**
- Create: `tests/test_prelaunch_continuation.py`
- Modify: `tests/test_stage_rewards.py`
- Modify: `dvgc/env.py:1580-1795`

**Interfaces:**
- Consumes: `run_guideline_event_trace(env, reference, geometry, thresholds, seed, maximum_control_ticks) -> dict[str, Any]`.
- Preserves: `state.info["prelaunch_airborne_count"]` and all snapshot fields.
- Produces: a jump signal that may latch inside the window without wheel support.

- [ ] **Step 1: Write the failing continuation test**

Create the standard deterministic Gate B runtime fixture, then run only the first 12 guideline control ticks:

```python
def test_early_airborne_diagnostic_does_not_terminate_guideline():
    env, reference, geometry, thresholds = _runtime()
    report = run_guideline_event_trace(
        env, reference, geometry, thresholds,
        seed=44_000, maximum_control_ticks=12,
    )
    assert report["environment_transitions"] == 12
    assert report["terminal"] is False
    assert report["end_code"] == END_NONE
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_prelaunch_continuation.py::test_early_airborne_diagnostic_does_not_terminate_guideline
```

Expected: FAIL because the current rollout terminates at tick 12 with `END_PRETAKEOFF_AIRBORNE`.

- [ ] **Step 3: Write the failing window-latch test**

Extend the identical fixed rollout beyond tick 12 and require that early loss of support does not block the deployable window event:

```python
def test_early_airborne_rollout_can_latch_jump_window_later():
    env, reference, geometry, thresholds = _runtime()
    report = run_guideline_event_trace(
        env, reference, geometry, thresholds,
        seed=44_000, maximum_control_ticks=30,
    )
    assert report["first_event_ticks"]["jump_window_entered"] >= 0
    assert report["end_code"] != END_PRETAKEOFF_AIRBORNE
```

- [ ] **Step 4: Run the window-latch test and verify RED**

Run the new test alone. Expected: FAIL because the current `takeoff_event` requires `wheel_any` and the rollout has already terminated.

- [ ] **Step 5: Add the reward-gating regression**

Add a pure reward test using a positive `vz` feature:

```python
def test_takeoff_ascent_progress_is_zero_before_window_and_positive_inside():
    cfg = default_config()
    feature = jp.zeros(16).at[8].set(cfg.takeoff_liftoff_vz)
    common = dict(
        cfg=cfg, objective="takeoff_to_ascent", feature=feature,
        previous_feature=feature, action=jp.zeros(4), previous_action=jp.zeros(4),
        next_entry=jp.asarray(False), hard_failure=jp.asarray(False),
        jump_latched=jp.asarray(True), joint_energy=jp.asarray(0.0),
    )
    outside = compute_stage_next_entry_reward(**common, window_active=jp.asarray(False))
    inside = compute_stage_next_entry_reward(**common, window_active=jp.asarray(True))
    assert float(outside["progress"]) == 0.0
    assert float(inside["progress"]) > 0.0
```

Run it and confirm it passes as a characterization of the existing reward boundary; no reward-function change is needed unless the environment integration test contradicts it.

Add a real environment regression that resets from the natural state with the
floating-base quaternion set by `_quat_from_euler_xyz` to 36 degrees roll or
76 degrees pitch, advances one zero-action tick, and requires respectively
`END_ROLL_LIMIT` or `END_PITCH_LIMIT`. This proves the approved hard limits,
not merely their source text.

- [ ] **Step 6: Implement the minimal environment change**

In `OrangeBikeDVGC.step`:

```python
takeoff_event = (
    (phase0 == STAGE_ID["approach"])
    & in_takeoff_window
    & (vx >= 0.90)
)

# Keep this diagnostic calculation and snapshot field unchanged.
prelaunch_airborne = jp.where(...)

hard_failure = (
    contact["prohibited"] | invalid_fail | roll_bad | pitch_bad | backward
    | back_edge | takeoff_task_failure | nonfinite
)
```

Remove the `END_PRETAKEOFF_AIRBORNE` assignment from active end-code selection. Do not renumber constants or remove the compatibility metric.

- [ ] **Step 7: Verify GREEN and retained hard limits**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_prelaunch_continuation.py tests/test_stage_rewards.py \
  tests/test_optional_runtime.py::test_nonfinite_action_is_explicit_finite_terminal_transition
```

The new parameterized roll/pitch environment regression and the existing
nonfinite regression must all pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add dvgc/env.py tests/test_prelaunch_continuation.py tests/test_stage_rewards.py
git commit -m "fix: continue early airborne rollouts"
```

---

### Task 2: Replace retired prelaunch-failure audit assumptions

**Files:**
- Modify: `dvgc/failure_video.py`
- Modify: `cli/render_two_phase_failures.py`
- Modify: `cli/build_two_phase_guideline_banks.py`
- Modify: `tests/test_failure_video.py`
- Modify: `tests/test_two_phase_guideline.py`

**Interfaces:**
- Produces named diagnostics `full_guideline_continuation` and `launch_history_window_latch`.
- Preserves MP4 plus `.states.npz`, telemetry, SHA-256, first-event, frame, transition, and action-schedule closure.
- Preserves the original Gate B `gate_pause` exception when a post-change event audit fails.

- [ ] **Step 1: Change tests to reject the retired terminal contract**

Replace assertions requiring scenario `full_guideline_prelaunch_airborne`, `end_code=9`, or failure reason `prelaunch_airborne` with:

```python
assert trace.summary["scenario"] == "full_guideline_continuation"
assert trace.summary["end_code"] != END_PRETAKEOFF_AIRBORNE
assert trace.summary["first_event_ticks"]["jump_window_entered"] >= 0
assert trace.summary["formal_training_transitions"] == 0
```

For the timing-explicit launch-history diagnostic, require initial action 73, actions 83/93/103, later jump-window latch, and no prelaunch terminal.

- [ ] **Step 2: Run failure-video and Gate B CLI tests and verify RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_failure_video.py tests/test_two_phase_guideline.py -x
```

Expected: FAIL because production scenarios and manifest validators still require the retired failure.

- [ ] **Step 3: Implement continuation/outcome audit scenarios**

Rename the scenario definitions without changing their fixed seeds or action timing:

```python
FAILURE_SCENARIOS = {
    "full_guideline_continuation": FailureScenario(...),
    "launch_history_window_latch": FailureScenario(...),
}
```

Capture through the fixed audit horizon or an actual terminal/recovery event. Record `audit_outcome`, `end_code`, event ticks, and terminal reason from the observed rollout. Manifest validation must forbid `END_PRETAKEOFF_AIRBORNE`, recompute NPZ trace digests and first-event ticks, and retain strict action/frame/transition accounting. Do not require a fabricated failure when the diagnostic continues successfully.

- [ ] **Step 4: Keep automatic archiving conditional on an actual Gate pause**

`cli.build_two_phase_guideline_banks` should render the two fixed contextual diagnostics only when `guideline_event_report.json` reports `gate_pause`. The report-derived Gate status remains authoritative; rendering cannot convert pause to pass or pass to pause.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_failure_video.py tests/test_two_phase_guideline.py \
  tests/test_two_phase_snapshot_roundtrip.py
```

Require all tests to pass with zero formal training transitions.

- [ ] **Step 6: Commit Task 2**

```bash
git add dvgc/failure_video.py cli/render_two_phase_failures.py \
  cli/build_two_phase_guideline_banks.py tests/test_failure_video.py \
  tests/test_two_phase_guideline.py
git commit -m "fix: audit post-prelaunch gate b outcomes"
```

---

### Task 3: Rebuild and validate Gate B

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Modify: `docs/RUNTIME_GATE.json` through the approved runtime-gate command.
- Create ignored evidence only under: `runs/two_phase/gate_b_20260803_prelaunch_continuation/`

**Interfaces:**
- Consumes the unchanged authoritative XML/config/reference and fixed seed namespace.
- Produces `guideline_event_report.json`; on pass also produces both v4 banks and `snapshot_roundtrip_report.json`.

- [ ] **Step 1: Run the authoritative Gate B builder once**

```bash
/home/qy/mujoco_playground/.venv/bin/python \
  -m cli.build_two_phase_guideline_banks \
  --config configs/default.json \
  --reference data/reference_jump.csv \
  --output runs/two_phase/gate_b_20260803_prelaunch_continuation \
  --seed 4100 \
  --perturbations nominal \
  --geometry-tolerance 2e-4 \
  --event-max-control-ticks 100
```

Do not retry with changed thresholds, offsets, seeds, XML, or actions.

- [ ] **Step 2: Audit the observed Gate B result**

Read the generated JSON and report exact first-event ticks, missing events,
Apex width, recovery hold, end code, physical failure/timeout, transition count,
bank counts, and round-trip status. If status is `gate_pause`, verify the new
MP4/NPZ hashes and inspect contact sheets. If status is `pass`, verify both
banks and every timing-explicit round-trip row before accepting Gate B.

- [ ] **Step 3: Run focused and full verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_prelaunch_continuation.py tests/test_stage_rewards.py \
  tests/test_failure_video.py tests/test_two_phase_guideline.py \
  tests/test_two_phase_snapshot_roundtrip.py
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
```

Run long GPU tests exclusively; discard and rerun any result contaminated by a concurrent CUDA process.

- [ ] **Step 4: Refresh the runtime fingerprint**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate \
  --work-dir runs/runtime_gate_prelaunch_continuation_20260803
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate --check-only
```

This is exactly 96 smoke transitions, not training evidence.

- [ ] **Step 5: Update and commit the evidence ledger**

Record branch, implementation commits, Gate B status, artifact paths/hashes,
event/round-trip outcomes, full pytest, preflight, runtime fingerprint, audit
environment transitions, runtime smoke transitions, formal training
transitions `0`, watchdog interlock, blockers, and the next separately
permitted action.

```bash
git add docs/EXPERIMENT_STATE.md docs/RUNTIME_GATE.json
git commit -m "docs: record prelaunch continuation gate b audit"
```

- [ ] **Step 6: Stop**

Do not start Gate C, experts, labeling, feasibility training, Soft Tube work,
PPO pilot, or formal training.
