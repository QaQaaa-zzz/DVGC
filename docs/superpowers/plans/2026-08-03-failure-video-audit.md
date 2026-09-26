# Dynamic Failure Video Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce, render, and automatically archive the two authoritative Gate B failures as annotated MP4 audit artifacts.

**Architecture:** `dvgc.failure_video` owns exact scenario capture, host-only frame rendering, and manifest hashing. `cli.render_two_phase_failures` is the stable manual entrypoint; `cli.build_two_phase_guideline_banks` invokes it only after its existing physical event gate fails.

**Tech Stack:** Python 3.12, JAX/MJX-Warp, MuJoCo 3.6, Pillow, mediapy H.264, pytest.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Do not modify environment dynamics, XML, reward, termination, observations, action mapping, matcher, or virtual environment.
- Rendering consumes captured states and cannot influence physical event results.
- Formal training transitions remain 0.

---

### Task 1: Exact failure trace capture

**Files:**
- Create: `dvgc/failure_video.py`
- Test: `tests/test_failure_video.py`

**Interfaces:**
- Produces: `capture_failure_scenario(env, reference, geometry, thresholds, scenario, seed) -> FailureTrace`
- Produces: `FailureTrace.frames`, `FailureTrace.telemetry`, and `FailureTrace.summary`.

- [ ] Write tests asserting scenario start indices `0` and `83`, deterministic action indices, full-trace `end_code=9`, launch-history lost support before/inside the window, and zero formal training transitions.
- [ ] Run the tests and verify RED because `dvgc.failure_video` does not exist.
- [ ] Implement immutable scenario definitions, exact grounded reset, external two-phase event advancement, and host-materialized state capture.
- [ ] Run the focused capture tests and verify GREEN.
- [ ] Commit `feat: capture reproducible two-phase failure traces`.

### Task 2: Annotated MP4 rendering and manifest

**Files:**
- Modify: `dvgc/failure_video.py`
- Create: `cli/render_two_phase_failures.py`
- Modify: `tests/test_failure_video.py`

**Interfaces:**
- Produces: `render_failure_trace(env, trace, output_path) -> dict`.
- Produces: `render_failure_archive(config_path, reference_path, output_dir, seed) -> dict`.

- [ ] Write tests requiring two nonempty MP4 files, exact telemetry fields, and SHA-256 values matching file bytes; monkeypatch `env.step` during rendering to prove rendering advances no dynamics.
- [ ] Run the tests and verify RED because rendering functions and CLI do not exist.
- [ ] Implement MuJoCo side-view rendering, Pillow overlays, slowed terminal holds, mediapy H.264 output, and canonical JSON manifest.
- [ ] Run focused rendering tests and verify GREEN.
- [ ] Run the real CLI into `runs/two_phase/gate_b_20260803_failure_videos/` and inspect both MP4s.
- [ ] Commit `feat: render annotated dynamic failure videos`.

### Task 3: Automatic Gate B failure archive

**Files:**
- Modify: `cli/build_two_phase_guideline_banks.py`
- Modify: `tests/test_two_phase_guideline.py`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: `render_failure_archive(...)`.
- Preserves: existing `ValueError("Guideline physical event trace entered gate_pause")`.

- [ ] Write tests proving a failed event trace invokes archiving before the existing exception and that renderer failure records an error without changing the physical Gate pause.
- [ ] Run the tests and verify RED.
- [ ] Add failure-only archive invocation and `failure_video_status.json`; keep the original exception authoritative.
- [ ] Run targeted tests, compileall, full pytest, local preflight, and runtime fingerprint verification.
- [ ] Update experiment state with paths, hashes, transition counts, and the future failure-video rule.
- [ ] Commit `docs: record gate b failure video evidence`.

