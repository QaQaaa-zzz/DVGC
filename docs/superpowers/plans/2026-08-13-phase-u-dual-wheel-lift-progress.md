# Phase U Dual-Wheel Lift Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deployable, window-gated, low-angular-rate dual-wheel lift credit and qualify one fresh Phase U experiment without changing success, termination, dynamics, or action semantics.

**Architecture:** The existing pure-JAX two-phase runtime derives the minimum terrain-relative clearance over collision-relevant wheel geoms and carries it in `ApexBandSignals`. The Phase U adapter converts that present-time signal into one bounded reward component using the existing legal-window latch and v5 angular-rate quality. The reward schema and hash remain exact and auditable.

**Tech Stack:** Python, dataclasses, JAX/JAX NumPy, MJX/MuJoCo geometry, pytest, Brax PPO, Git.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Keep `assets/orange_bike_4kg_horizontal.xml` at its authoritative 2 kg payload and preserve +/-50 N m hip/knee limits.
- Do not change XML geometry, action mapping, reset, observations, termination, thresholds, deadline positions, network, optimizer, horizon, seed namespace, or PPO layout.
- Before window entry, including early airborne, all jump/lift/ascent task progress remains exactly zero.
- Dual-wheel lift credit is neither liftoff, Apex success, done, nor a safety claim.
- Run one 256-environment smoke before any fresh formal authorization; do not resume v5.
- Formal training and diagnostic/evaluation transitions remain separately accounted.

---

### Task 1: Pure-JAX minimum wheel clearance signal

**Files:**
- Modify: `dvgc/two_phase_semantics.py`
- Modify: `dvgc/two_phase_runtime.py`
- Test: `tests/test_two_phase_runtime.py`
- Test: `tests/test_two_phase_semantics.py`

**Interfaces:**
- Produces: `ApexBandSignals.minimum_wheel_terrain_clearance`
- Consumes: existing `TwoPhaseGeometry.wheel_mask`, analytic support bounds, and `_terrain_clearances`.

- [ ] Write tests that instantiate `ApexBandSignals` with the new field and assert the runtime returns the minimum clearance over wheel-mask geoms, including `jax.jit` and `jax.vmap` paths.
- [ ] Run the focused tests and confirm RED due to the missing field/value.
- [ ] Add the field to `ApexBandSignals`; in `extract_apex_band_signals`, compute `jp.min(jp.where(wheel_mask, terrain_clearances, jp.inf), axis=-1)` and populate it.
- [ ] Update all real test constructors with explicit values and run the focused tests to GREEN.

### Task 2: Bounded reward component and exact config schema

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `configs/phase_expert_smoke.json`
- Modify: `configs/phase_expert_phase_u.json`
- Test: `tests/test_phase_expert_training.py`

**Interfaces:**
- Produces: `PhaseURewardConfig.dual_wheel_lift_progress_weight`, `PhaseURewardConfig.dual_wheel_lift_progress_target`, and metric `phase_expert/reward_component/dual_wheel_lift_progress`.
- Consumes: `ApexBandSignals.minimum_wheel_terrain_clearance`, legal-window latch, and the existing v5 `ascent_rate_quality`.

- [ ] Add tests for exact-schema loading and validation: finite non-negative weight, finite positive target, and stable reward hash.
- [ ] Add direct component tests proving zero before window, zero at/below ground, half credit at 0.0075 m, full bounded credit at/above 0.015 m, half credit at half angular-rate cap, and zero at the cap.
- [ ] Add adapter regression proving positive lift credit cannot set liftoff, success, done, physical failure, or task failure.
- [ ] Run those tests and confirm RED because fields/component are absent.
- [ ] Add the two dataclass fields, validation, reward component name, and formula; set both stable configs to weight 4.0 and target 0.015.
- [ ] Run focused Phase U tests to GREEN and update expected hashes/counts only from actual deterministic output.

### Task 3: Qualification and documentation

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: committed source, reward hash, test/runtime outputs.
- Produces: recoverable current-state record and a fresh smoke permission marker.

- [ ] Run `/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli`.
- [ ] Run targeted two-phase and Phase U tests.
- [ ] Run the full `pytest -q` suite.
- [ ] Run `bash scripts/local_preflight.sh`.
- [ ] Run a fresh managed `python -m cli.runtime_gate` output under `runs/two_phase/runtime_gate/` and confirm its 64+32 transition accounting and current fingerprint.
- [ ] Record the v5 completed result, frozen stochastic diagnostics, rejected negative-knee interpretation, new contract hash, and exact validation evidence in `docs/EXPERIMENT_STATE.md`.
- [ ] Explicitly stage source/tests/config/docs, commit focused changes, and push the branch.

### Task 4: Smoke and fresh formal run

**Files:**
- Runtime outputs only under ignored `runs/two_phase/`
- Modify after evidence: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Produces: run-bound authorization, smoke evidence, formal status/metrics/checkpoints/videos.

- [ ] Preflight one fresh 256-env smoke with exact purpose, inputs, ceilings, stopping conditions, output directory, source/reward/XML/config hashes, and seed.
- [ ] Run 1 PPO rollout block; verify finite update, checkpoint/sidecar, resume path, fixed evaluation, closed outcome accounting, media hashes, and zero contract/runtime faults.
- [ ] If engineering integrity passes, create one exact authorization for fresh initialization up to 998,400 aligned Phase U training transitions at checkpoints 0/102,400/256,000/505,600/755,200/998,400.
- [ ] Launch it persistently with a compact watcher that writes only a terminal marker; inspect startup once, then stop active polling.
- [ ] At terminal or Gate Pause, audit fixed physical metrics, all sidecars, interaction accounting, and representative failure videos/state traces.
- [ ] If at least eight independent Apex-success parents exist, allow the existing bounded candidate/continuation hooks; otherwise document the evidence and form exactly one new hypothesis.
- [ ] Commit and push each validated documentation round without committing run artifacts.
