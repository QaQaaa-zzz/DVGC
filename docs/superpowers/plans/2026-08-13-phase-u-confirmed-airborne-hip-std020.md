# Phase U Confirmed-Airborne Hip Exploration 0.20 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase only the stable Phase U hip exploration prior from 0.10 to 0.20 so PPO can sample the smallest currently demonstrated confirmed-liftoff action region.

**Architecture:** Keep all production Python and physical/task contracts unchanged. Bind the single hypothesis through the two stable JSON configs and their existing regression test, then requalify the changed runtime fingerprint before any fresh run-bound training authorization.

**Tech Stack:** JSON configuration, pytest, JAX/Brax PPO, MuJoCo/MJX runtime gate.

## Global Constraints

- Work only in `/home/qy/DVGC` using `/home/qy/mujoco_playground/.venv/bin/python`.
- Change only the hip entry of `policy_initial_action_std`: `[0.05, 0.05, 0.10, 0.05] -> [0.05, 0.05, 0.20, 0.05]`.
- Do not modify reward, reset, XML, safety, thresholds, observation/history, PPO layout, optimizer, horizon, or action mapping.
- Do not edit historical run-bound configs or run artifacts.
- Use 256 parallel environments and sparse terminal/checkpoint monitoring.

---

### Task 1: Stable exploration contract

**Files:**
- Modify: `tests/test_phase_expert_training.py`
- Modify: `configs/phase_expert_smoke.json`
- Modify: `configs/phase_expert_phase_u.json`

**Interfaces:**
- Consumes: `resolve_policy_initial_action_std(config) -> tuple[float, float, float, float]`.
- Produces: stable config contract `(0.05, 0.05, 0.20, 0.05)`.

- [ ] Change `test_phase_u_configs_select_explicit_exploration_and_reward_hypothesis` to expect `(0.05, 0.05, 0.20, 0.05)` and matching JSON order.
- [ ] Run `python -m pytest -q tests/test_phase_expert_training.py::test_phase_u_configs_select_explicit_exploration_and_reward_hypothesis`; verify RED reports actual hip value `0.10`.
- [ ] Change only index 2 of `policy_initial_action_std` in both stable configs to `0.20`.
- [ ] Re-run the focused test and verify GREEN.
- [ ] Run the Phase U targeted test set and confirm no reward or semantic regression.

### Task 2: Static and runtime requalification

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Runtime output: `runs/two_phase/runtime_gate/<fresh_run_id>/` (ignored)

**Interfaces:**
- Consumes: stable config hash and current source/runtime fingerprint.
- Produces: compile, full-test, preflight, and managed runtime-gate evidence.

- [ ] Run `python -m compileall dvgc cli`.
- [ ] Run `python -m pytest -q`.
- [ ] Run `bash scripts/local_preflight.sh`.
- [ ] Run a fresh managed `python -m cli.runtime_gate` with a unique run directory and verify its fixed 64+32 transition update/resume contract.
- [ ] Record exact test counts, runtime-gate result, and transition accounting in `docs/EXPERIMENT_STATE.md`.
- [ ] Explicitly stage the two configs, test, design/plan, and experiment-state document; commit and push the focused validation commit.

### Task 3: Fresh engineering smoke and formal authorization decision

**Files:**
- Runtime output: `runs/two_phase/phase_experts/<fresh_smoke_run_id>/` (ignored)
- Runtime authorization: `runs/two_phase/authorizations/<fresh_formal_run_id>.json` (ignored)
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: validated stable config, current source/tree/XML/threshold hashes.
- Produces: one smoke audit and, only if clean, one new formal run-bound authorization.

- [ ] Create a fresh 256-env smoke config by copying the stable smoke contract without altering any other variable; record purpose, inputs, cost ceiling, stopping condition, and output directory before running.
- [ ] Issue a smoke-bound authorization and run exactly one PPO rollout block plus fixed evaluations.
- [ ] Validate checkpoint sidecars, closed outcome accounting, MP4/NPZ hashes, finite metrics, and absence of runtime/provenance/safety corruption.
- [ ] If smoke integrity passes, create one fresh formal config with the existing 998,400-transition schedule and only hip std `0.20`, then issue one exact run-bound authorization.
- [ ] Launch persistently with a completion watcher, inspect startup once, record status/metrics/resume paths, and stop active polling.
- [ ] If smoke integrity fails, do not authorize formal training; enter Gate Pause with the actual reason.

