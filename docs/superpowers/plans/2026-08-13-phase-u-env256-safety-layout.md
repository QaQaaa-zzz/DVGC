# Phase U 256-Environment Safety Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the unstable 512-environment Phase U runtime layout with a qualified 256-environment layout while preserving per-minibatch sample count and every scientific/physical contract.

**Architecture:** Change only stable training configuration and run-bound manifests. Pair 256 environments with 16 minibatches so each minibatch still contains 400 samples; use one 6,400-transition rollout block as the native checkpoint and bounded acquisition unit.

**Tech Stack:** Python, JSON configuration, Brax PPO, JAX/MJX, pytest.

## Global Constraints

- Do not launch or resume any 512-environment PPO process.
- Keep the authoritative 2 kg XML, +/-50 N m limits, action mapping, reward,
  reset, observations, safety gates, and PPO optimizer hyperparameters fixed.
- Use `/home/qy/mujoco_playground/.venv/bin/python` without reconfiguration.
- Run artifacts remain ignored and must not be committed.

### Task 1: Bind the stable layout contract

**Files:**
- Modify: `tests/test_phase_expert_training.py`
- Modify: `configs/phase_expert_phase_u.json`

- [ ] Change the existing stable-layout test to require 256 environments, 16
  minibatches, a 6,400-transition block, 30 blocks for 192,000 transitions,
  and 19,200 transitions each for three candidate/continuation checkpoints.
- [ ] Run that test and confirm it fails against the old 512/32 layout.
- [ ] Change only formal layout/seed/cadence/acquisition values to the 256
  layout and rerun the test.

### Task 2: Requalify source

**Files:**
- Modify: `docs/RUNTIME_GATE.json` through the stable CLI
- Modify: `docs/EXPERIMENT_STATE.md`

- [ ] Run the two phase-expert test files.
- [ ] Run compileall, full pytest, and local preflight.
- [ ] Run one fresh GPU runtime gate and verify `--check-only`.
- [ ] Record exact evidence, explicitly separating the interrupted/no-GPU
  attempts from the successful runtime gate.
- [ ] Commit focused validated source and documentation, then push the branch.

### Task 3: Qualify and launch

**Files:**
- Create ignored 256-env smoke/formal configs and authorization manifests
  under `runs/two_phase/`.
- Modify: `docs/EXPERIMENT_STATE.md`

- [ ] Preflight and execute one 6,400-transition 256-env smoke.
- [ ] Validate checkpoint sidecar, accounting, outcome categories, and all
  MP4/NPZ hashes; stop on any host/GPU abnormality.
- [ ] If clean, issue a fresh run-bound 998,400-transition authorization,
  launch it persistently, and inspect only startup/fixed milestones/terminal
  state.

