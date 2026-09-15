# Phase U Coordinated Joint-Propulsion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the downstream v7 synchronized-wheel velocity reward with bounded, legal-window-only coordinated actual hip/knee propulsion credit so PPO receives physical launch-direction feedback before liftoff.

**Architecture:** `PhaseExpertEnvAdapter` reads the post-step authoritative hip and knee DoF velocities through immutable model addresses and passes one synchronized signed velocity to the pure-JAX Phase U reward. The reward requires both joints to advance in the approved physical direction, applies an evidence-bounded deadband and the existing angular-rate quality, and leaves environment, event, success, termination, observation, XML, action mapping, PPO, and evaluation contracts unchanged.

**Tech Stack:** Python 3.11, JAX/JAX NumPy, MuJoCo/MJX, Brax PPO, pytest, JSON and SHA-256 provenance.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Do not modify `dvgc/env.py`, XML, 2 kg payload, +/-50 N m limits, action mapping, observation/history, reset, thresholds, deadlines, PPO/network/optimizer/exploration/horizon, or fixed seeds.
- Keep pre-window task-progress reward exactly zero and early airborne nonterminal/nonsuccess.
- Keep the existing public metric/component name `dual_wheel_lift_progress`, but bind v8 semantics and exact joint-propulsion config fields in every manifest/hash.
- Preserve `dvgc.two_phase_runtime.wheel_terrain_clearances` as a tested geometry API even though v8 reward no longer consumes it.
- Do not resume v7; every dynamic qualification after source validation needs a fresh run-bound authorization.
- Explicitly stage paths and preserve the user-owned `.vscode/` directory.

---

### Task 1: Reward schema and pure-JAX coordinated progress

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `configs/phase_expert_phase_u.json`
- Modify: `configs/phase_expert_smoke.json`
- Test: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: `coordinated_joint_velocity: Any`, existing legal-window state, and existing angular-rate quality.
- Produces: `dual_wheel_lift_progress` metric with v8 semantics and config fields `coordinated_joint_propulsion_weight`, `coordinated_joint_velocity_deadband`, and `coordinated_joint_velocity_target`.

- [ ] **Step 1: Write failing schema and reward tests**

Replace v7 field assertions with exact v8 fields and values 4.0, 0.15, and 2.0. Add invalid nonfinite, negative, zero-target, and `deadband >= target` cases. Call `phase_u_reward_components` with `coordinated_joint_velocity` and assert pre-window zero, 0.14/0.15 zero, 1.075 half credit, and 2.0/3.0 full credit. Assert angular-rate qualification and reward-hash drift for every field.

- [ ] **Step 2: Run RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py -k \
  'coordinated_joint or dual_wheel or reward_contract_hash or reward_manifest'
```

Expected: fail because v7 still defines wheel-velocity fields and function input.

- [ ] **Step 3: Implement the minimal schema and reward**

Replace the three v7 reward fields in `PhaseURewardConfig`; validate finite nonnegative deadband, finite positive target, and strict ordering. Rename the function input to `coordinated_joint_velocity`, compute deadbanded normalized progress, multiply by weight and existing angular-rate quality, and keep the public component key. Set semantics to `phase_u.coordinated_actual_joint_propulsion_credit.v8`. Update only the corresponding JSON fields in both stable configs.

- [ ] **Step 4: Run GREEN**

Run the RED command, then all of `tests/test_phase_expert_training.py`.

---

### Task 2: Adapter reads real post-step hip/knee velocity

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Test: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: `raw.data.qvel`, authoritative `_joint_qvel['hip_joint']` and `_joint_qvel['knee_joint']` addresses.
- Produces: `min(max(hip_qvel, 0), max(-knee_qvel, 0))` passed to `phase_u_reward_components`.

- [ ] **Step 1: Write failing adapter tests**

Extend the fake base environment with immutable joint DoF addresses and scripted post-step qvel. Assert both approved directions earn progress, either joint alone earns zero, either wrong sign earns zero, and changing action/ctrl without changing qvel earns zero. Assert the reward does not change event, success, task failure, physical failure, timeout, or done values.

- [ ] **Step 2: Run RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py -k \
  'coordinated_joint or physical_joint or previous_wheel'
```

Expected: fail because the adapter still carries previous wheel clearances.

- [ ] **Step 3: Implement the minimal adapter behavior**

Resolve the two DoF addresses once in the adapter constructor. Remove the reward-only clearance extractor and `phase_expert/previous_wheel_terrain_clearances` state. After the base step, gather the two qvel values, form the signed synchronized velocity, and pass it to the reward. Do not read actions or controls for this component and do not change base environment state.

- [ ] **Step 4: Run GREEN and regressions**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py \
  tests/test_two_phase_runtime.py \
  tests/test_two_phase_semantics.py \
  tests/test_training_budget.py
```

---

### Task 3: Full source/runtime qualification and recoverable evidence

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Modify only if an existing contract assertion requires it: `PROJECT.md`, `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`

**Interfaces:**
- Consumes: completed v7 terminal audit, v8 reward/source hashes, validation and runtime-gate reports.
- Produces: one committed, remote-synced v8 qualification marker and exact next permitted smoke.

- [ ] **Step 1: Run static and full verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python -m pytest -q
XLA_PYTHON_CLIENT_PREALLOCATE=false bash scripts/local_preflight.sh
```

- [ ] **Step 2: Run a fresh managed runtime gate**

Run `python -m cli.runtime_gate` with a new v8 report/work directory. Require exact 64 update plus 32 resume transitions, current fingerprint, snapshot/restore, policy determinism, and no runtime fault.

- [ ] **Step 3: Refresh static threshold provenance only**

Copy approved values and geometry manifest, update only current source hashes and canonical manifest hash, validate with `load_phase_expert_threshold_manifest`, and prove all selected thresholds/raw extrema/margins/anchors are byte-equal to v7.

- [ ] **Step 4: Update experiment state**

Record v7 completion at 998,400, six-panel all-grounded outcomes, 157/157 sidecars, 48 MP4 plus 48 NPZ hashes, training-period sparse credit, grounded joint-velocity bound, chosen v8 hypothesis, exact hashes, tests, preflight, runtime gate, and zero new formal v8 training transitions.

- [ ] **Step 5: Commit and push qualified source/docs**

Explicitly stage implementation, tests, stable configs, and docs; exclude `.vscode/` and `runs/`.

---

### Task 4: Fresh smoke and formal v8 retry

**Files:**
- Create ignored artifacts only under `runs/two_phase/`.
- Modify after verified evidence: `docs/EXPERIMENT_STATE.md`.

**Interfaces:**
- Consumes: committed HEAD/source/model/config/threshold/reward identities.
- Produces: one 256-env smoke and, only if clean, one fresh v8 formal run capped at 998,400 PPO-training transitions.

- [ ] **Step 1: Build and preflight one smoke authorization**

Bind purpose, identities, fresh seed/run ID, 256 environments, 6,400 training, 1,600 Brax evaluation, 1,600 fixed evaluation ceiling, zero candidate/continuation, 9,600 total ceiling, and stopping condition. Require preflight to report zero executed transitions.

- [ ] **Step 2: Run and audit one smoke**

Require finite reset/reward/update/checkpoint/warm-start/fixed evaluation, recursive sidecar identity, closed outcomes/rates, exact accounting, and eight MP4/NPZ hashes. Treat physical performance as diagnostic only.

- [ ] **Step 3: Commit and push smoke evidence**

Update only `docs/EXPERIMENT_STATE.md`, run `git diff --check`, commit explicitly, and push the branch so formal authorization binds the documented HEAD.

- [ ] **Step 4: Build exact formal input and authorization**

Prove canonical equality to v7 formal input except the three v8 fields replacing the three v7 fields. Bind fresh seed/run ID, current committed HEAD/source/config/XML/threshold/reward hashes, 256 environments, effective checkpoints 0/102,400/256,000/505,600/755,200/998,400, 998,400 training, 9,600 fixed evaluation, candidate/continuation ceilings each 38,400, and total ceiling 1,084,800. Require zero-transition preflight.

- [ ] **Step 5: Launch once and supervise sparsely**

Use a persistent resumable process plus a one-minute PID-only completion watcher. Audit startup once, then inspect only fixed panels, abnormal exit, or terminal state.

- [ ] **Step 6: Audit terminal evidence and choose the next gate**

Validate all sidecars, six closed outcome panels, all media hashes, finite metrics, and physical launch/Apex/return/component/saturation distributions. If at least eight independent Apex-success parents exist, start candidate snapshots and bounded continuation probing. Otherwise preserve videos and select one new evidence-backed hypothesis without stacking changes.
