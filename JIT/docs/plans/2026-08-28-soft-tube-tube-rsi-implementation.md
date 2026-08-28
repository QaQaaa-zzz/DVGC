# Learned Soft Tube and Unified Tube-RSI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a TRAIN-only learned Soft Tube, phase-balanced real-snapshot Tube-RSI, and one bounded single-policy engineering pilot.

**Architecture:** A host-side builder validates frozen identities and joins label rows to real snapshot catalogs before scoring each phase with its own frozen value model. A JAX sampler owns the 50/50 phase mixture and within-phase soft weights; a unified environment consumes sampled snapshots while one policy owns every action.

**Tech Stack:** Python 3.11, JAX, Flax, NumPy, MuJoCo MJX Warp, Brax PPO, pytest.

## Global Constraints

- Modify only `JIT/`; preserve all unrelated dirty paths.
- Use `/home/qy/mujoco_playground/.venv/bin/python` without changing it.
- Never retrain or modify `pi_up_star`, `pi_down_star`, `V_up`, or `V_down`.
- Never inspect TEST outcomes or use validation/TEST for Tube membership, tuning, resets, diagnostics, or PPO.
- Use only real saved snapshots; never mutate or synthesize qpos/qvel.
- Deployment is one `pi_unified`, never an expert-switching router.
- Commit only after a large milestone, not after individual TDD steps.

---

### Task 1: Soft Tube contract and builder

**Files:**
- Create: `JIT/src/jit_dvgc/soft_tube.py`
- Create: `JIT/cli/build_soft_tube.py`
- Create: `JIT/tests/test_soft_tube.py`

**Interfaces:**
- Consumes: frozen expert manifest, two value-model directories, exact upstream/downstream label and catalog paths.
- Produces: `build_soft_tube(inputs: SoftTubeInputs, output_dir: Path) -> dict[str, Any]`, `load_soft_tube(path: Path) -> SoftTubeArtifact`.

- [x] Write failing pure tests for identity mismatch, TRAIN-only selection,
  phase-specific scorer calls, exact state deduplication, conflict rejection,
  missing snapshots, and no validation/TEST output.
- [x] Run `PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests/test_soft_tube.py -q` and confirm failure is missing `jit_dvgc.soft_tube`.
- [x] Implement strict input models, hash verification, split-first row filtering,
  catalog joins, phase-local scoring, and atomic artifact creation.
- [x] Persist `manifest.json`, `entries.json`, and `diagnostics.json` with
  `value_score`, literal weight mapping, claim boundary, identities, and zero
  interaction accounting.
- [x] Re-run the focused test until green; do not commit yet.

### Task 2: Phase-balanced Tube-RSI sampler

**Files:**
- Create: `JIT/src/jit_dvgc/tube_rsi.py`
- Create: `JIT/tests/test_tube_rsi.py`
- Modify: `JIT/src/jit_dvgc/snapshot_pool.py`

**Interfaces:**
- Consumes: `SoftTubeArtifact` entries and the existing compatibility identity.
- Produces: `TubeRSIPool.from_artifact(...)`, `sample(rng)`, and
  `sample_at(phase_index, entry_index)` returning real snapshot pytrees plus
  `tube_phase` and stable entry indices.

- [x] Write failing tests proving literal 50/50 phase mass, monotonic positive
  weights, low-score support, fixed-seed determinism, both-phase support, and
  rejection of validation/TEST entries.
- [x] Run the focused test and confirm failure is the missing sampler.
- [x] Implement phase-first Bernoulli selection plus within-phase categorical
  selection; reuse `SnapshotPool` stacking and compatibility checks.
- [x] Re-run both Soft Tube and Tube-RSI tests until green; do not commit yet.

### Task 3: Unified reset and step smoke

**Files:**
- Create: `JIT/src/jit_dvgc/unified_env.py`
- Create: `JIT/configs/tube_rsi_smoke.json`
- Create: `JIT/src/jit_dvgc/tube_rsi_smoke.py`
- Create: `JIT/cli/smoke_tube_rsi.py`
- Create: `JIT/tests/test_tube_rsi_smoke.py`

**Interfaces:**
- Consumes: a validated Soft Tube and `TubeRSIPool`.
- Produces: one unified environment reset path and a closed smoke report with
  exactly 16 diagnostic interactions and zero training transitions.

- [x] Write failing host/GPU tests for upstream and downstream restore, one-step
  finiteness, phase initialization/transition, identical single-policy action
  path, XML/config mismatch rejection, and exact accounting.
- [x] Run the focused tests and confirm expected missing unified behavior.
- [x] Add the smallest unified config/state-machine extension that reuses
  existing Phase U and Phase D reward/terminal functions without new reward
  constants or expert calls.
- [x] Implement the predeclared fixed-index 8-up/8-down smoke and closed report.
- [x] Run focused host/GPU tests until green; do not commit yet.

### Task 4: Artifact construction and full verification

**Files:**
- Modify: `JIT/scripts/local_preflight.sh`
- Modify: `JIT/README.md`
- Update: `JIT/planning/task_plan.md`, `JIT/planning/findings.md`, `JIT/planning/progress.md`

**Interfaces:**
- Consumes: all validated code and frozen local artifacts.
- Produces: ignored `JIT/runs/soft_tube/<run_id>/` and
  `JIT/runs/tube_rsi/<run_id>/` artifacts.

- [x] Run targeted tests, then `bash JIT/scripts/local_preflight.sh`.
- [x] Build the Soft Tube from the exact hash-bound inputs without emitting
  validation or TEST diagnostics.
- [x] Audit counts, score/weight quantiles, roles, sources, near-boundary TRAIN
  entries, hashes, snapshot existence, and `test_data_used=false`.
- [x] Run the 16-interaction smoke and verify every restore/step plus accounting.
- [x] Record `SOFT_TUBE=GO` and `TUBE_RSI_SMOKE=GO` only if all evidence closes.

### Task 5: One-block unified PPO engineering pilot

**Files:**
- Create: `JIT/configs/pi_unified_pilot.json`
- Create: `JIT/src/jit_dvgc/unified_training.py`
- Create: `JIT/cli/train_unified.py`
- Create: `JIT/tests/test_unified_training.py`
- Update: `JIT/README.md`

**Interfaces:**
- Consumes: the GO Soft Tube and Tube-RSI pool.
- Produces: one fresh Actor/critic PPO checkpoint from exactly 25,600 training
  transitions, with no expert/value calls in policy inference.

- [x] Write and run failing tests for gate requirements, fresh initialization,
  one Actor, exact one-block accounting, checkpoint restore, and TEST exclusion.
- [x] Implement the minimum orchestration by reusing the existing PPO/checkpoint
  infrastructure and unified environment.
- [x] Run focused and full JIT verification before launch.
- [x] Predeclare and persistently launch the pilot; inspect startup once.
- [x] Explicitly stage only intended JIT source/tests/config/docs/plans, audit
  the index, and create one larger milestone commit after the pilot is running.
