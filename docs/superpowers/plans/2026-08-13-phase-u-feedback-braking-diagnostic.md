# Phase U Feedback-Braking Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one bounded natural-start diagnostic that tests whether deployable pitch/pitch-rate feedback can preserve coordinated upward propulsion while suppressing unsafe launch rotation.

**Architecture:** A focused diagnostic module owns the frozen controller grid, pure action law, outcome accounting, ranking, and media selection. A stable CLI constructs the already-qualified Phase-U adapter, evaluates every branch exactly once, writes immutable manifests before outcomes, and renders selected traces after physical outcomes are fixed.

**Tech Stack:** Python 3.11, JAX/JAX NumPy, MuJoCo/MJX, NumPy, pytest, JSON/SHA-256, existing DVGC renderer and two-phase runtime.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Do not modify `dvgc/env.py`, XML, 2 kg payload, +/-50 N m limits, action mapping, observation/history, reset, thresholds, deadlines, reward, PPO, or fixed evaluation seeds.
- Run exactly 384 branches from one fixed natural-reset seed, at most 80 real control ticks each and at most 30,720 diagnostic transitions.
- Record PPO training transitions as exactly zero.
- Freeze the manifest before any branch outcome and never adapt the grid after results.
- Do not call any result an expert, reachable state, safe state, Tube, or training reset.
- Explicitly stage paths, exclude `runs/` and `.vscode/`, and preserve every retained diagnostic video/trace in ignored run output.

---

### Task 1: Pure controller, grid, accounting, and ranking contracts

**Files:**
- Create: `dvgc/phase_u_launch_diagnostic.py`
- Create: `tests/test_phase_u_launch_diagnostic.py`

**Interfaces:**
- Consumes: `pitch`, `pitch_rate`, `window_latched`, `active_age`, and immutable `FeedbackLaunchSpec`.
- Produces: `feedback_launch_specs()`, `feedback_launch_action(...)`, `close_diagnostic_outcomes(rows)`, `rank_diagnostic_row(row)`, and `select_representative_rows(rows, maximum=8)`.

- [ ] **Step 1: Write failing grid and action tests**

Test that the grid has exactly 384 unique specs with the frozen Cartesian values. Test neutral pre-window action, declared feedback signs and clipping, nonnegative knee proportional to positive hip only, and neutral action after `active_ticks`.

- [ ] **Step 2: Run RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_u_launch_diagnostic.py
```

Expected: import failure because `dvgc.phase_u_launch_diagnostic` does not exist.

- [ ] **Step 3: Implement the minimal pure module**

Define frozen `FeedbackLaunchSpec`, exact nested-product grid, a JAX-compatible action function, mutually exclusive standard outcome closing, lexicographic ranking, and deterministic representative selection by terminal reason plus best progress and up to eight Apex rows.

- [ ] **Step 4: Run GREEN**

Run the RED command and require all tests to pass.

---

### Task 2: Stable diagnostic CLI and immutable provenance

**Files:**
- Create: `cli/diagnose_phase_u_feedback_launch.py`
- Modify: `tests/test_phase_u_launch_diagnostic.py`

**Interfaces:**
- Consumes: exact threshold manifest, default config, stable Phase-U training config for reward semantics, output directory, seed, horizon, and the Task-1 pure functions.
- Produces: `frozen_manifest.json`, `outcomes.jsonl`, `diagnostic_report.json`, `representative_media.json`, MP4 files, and aligned NPZ traces.

- [ ] **Step 1: Write failing manifest/runtime tests**

Test no-overwrite output, exact SHA-bound manifest fields, 384 branches, 30,720 ceiling, zero PPO transitions, forbidden-claim booleans, closed report accounting, fixed grid ordering, and media selection independent of render status. Use a small injected fake runner for CLI orchestration tests; do not mock the pure contracts.

- [ ] **Step 2: Run RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_u_launch_diagnostic.py
```

Expected: fail because the CLI orchestration API does not exist.

- [ ] **Step 3: Implement the minimal CLI**

Build a `ValidatedPhaseExpertRunSpec` in preflight mode, then construct the formal Phase-U adapter. Run each frozen spec from the same reset seed, start active age on the adapter's monotonic `jump_window_entered` event, gather physical/two-phase metrics, stop on adapter done or 80 ticks, and atomically append outcomes. Write the frozen manifest before the first rollout. Render only after the report is closed, reusing the existing phase-expert frame/video helper and writing qpos/qvel/ctrl/action NPZ arrays.

- [ ] **Step 4: Run GREEN and related regressions**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_u_launch_diagnostic.py \
  tests/test_phase_expert_training.py \
  tests/test_two_phase_runtime.py \
  tests/test_two_phase_semantics.py
```

---

### Task 3: Qualification, one frozen run, and scientific decision

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored artifacts only under: `runs/two_phase/diagnostics/phase_u_2kg_feedback_braking_20260813_seed731000/`

**Interfaces:**
- Consumes: qualified source, frozen manifest, authoritative XML/config/threshold/reward identities.
- Produces: one complete diagnostic evidence set and exactly one next decision.

- [ ] **Step 1: Run source qualification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
XLA_PYTHON_CLIENT_PREALLOCATE=false /home/qy/mujoco_playground/.venv/bin/python -m pytest -q
XLA_PYTHON_CLIENT_PREALLOCATE=false bash scripts/local_preflight.sh
```

- [ ] **Step 2: Commit and push the qualified diagnostic code**

Explicitly stage the new module, CLI, tests, and this plan; run `git diff --check`; commit and push the branch. Do not stage `.vscode/` or `runs/`.

- [ ] **Step 3: Run the diagnostic once**

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
/home/qy/mujoco_playground/.venv/bin/python \
  -m cli.diagnose_phase_u_feedback_launch \
  --config configs/default.json \
  --training-config configs/phase_expert_phase_u.json \
  --threshold-manifest runs/two_phase/gate_c1_20260813_coordinated_joint_v8_threshold_refresh/threshold_manifest.json \
  --seed 731000 \
  --horizon 80 \
  --run runs/two_phase/diagnostics/phase_u_2kg_feedback_braking_20260813_seed731000
```

Require no overwrite, exactly 384 outcomes, no more than 30,720 transitions, zero PPO transitions, and no parameter adaptation.

- [ ] **Step 4: Audit all outputs**

Validate closed outcome counts/rates, finite metrics, exact source/model/config/threshold/reward hashes, all 384 parameter identities, total transition accounting, representative media hashes, and every NPZ timing length. Inspect the best and each distinct terminal-class video.

- [ ] **Step 5: Record and commit the decision**

Update `docs/EXPERIMENT_STATE.md` with the exact run, budget, outcome classes, best physical residuals, media paths, and claim boundary. If Apex exists, select only an evidence-backed physical quantity for the next PPO hypothesis; if only safer partial progress exists, likewise select one quantity; if useful upward motion always remains unsafe, record the physical/control blocker. Commit and push docs. Do not start PPO within this task.
