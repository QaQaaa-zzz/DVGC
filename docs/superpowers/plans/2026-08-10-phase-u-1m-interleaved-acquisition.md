# Phase U 1M Interleaved Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the validated Gate C1 Phase U smoke into an authorization-gated persistent run with a bounded physical reward, checkpoint evaluations, truthful warm-start resume, and evidence-gated candidate acquisition.

**Architecture:** Keep `OrangeBikeDVGC` unchanged and extend the external `PhaseExpertEnvAdapter`. Separate pure reward/evaluation/checkpoint/acquisition contracts from the run orchestrator so each is unit-testable. Execute one persistent Brax training process; publish aligned effective milestones and separately account for all non-training interactions.

**Tech Stack:** Python 3.12, JAX/MJX, Brax PPO, MuJoCo host rendering/audit, pytest, atomic JSON/JSONL artifacts.

## Global Constraints

- Work only in `/home/qy/DVGC` on `agent/two-phase-soft-tube`.
- Use `/home/qy/mujoco_playground/.venv/bin/python`; do not modify the environment.
- Keep natural Phase U reset, authoritative XML, 4 kg payload, +/-50 N m limits, action mapping, PPO optimization values, network, normalizer, horizon, and safety termination unchanged.
- Change one scientific hypothesis only: Phase U reward.
- Reference data supplies broad scales only; no state reset, action replay, imitation, pointwise tracking, reference time, or reference index.
- Cap Phase U training at 1,000,000 total environment transitions.
- Candidate snapshots, provisional labels, and expert trajectories are not learned soft Tubes.
- Do not start formal V_up, Phase D expert training, Soft Tube declaration, unified PPO, or JCE/JEL.
- Use red-green TDD for every production behavior change and explicitly stage focused commits.

---

### Task 1: Freeze the revised method and run contract

**Files:**
- Modify: `PROJECT.md`
- Modify: `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`
- Modify: `docs/EXPERIMENT_STATE.md`
- Create: `docs/superpowers/specs/2026-08-10-phase-u-1m-interleaved-acquisition-design.md`

**Interfaces:**
- Consumes: the user-approved execution specification.
- Produces: the authoritative expert/data-acquisition overlap, continuation-feasible terminology, reward boundary, 1M ceiling, and formal relabel contract.

- [ ] **Step 1: Update all three authoritative documents and add the exact design.**

- [ ] **Step 2: Scan for placeholders and contradictory old sequential wording.**

Run: `rg -n "TBD|TODO|train both experts to convergence|Tube = expert" PROJECT.md docs/METHOD_TWO_PHASE_SOFT_TUBE.md docs/EXPERIMENT_STATE.md docs/superpowers/specs/2026-08-10-phase-u-1m-interleaved-acquisition-design.md`

Expected: no placeholder and no statement treating trajectories as Tubes.

- [ ] **Step 3: Validate and commit only the documents.**

Run: `git diff --check`

Commit paths: `PROJECT.md docs/METHOD_TWO_PHASE_SOFT_TUBE.md docs/EXPERIMENT_STATE.md docs/superpowers/specs/2026-08-10-phase-u-1m-interleaved-acquisition-design.md docs/superpowers/plans/2026-08-10-phase-u-1m-interleaved-acquisition.md`

Commit message: `docs: define interleaved phase expert acquisition`

### Task 2: Implement the bounded Phase U reward and metric decomposition

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `configs/phase_expert_smoke.json`
- Create: `configs/phase_expert_phase_u.json`
- Modify: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: `ApexBandSignals`, `ApexBandThresholds`, monotonic two-phase event state, current and previous actions.
- Produces: `PhaseURewardConfig`, `phase_u_reward_components(...) -> dict[str, Array]`, static per-component Brax metrics, and a reward-contract hash over all weights/scales/bounds.

- [ ] **Step 1: Write failing reward tests.**

Add tests that assert the exact component key set, finite configured bounds,
zero jump/ascent/clearance/Apex progress before `jump_window_entered`, nonzero
post-entry ascent and clearance gradients, no early-airborne success, one-time
window/Apex bonuses, independent physical/task/contact penalties, and JIT/vmap
compatibility. The production change that makes these tests pass is the new
reward signature and metric publication.

- [ ] **Step 2: Run RED.**

Run: `/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py -k 'reward or pre_window'`

Expected: failures for missing component names/config fields and post-entry gate.

- [ ] **Step 3: Implement the minimal pure-JAX reward.**

Use the exact weights/scales in the design spec. Add a bounded interval-proximity helper:

```python
def _interval_proximity(value, lower, upper):
    width = jp.maximum(upper - lower, 1.0e-6)
    distance = jp.maximum(lower - value, jp.maximum(value - upper, 0.0))
    return jp.clip(1.0 - distance / width, 0.0, 1.0)
```

Pass monotonic `event.jump_window_entered`, its transition, Apex eligibility,
physical failure, and task failure to `phase_u_reward_components`. Publish every
component at reset and step under `phase_expert/reward_component/<name>`.

- [ ] **Step 4: Run GREEN and focused adapter regression tests.**

Run: `/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py -k 'reward or adapter or window or failure'`

Expected: all selected tests pass.

- [ ] **Step 5: Commit the reward-only hypothesis.**

Commit paths: `dvgc/phase_expert_training.py configs/phase_expert_smoke.json configs/phase_expert_phase_u.json tests/test_phase_expert_training.py`

Commit message: `feat: add bounded phase u task reward`

### Task 3: Add physical held-out evaluation and aligned milestones

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `dvgc/runtime.py`
- Modify: `cli/train_phase_expert.py`
- Modify: `configs/phase_expert_phase_u.json`
- Modify: `tests/test_phase_expert_training.py`
- Modify: `tests/test_optional_runtime.py`

**Interfaces:**
- Consumes: fixed deterministic policy params, evaluation seeds, PPO rollout block size, requested milestones.
- Produces: `align_phase_u_checkpoints(...)`, per-row physical metrics and component sums, evaluation summaries, milestone callbacks, and separate interaction counters.

- [ ] **Step 1: Write failing schedule and evaluation tests.**

Assert requested `(0,100000,250000,500000,750000,1000000)` maps with a 1,600
block to `(0,100800,251200,500800,750400,1000000)`, rejects duplicates/out-of-
range milestones, and never exceeds the 1M ceiling. Add deterministic synthetic
rollout tests for window/liftoff/clearance/Apex rates, attitude violations,
forward retention, action saturation, extrema, and reward-component totals.

- [ ] **Step 2: Run RED.**

Run: `/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py tests/test_optional_runtime.py -k 'checkpoint_schedule or physical_evaluation or run_evals'`

Expected: missing helper/metrics/runtime option failures.

- [ ] **Step 3: Implement schedule/evaluation and callback controls.**

Expose optional `run_evals` and optional automatic checkpoint directory in
`make_ppo_train_fn`. Keep training stochastic. Use per-block host callbacks with
Brax internal evaluation disabled; run external fixed evaluation only at aligned
milestones. Record requested/effective milestone pairs and every interaction
category atomically.

- [ ] **Step 4: Run GREEN and all Gate C tests.**

Run: `/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py tests/test_optional_runtime.py tests/test_training_budget.py`

Expected: all pass.

- [ ] **Step 5: Commit the physical checkpoint protocol.**

Commit message: `feat: add phase u checkpoint evaluation protocol`

### Task 4: Correct resume identity and add evidence-gated acquisition hooks

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Create: `dvgc/phase_candidate_acquisition.py`
- Modify: `cli/train_phase_expert.py`
- Create: `tests/test_phase_candidate_acquisition.py`
- Modify: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: checkpoint params/hash, stochastic acquisition seeds, fixed evaluation summary, online states and event timings.
- Produces: `checkpoint_payload="normalizer_policy_value"`, `optimizer_state_included=false`, warm-start resume validation, unique parent trajectory hashes, candidate eligibility, timing-explicit v4 snapshot hooks, and separately counted acquisition/continuation diagnostics.

- [ ] **Step 1: Write failing checkpoint-truth and acquisition tests.**

Reject `full_training_state=true`; require explicit absent optimizer/env-step
state and warm-start semantics. Test eligibility requires fixed Apex success,
eight distinct successful seeds and content hashes, and clean contracts. Test
duplicate parents, one lucky success, invalid FIFO/timing, and failed physical
validation are rejected before candidate admission.

- [ ] **Step 2: Run RED.**

Run: `/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_expert_training.py tests/test_phase_candidate_acquisition.py -k 'checkpoint or acquisition or parent'`

Expected: current full-state contract and missing acquisition module fail.

- [ ] **Step 3: Implement minimal warm-start and harvesting hooks.**

Do not synthesize snapshots. Capture only live state/timing/history/last-action
records, bind parent trajectory, checkpoint, XML/config/action hashes, and write
candidate manifests under the run. Continuation probing remains disabled until
eligibility is true and uses Gate A closed outcome accounting when enabled.

- [ ] **Step 4: Run GREEN and round-trip regressions.**

Run: `/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_phase_candidate_acquisition.py tests/test_phase_expert_training.py tests/test_two_phase_snapshot_roundtrip.py`

Expected: all pass.

- [ ] **Step 5: Commit the truthful resume and acquisition boundary.**

Commit message: `feat: add phase u candidate acquisition gate`

### Task 5: Verify, smoke, and launch the persistent authorized run

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Runtime outputs: `runs/two_phase/phase_experts/<run_id>/` (ignored)
- Runtime authorization: `runs/two_phase/authorizations/<authorization>.json` (ignored)

**Interfaces:**
- Consumes: clean committed source, fixed hashes, disabled watchdog, bounded smoke and 1M authorizations.
- Produces: fresh smoke evidence and, only if it passes, one persistent formal Phase U process with status/metrics/resume evidence.

- [ ] **Step 1: Run static and full verification.**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate --check-only
```

- [ ] **Step 2: Recheck watchdog and source cleanliness.**

Require timer/service inactive, active pointer absent, authoritative hashes
current, `.vscode/` preserved, and no uncommitted source/config/doc changes.

- [ ] **Step 3: Issue a one-use smoke authorization and run 1–4 rollout blocks.**

Require finite reward/gradients, checkpoint warm-start validation, fixed
physical evaluation, metrics/status/accounting closure, and failure videos.

- [ ] **Step 4: Audit smoke without changing another hypothesis.**

If a pause condition occurs, stop and record it. Zero success alone does not
pause. If engineering integrity passes, issue a distinct source/run-bound 1M
authorization with the fixed milestone schedule.

- [ ] **Step 5: Launch once as a persistent process and inspect startup once.**

Record PID, command, run ID, `status.json`, `metrics.jsonl`, log path, authorized
budget, checkpoint schedule, and warm-start resume command. Do not repeatedly
poll the process or wait for 1M completion in this session.

- [ ] **Step 6: Update and commit experiment state, then push the branch.**

Report all transition categories actually consumed. Keep formal V_up, formal
Tube-up, Phase D real Apex seed count, and promotion claims explicitly false or
zero unless artifacts genuinely exist.
