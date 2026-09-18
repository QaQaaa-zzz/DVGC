# Initial Three-Step Pulse Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a paired 10,000-episode comparison for four frozen policies with the only disturbance at control ticks 0–2.

**Architecture:** Add a configuration-driven onset override to the existing batched comparison preparer and retain the existing fixed-random rollout implementation. Add an array-level verification gate for the declared initial-only pulse, then freeze a committed source snapshot and launch the existing finite supervisor and watcher.

**Tech Stack:** Python, NumPy, JAX/MJX runtime, pytest, existing JIT batch supervisor.

## Global Constraints

- Four policies and 10,000 disturbed episodes per policy; maximum 16,000,400 control interactions including one new nominal rollout.
- Pulse ticks are exactly 0, 1, and 2 at 20 ms per control tick.
- No training and no changes to checkpoint, reset, reward, amplitude, horizon, or success criterion.
- No disturbance after tick 2; the jump policy controls every tick.
- Preserve the old experiment and all unrelated working-tree files.

---

### Task 1: Generic onset override and timing verification

**Files:**
- Modify: `JIT/src/jit_dvgc/batched_pulse_comparison.py`
- Modify: `JIT/src/jit_dvgc/analysis/batched_pulse_report.py`
- Modify: `JIT/cli/run_rsi_comparison.py`
- Modify: `JIT/tests/test_batched_pulse_comparison.py`

**Interfaces:**
- Consumes: completed three-policy comparison directory, `pulse_start_schedule: list[int]`, and an optional JSON additional-method manifest.
- Produces: `prepare(..., pulse_start_schedule=None, additional_methods=None)` and a frozen batch spec whose method templates share the overridden schedule.

- [x] Add failing tests proving `[0]` reaches the common contract and all method templates while the source files remain unchanged.
- [x] Add failing tests proving an additional frozen method is locked, gets a matched nominal rollout, and appears in a dynamically sized report.
- [x] Add a failing tape-verification test that inserts requested disturbance after tick 2 and expects rejection under the initial-only contract.
- [x] Implement CLI parsing and generic validation for the optional schedule override.
- [x] Implement array-level verification that the declared three-step initial pulse has no requested disturbance after tick 2.
- [x] Run the focused tests and confirm they pass.

### Task 2: Freeze and launch the corrected experiment

**Files:**
- Create: `JIT/runs/experiments/four_policy_initial_pulse_10k_20260918/`
- Modify: `JIT/docs/CURRENT_STATUS.md`
- Modify: `JIT/docs/RSI_FROM_SCRATCH_COMPARISON.md`
- Modify: `JIT/AGENTS.md`

**Interfaces:**
- Consumes: committed Task 1 code and the completed three-policy comparison as policy/provenance source.
- Produces: immutable `spec.json`, source snapshot, launch receipt, live status, watcher state, batch receipts, and final reports.

- [x] Run all related CPU tests and inspect the staged diff.
- [x] Commit and push the code and protocol documentation.
- [x] Prepare the new run with `--episodes 10000 --batch-size 256 --seed 9183002 --pulse-start-schedule 0 --resource-wait-timeout-seconds 604800` and the locked `lineage_repair_0070` additional-method manifest, so the run waits for the active recovery job instead of stopping it.
- [x] Verify the frozen contract has schedule `[0]`, three steps, all four requested policies, and a 16,000,400-interaction ceiling.
- [x] Launch from an isolated committed source snapshot and verify the supervisor, first child, and desktop watcher are alive.
- [x] Record live PIDs, status paths, exact protocol, and the superseded interpretation of the mixed-onset plot.

### Task 3: Completion report

**Files:**
- Create: `JIT/runs/experiments/four_policy_initial_pulse_10k_20260918/comparison/`
- Modify: `JIT/runs/experiments/four_policy_initial_pulse_10k_20260918/INDEX.md`
- Modify: `JIT/docs/CURRENT_STATUS.md`

**Interfaces:**
- Consumes: all 40 verified batch receipts.
- Produces: PNG/PDF/SVG, plot-data NPZ, 30,000-row CSV, summary JSON, and source hashes.

- [ ] Confirm all four denominators equal 10,000 and all batch timing gates passed.
- [ ] Generate the readable x≤5.5 m report without changing statistical denominators.
- [ ] Inspect the image and reconcile plotted values against the summary and CSV.
- [ ] Commit and push the completed status and reusable reporting changes; keep large run artifacts in the experiment directory.
