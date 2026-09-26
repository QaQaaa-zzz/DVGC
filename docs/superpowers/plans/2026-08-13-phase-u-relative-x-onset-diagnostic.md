# Phase U Relative-X Onset Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the bounded diagnostic with one frozen full-structure relative-x onset variable and run the complete 1,152-branch comparison.

**Architecture:** Add a pure timing-spec product and onset action gate to the existing diagnostic module. Extend the stable CLI with an explicit `relative_x_onset` mode whose manifest, runner, accounting, and media reuse the first diagnostic contracts while recording onset and latch ticks separately.

**Tech Stack:** Python 3.11, JAX, MJX/Warp GPU, NumPy, pytest, JSON/SHA-256.

## Global Constraints

- Do not change environment, XML, reward, thresholds, action mapping, PPO, reset, observation, or the 384 feedback controller values.
- Exact onsets are 1.17, 1.12, and 1.07 m; exact branch count is 1,152; horizon is 80; seed is 731100; ceiling is 92,160 diagnostic transitions; PPO transitions are zero.
- Preserve pre-window zero reward and early-airborne nonsuccess/nonterminal semantics.
- Never adapt the grid after outcomes or claim expert/reachability/safety/Tube/reset status.

### Task 1: Pure onset contracts

**Files:**
- Modify: `dvgc/phase_u_launch_diagnostic.py`
- Modify: `tests/test_phase_u_launch_diagnostic.py`

- [ ] Write tests for exact product, threshold equality, pre-onset neutral action, and latch-independent monotonic active age.
- [ ] Run focused tests and observe RED due to missing onset API.
- [ ] Add frozen `RelativeXLaunchSpec`, `relative_x_launch_specs()`, and `relative_x_launch_action(...)` with no changes to the underlying feedback equation.
- [ ] Run focused tests GREEN.

### Task 2: CLI mode and provenance

**Files:**
- Modify: `cli/diagnose_phase_u_feedback_launch.py`
- Modify: `tests/test_phase_u_launch_diagnostic.py`

- [ ] Write failing tests for 1,152 manifest entries, 92,160 ceiling, separate onset/latch ticks, and zero-PPO claim boundary.
- [ ] Add `--mode relative_x_onset`, mode-specific exact specs, seed/horizon validation, and runner onset-age handling from pre-step formal Apex signals.
- [ ] Run focused and related regressions GREEN.

### Task 3: Qualify, run once, and decide

**Files:**
- Modify after evidence: `docs/EXPERIMENT_STATE.md`
- Create ignored run: `runs/two_phase/diagnostics/phase_u_2kg_relative_x_onset_20260813_seed731100/`

- [ ] Run compileall, full pytest, and local preflight.
- [ ] Commit/push source before dynamics.
- [ ] Execute all 1,152 branches exactly once; render outcome-driven representatives afterward.
- [ ] Audit hashes, identity, finite values, accounting, media timing, physical metrics, onset/latch ticks, and videos.
- [ ] Record and push the evidence-backed decision; do not automatically start PPO.
