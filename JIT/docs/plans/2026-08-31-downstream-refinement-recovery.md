# Downstream Refinement Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make downstream refinement fail correctly during zero-interaction audit, bind the exact completed coarse-search evidence, and safely resume completed duration checkpoints before spending new interactions.

**Architecture:** Factor pre-runtime preparation into one shared audit path used by the CLI and real search. Add a strict duration-checkpoint loader that validates completed label or zero-candidate artifacts and leaves partial durations fail-closed.

**Tech Stack:** Python 3.11, pytest, JAX, existing `jit_dvgc` artifact loaders and SHA-256 provenance helpers.

## Global Constraints

- Work only in `/home/qy/DVGC`, with all edits under `JIT/`.
- Use `/home/qy/mujoco_playground/.venv/bin/python` without modifying the environment.
- Preserve the current user-owned dirty files and all completed run artifacts.
- Do not change XML, physics, rewards, actions, strengths, duration grid 17..32, 400-tick labels, or readiness minima.
- Do not start environment interactions until static, CPU/GPU, preflight, and enhanced audit gates pass.
- Explicitly stage only the listed JIT paths; do not use broad Git staging.

---

### Task 1: Bind exact coarse-search readiness

**Files:**
- Modify: `JIT/configs/envelope_iter0_downstream_refinement.json`
- Modify: `JIT/handoff/2026-08-31/ENVELOPE_ITER0_DOWNSTREAM_REFINEMENT_PRELAUNCH.json`
- Test: `JIT/tests/test_downstream_transition_refinement.py`

**Interfaces:**
- Consumes: completed coarse-search `summary.json` readiness values.
- Produces: exact expected readiness objects including `candidate_count`.

- [x] Add a test that requires upstream `candidate_count == 571`, downstream `candidate_count == 565`, and `candidate_count == positive_count + negative_count` for both phases.
- [x] Run the focused test and verify RED because the config omits both fields.
- [x] Add the two literal candidate counts to config and prelaunch declaration without changing any outcome count.
- [x] Run the focused test and verify GREEN.

### Task 2: Make audit-only validate real artifacts

**Files:**
- Modify: `JIT/src/jit_dvgc/downstream_transition_refinement.py`
- Modify: `JIT/cli/refine_downstream_transition_band.py`
- Test: `JIT/tests/test_downstream_transition_refinement.py`

**Interfaces:**
- Produces: `audit_downstream_transition_refinement(config_path: Path) -> dict[str, Any]`.
- The report includes status, config/frozen/checkpoint/Tube identities, prior label count/readiness, and downstream anchors.

- [x] Add a real-artifact test that calls the audit on the checked config and expects status `artifact_audit_valid`, 1,136 prior labels, and five downstream anchors.
- [x] Run the test and verify RED because the audit function does not exist.
- [x] Factor config, frozen policy, prior search, formal config, Tube, checkpoint payload hash, and anchor selection into a shared preparation helper.
- [x] Make both the public audit and real search consume that helper.
- [x] Change CLI `--audit-only` to print the artifact audit report.
- [x] Run the real CLI audit and verify it closes exact artifacts before output creation.
- [x] Run the focused audit tests and verify GREEN.

### Task 3: Validate completed duration checkpoints

**Files:**
- Modify: `JIT/src/jit_dvgc/downstream_transition_refinement.py`
- Test: `JIT/tests/test_downstream_transition_refinement.py`

**Interfaces:**
- Produces: `_load_completed_duration(duration_root, *, duration, policy_record, frozen_manifest_sha256) -> tuple[list[dict[str, Any]], dict[str, Any]]`.
- Empty rows are valid only for a fully validated `no_candidates` duration.

- [x] Add a test fixture containing a completed zero-candidate acquisition plus `duration_summary.json`; assert that resume returns no labels and preserves acquisition interaction accounting.
- [x] Run it and verify RED because current resume requires label files.
- [x] Implement the zero-candidate completed branch with exact acquisition protocol/catalog/summary checks.
- [x] Run it and verify GREEN.
- [x] Add a completed-label fixture and a protocol-SHA mutation; verify RED against the current shallow resume path.
- [x] Reuse the existing catalog validator and add exact protocol, policy, count, state-coverage, and split checks until GREEN.
- [x] During final review, add RED/GREEN coverage for canonical protocol self-hash validation and exact label-to-candidate parent-group binding.
- [x] Integrate the helper into the duration reconstruction loop and write progress for completed zero-candidate durations.
- [x] Run the full refinement test file and verify GREEN.

### Task 4: Verification and interaction gate

**Files:**
- Verify all modified JIT paths.
- Update: `JIT/planning/task_plan.md`
- Update: `JIT/planning/findings.md`
- Update: `JIT/planning/progress.md`

**Interfaces:**
- Produces: a committed, provenance-bindable repair only after all zero-interaction gates pass.

- [x] Run `py_compile` on the refinement source and CLI.
- [x] Re-run the four focused CPU test files after final review (25 passed).
- [x] Re-run the two declared GPU test files with `-m gpu` after final review (4 passed).
- [x] Re-run `bash JIT/scripts/local_preflight.sh` after final review (410 CPU + 14 GPU passed).
- [x] Re-run enhanced `--audit-only` after final review and verify exact policy, prior search, Tube, checkpoint, and five-anchor identities.
- [x] Review `git diff --check`, changed paths, and user-owned dirty paths.
- [ ] Explicitly stage only the validated JIT repair/docs/tests/planning paths and create one focused commit so `repository_head` truthfully identifies the code used for interactions.
- [ ] Re-run enhanced audit at the committed HEAD.
- [ ] Launch the absent declared refinement run without `--resume`, teeing its log and preserving the exact exit code.
- [ ] Inspect only progress/completion/abnormal exit; stop immediately at downstream readiness or duration-32 exhaustion.

### Task 5: Recover the duration-23 GPU allocation failure

- [x] Preserve Luna's read-only failure diagnosis and exact interaction counts.
- [x] Add RED tests for a source-bound repair resume, failed-label evidence
  validation/accounting, retry-directory selection, and shared compiled runtime
  callables.
- [x] Implement the smallest repair-resume path without accepting partial label
  rows or re-running the completed duration-23 acquisition.
- [x] Add RED/GREEN coverage that a completed retry revalidates its preserved
  failed-attempt artifact and includes both failed and successful label cost.
- [x] Re-run focused/static/full preflight and artifact audit gates (29 focused
  CPU, 4 focused GPU, 414 full CPU, and 14 full GPU tests passed).
- [ ] Commit the repair, invoke one explicit repair resume, and delegate sparse
  run monitoring/result analysis to Luna medium.
