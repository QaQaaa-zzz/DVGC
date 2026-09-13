# Two-Phase Gate B Implementation Plan

> **Historical plan:** The complete reference replay and guideline-bank
> promotion requirements in this plan are superseded by
> `docs/superpowers/specs/2026-08-03-two-phase-gate-b-design.md`. Keep this file
> only as implementation provenance; do not resume its replay-repair or bank
> promotion tasks.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-JAX two-phase runtime adapter, reproducible guideline threshold and initial-bank builder, and authoritative timing-explicit round-trip validator without training any policy or model.

**Architecture:** `dvgc.two_phase_runtime` owns immutable geometry and pure batched signal/event extraction. `dvgc.two_phase_guideline` owns deterministic reference envelopes, manifests, reconstructed physical states, and v4 bank records. `dvgc.two_phase_roundtrip` owns explicit replay comparisons, while one CLI orchestrates ignored run artifacts and gate reporting.

**Tech Stack:** Python 3.12, JAX/MJX, MuJoCo host audit, NumPy/Pandas, existing DVGC v4 snapshots and `SnapshotBank`, pytest.

## Global Constraints

- Work only in `/home/qy/DVGC` on `agent/two-phase-soft-tube`, based on `5331896bee08a920321a9b39b496f66c7b9b0879`.
- Use `/home/qy/mujoco_playground/.venv/bin/python`; never change the virtual environment.
- Keep `dvgc/env.py`, `env.step`, reward, termination, observation, reset, action mapping, XML, and matcher unchanged unless a failing test proves an inaccessible immutable static value and the user separately approves the edit.
- Formal online geometry and signals are pure JAX; host `mj_geomDistance` is representative audit only.
- No experts, continuation labels, feasibility models, Soft Tubes, unified PPO, pilot, or formal training.
- Run artifacts and banks live under `runs/two_phase/gate_b_<run_id>/` and remain uncommitted.
- Every production behavior follows red-green TDD.
- Explicitly stage focused paths; never use `git add .` or `git add -A`.
- Gate B failure produces `gate_pause`; never weaken thresholds or enter Gate C.

---

### Task 1: Pure-JAX immutable geometry and runtime signals

**Files:**
- Create: `dvgc/two_phase_runtime.py`
- Create: `tests/test_two_phase_runtime.py`

**Interfaces:**
- Consumes: Gate A `ApexBandSignals`, `RecoverySignals`, threshold dataclasses, and `advance_recovery_hold_count`.
- Produces: `TwoPhaseGeometry`, `TwoPhaseEventState`, `build_two_phase_geometry`, `geometry_manifest`, `validate_geometry_manifest`, `collision_geom_support_bounds`, `extract_apex_band_signals`, `extract_recovery_signals`, and `extract_two_phase_events`.

- [ ] **Step 1: Write failing geometry coverage and sign tests**

Use the real authoritative `MjModel` to assert the manifest lists every geom and that every collision-relevant robot geom is supported. Build hand-checked synthetic box, cylinder, and ellipsoid transforms and assert literal x/z support bounds. Assert `obstacle_relative_x` is positive/boundary/negative around the obstacle front using all robot geoms, so deleting the leading geom or replacing the calculation with root x fails.

- [ ] **Step 2: Verify geometry tests fail for the missing module**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_two_phase_runtime.py
```

Expected: collection failure because `dvgc.two_phase_runtime` does not exist.

- [ ] **Step 3: Implement minimal immutable geometry and support formulas**

Create frozen dataclasses containing JAX arrays for geom ids/types/sizes/body ownership, obstacle bounds, wheel/body masks, root/joint ids, and physical tolerances. Implement world-direction support radii for MuJoCo sphere, capsule, ellipsoid, cylinder, box, and mesh rejection. Build and validate the manifest from collision flags rather than names alone.

- [ ] **Step 4: Verify geometry tests are green**

Run the Task 1 test file and retain literal expected values independent of production helpers.

- [ ] **Step 5: Add failing signal, `jit`, and `vmap` tests**

Construct complete synthetic MJX-like data/info pytrees and the real environment state. Assert Apex extraction uses full-structure clearance and exact obstacle sign; Recovery extraction rejects unsupported wheels, illegal landing x/y, body penetration, and physical failure. Compile one extraction with `jax.jit` and batch at least two states with `jax.vmap`. Mutating reward, legacy phase id, matcher-like fields, or reference indices must not change outputs.

- [ ] **Step 6: Implement minimal signal extraction**

Read only `state.data`, approved deployable `state.info` history, config scalars, and immutable geometry. Compute angular norm from the gyro sensor/body velocity, CoM/root velocity from state, current support/body legality from analytic geometry and deployable history, and carry only the caller-supplied previous two-phase hold count.

- [ ] **Step 7: Add failing event-order and immutability tests**

Feed a literal physical sequence and assert the first occurrence order of all ten events, Apex consecutive width, recovery counter reset, and unchanged input leaves. Prove legacy phase fields cannot select `propulsion_ascent` or `descent_recovery`.

- [ ] **Step 8: Implement pure event-state transition and verify green**

Return a new frozen `TwoPhaseEventState` with latches, first-event ticks, previous vertical velocity, and recovery hold. Use JAX logical operations only. Run Task 1 tests.

- [ ] **Step 9: Commit Task 1**

```bash
git add -- dvgc/two_phase_runtime.py tests/test_two_phase_runtime.py
git commit -m "feat: add two-phase runtime signal extraction"
```

### Task 2: Deterministic guideline thresholds and geometry audit

**Files:**
- Create: `dvgc/two_phase_guideline.py`
- Create: `tests/test_two_phase_guideline.py`

**Interfaces:**
- Consumes: `ReferenceTrajectory`, config/file hashes, Task 1 geometry/signals, and Gate A thresholds.
- Produces: `GuidelineMargins`, `GuidelineSelection`, `build_threshold_manifest`, `canonical_manifest_hash`, `select_guideline_indices`, `reconstruct_guideline_state`, `audit_geometry_clearance`, and `validate_guideline_event_order`.

- [ ] **Step 1: Write failing fixed-index and threshold tests**

Use a small literal DataFrame to assert front/middle/back launch selection, Apex pre/nearest/post selection by vertical-speed sign/nearest absolute value, early-descent selection, and recovery slice selection. Assert thresholds equal hand-calculated extrema plus named literal margins and cannot consume labels, rewards, policy results, or retry feedback.

- [ ] **Step 2: Verify red and implement deterministic selection/thresholds**

Run `tests/test_two_phase_guideline.py`, implement only the fixed selection and arithmetic, then rerun green.

- [ ] **Step 3: Add failing manifest stability and provenance tests**

Assert two builds with the same XML/reference/config/code/margins/seed have identical canonical JSON and SHA-256. Change each authoritative input and assert the hash changes. Require feature definitions/units, raw extrema, selected thresholds, source categories, controller provenance, geometry hash, and creation seed. Reject any expert/trained-policy vocabulary or label/result field.

- [ ] **Step 4: Implement canonical manifest and provenance validation**

Use sorted-key compact JSON and normalized finite Python scalars. Hash actual XML, CSV, config, Task 1 source, and geometry manifest. Run tests green.

- [ ] **Step 5: Add failing host cross-audit and event-order tests**

For representative real states, assert audit rows include JAX clearance, `mj_geomDistance`, absolute difference, sign agreement, and nearest pair. Use literal event ticks to detect missing/inverted events and insufficient Apex/recovery width.

- [ ] **Step 6: Implement host audit and event report**

Keep all MuJoCo calls in host-only functions. Return `gate_pause` on coverage/sign/order failure. Run Task 2 tests.

- [ ] **Step 7: Commit Task 2**

```bash
git add -- dvgc/two_phase_guideline.py tests/test_two_phase_guideline.py
git commit -m "feat: add reproducible two-phase guideline contracts"
```

### Task 3: Full v4 guideline banks and CLI

**Files:**
- Modify: `dvgc/two_phase_guideline.py`
- Create: `cli/build_two_phase_guideline_banks.py`
- Modify: `tests/test_two_phase_guideline.py`

**Interfaces:**
- Consumes: Task 2 selections/manifest, `OrangeBikeDVGC`, `snapshot_record_v4`, `validate_snapshot_v4`, `validate_phase_snapshot`, and `SnapshotBank`.
- Produces: ignored Phase U/D bank files, manifests, event/geometry reports, and a closed Gate B build report.

- [ ] **Step 1: Write failing bank-record tests**

Build a tiny real-runtime fixture and assert every record is v4, contains an explicit two-phase context, retains only the mechanically necessary top-level legacy phase, and validates through both v4 and overlay validators. Assert new code refuses a missing explicit method phase and never derives it from legacy phase.

- [ ] **Step 2: Verify red and implement continuous-history capture**

Reconstruct a proposal from reference pose/joints/velocity, then advance consecutive real control ticks with reference actions until FIFO valid equals three. Capture only through `snapshot_record_v4`; validate history ordering, last action, ctrl timing, actor observation, and timing contract before admission.

- [ ] **Step 3: Add failing coverage, perturbation, and provenance tests**

Assert U contains launch front/middle/back, D contains Apex pre/nearest/post plus early descent, parent/trajectory ids are stable, perturbations are deterministic/small, and reference index/time is provenance only. Assert controller naming cannot include expert/`pi_up`/`pi_down`/trained-policy language.

- [ ] **Step 4: Implement bank assembly and validators**

Use a fixed nominal plus explicitly declared small perturbation table. Save `SnapshotBank` payloads only after all records pass. Record exact construction transitions separately from all training budgets.

- [ ] **Step 5: Add failing CLI behavior tests**

Invoke the CLI against a temporary output root. Assert it writes the run manifest before dynamics, refuses nonempty/incompatible outputs, produces stable filenames/hashes, reports stopping conditions/cost, and enters `gate_pause` without threshold relaxation on a failed core gate.

- [ ] **Step 6: Implement CLI orchestration and verify green**

Provide explicit `--config`, `--reference`, `--output`, `--seed`, and bounded representative-count arguments. Never expose label/training knobs. Run Task 2/3 tests.

- [ ] **Step 7: Commit Task 3**

```bash
git add -- dvgc/two_phase_guideline.py cli/build_two_phase_guideline_banks.py tests/test_two_phase_guideline.py
git commit -m "feat: build guideline thresholds and initial banks"
```

### Task 4: Authoritative two-phase snapshot round-trip

**Files:**
- Create: `dvgc/two_phase_roundtrip.py`
- Create: `tests/test_two_phase_snapshot_roundtrip.py`
- Modify: `cli/build_two_phase_guideline_banks.py`

**Interfaces:**
- Consumes: timing-explicit v4 rows, explicit `restore_snapshot_mode`, Task 1 signal/event extraction, and real `env.step`.
- Produces: `compare_two_phase_roundtrip` and a closed report with continuous differences, exact discrete equality, and identifiers.

- [ ] **Step 1: Write failing explicit-restore and comparison tests**

Patch the compatibility fallback to raise and verify the round-trip still works through explicit independent reconstruction. Deliberately perturb qpos, ctrl, last action, actor/privileged observation, FIFO/history, support/event latch, terminal flags, and two-phase signal values one at a time; each must name the failed field.

- [ ] **Step 2: Verify red and implement comparison primitives**

Implement field-specific max-absolute differences and exact discrete comparisons with immutable named tolerances. Include snapshot and parent ids in every row.

- [ ] **Step 3: Add failing real one-to-three-tick replay tests**

Capture a real v4 state, form original/restored branches with the same PRNG seed/action count, and compare all required fields plus Task 1 signals/event state. Require representative labels `pre`, `nearest`, `post`, and `boundary` in selection.

- [ ] **Step 4: Implement real replay and integrate CLI**

Call only `restore_snapshot_mode(..., timing_explicit_independent_reconstruction)`. Synchronize device results before the next representative. Persist a compact report and set `gate_pause` on any failed field.

- [ ] **Step 5: Run all new targeted tests green**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_two_phase_runtime.py \
  tests/test_two_phase_guideline.py \
  tests/test_two_phase_snapshot_roundtrip.py
```

- [ ] **Step 6: Commit Task 4**

```bash
git add -- dvgc/two_phase_roundtrip.py cli/build_two_phase_guideline_banks.py tests/test_two_phase_snapshot_roundtrip.py
git commit -m "test: validate two-phase dynamic snapshot contracts"
```

### Task 5: Runtime fingerprint and real Gate B artifact build

**Files:**
- Modify: `cli/runtime_gate.py`
- Modify: an existing runtime-gate test or `tests/test_two_phase_snapshot_roundtrip.py`
- Generate ignored: `runs/two_phase/gate_b_<run_id>/...`

**Interfaces:**
- Consumes: all Gate B source closure and CLI.
- Produces: a fresh runtime fingerprint/gate report plus reproducible real bank artifacts.

- [ ] **Step 1: Write failing fingerprint-closure test**

Assert changing any Gate B runtime/guideline/roundtrip/CLI source changes `source_fingerprint`, without source-text-only assertions.

- [ ] **Step 2: Add Gate B sources to runtime fingerprint and verify green**

Make the minimal closure edit in `cli/runtime_gate.py`; do not alter runtime behavior or PPO accounting.

- [ ] **Step 3: Record and execute the bounded Gate B build**

Create a unique ignored output directory and pre-run manifest, then run the CLI once with the approved deterministic seed and fixed bounds. Do not retry with relaxed thresholds. Record actual nontraining construction transitions and all artifact hashes/counts.

- [ ] **Step 4: Audit artifacts read-only**

Reload both banks, recompute manifest/bank hashes, re-run record validators, inspect event order/Apex width/recovery hold/round-trip failures, and classify PASS or `gate_pause`.

- [ ] **Step 5: Commit fingerprint/test change only if validated**

```bash
git add -- cli/runtime_gate.py tests/test_two_phase_snapshot_roundtrip.py
git commit -m "test: include gate b runtime in fingerprint"
```

### Task 6: Required verification, documentation, and stop

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: committed Gate B source, ignored artifacts, watchdog state, and all verification reports.
- Produces: final recoverable Gate B state and no Gate C action.

- [ ] **Step 1: Run required source verification**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_two_phase_semantics.py \
  tests/test_feasibility.py \
  tests/test_training_budget.py \
  tests/test_two_phase_runtime.py \
  tests/test_two_phase_guideline.py \
  tests/test_two_phase_snapshot_roundtrip.py
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
```

- [ ] **Step 2: Recheck watchdog and run fresh runtime gate**

Require timer disabled/inactive and service inactive, then run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate
```

Record the runtime gate's actual smoke transitions separately; formal training transitions remain zero.

- [ ] **Step 3: Update experiment state**

Record branch/HEAD lineage, watchdog pre/commands/post state, restore command, threshold manifest path/hash, bank paths/hashes/counts, Apex width, round-trip status, tests/preflight/runtime gate, construction/runtime-smoke/formal transition accounting, blockers, and separately reviewed Gate C as the only possible next action.

- [ ] **Step 4: Verify final tree and commit documentation**

Re-run affected document/current-state tests, `git diff --check`, and the full required verification if the source fingerprint changed after the last run. Explicitly stage only `docs/EXPERIMENT_STATE.md` and any still-uncommitted validated source/test paths.

```bash
git add -- docs/EXPERIMENT_STATE.md
git commit -m "docs: record gate b completion"
```

- [ ] **Step 5: Stop**

Report Gate B PASS or `gate_pause`, branch, HEAD, changed files, contracts, hashes/counts, all verification, runtime-smoke transitions, formal training transitions zero, blockers, and next permitted action. Do not create expert-training code or start Gate C.
