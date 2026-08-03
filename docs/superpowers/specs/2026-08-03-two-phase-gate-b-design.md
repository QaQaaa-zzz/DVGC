# Two-Phase Gate B Runtime and Guideline-Bank Design

## Status and claim boundary

This document is the approved Gate B implementation contract. Gate B connects
the Gate A two-phase semantics to real immutable-model state, constructs
reproducible guideline-derived thresholds and initial reset banks, and validates
timing-explicit replay. It does not train an expert or feasibility model, label
continuations, build a learned soft Tube, create unified PPO, or support a
JCE/JEL claim.

The implementation branch is `agent/two-phase-soft-tube`, created from
`5331896bee08a920321a9b39b496f66c7b9b0879`.

## Safety interlock

Before design work, the user systemd state was read as timer disabled, timer
inactive, and service inactive. The authorized commands were reapplied:

```bash
systemctl --user disable --now dvgc-pipeline-watchdog.timer
systemctl --user stop dvgc-pipeline-watchdog.service || true
```

Post-state was again disabled/inactive/inactive. Unit files were neither
modified nor deleted. `runs/ACTIVE_PIPELINE.json` was absent, so no new pointer
backup was necessary. The retained legacy pointer backup documented in
`docs/EXPERIMENT_STATE.md` remains unchanged. Restoration still requires
separate authorization and uses:

```bash
systemctl --user enable --now dvgc-pipeline-watchdog.timer
```

## Runtime architecture

The formal online path is:

```text
state.data/state.info + immutable XML geometry
  -> pure JAX dvgc.two_phase_runtime adapter
  -> ApexBandSignals / RecoverySignals / TwoPhaseEventState
```

The adapter is external to `OrangeBikeDVGC`. It does not modify `env.step`,
`state.info`, reward, termination, observation, reset, action mapping, XML, or
matcher behavior. Its public extraction and event-transition functions accept
scalar or batched MJX state leaves and must pass both `jax.jit` and `jax.vmap`.
They never convert traced values to Python booleans and never mutate inputs.

`dvgc/env.py` remains unchanged by default. If a red test proves that an
immutable static quantity cannot be obtained safely from the current model or
config, implementation pauses and reports the exact missing quantity before
any additive environment edit.

Host MuJoCo is an audit path only. `mj_geomDistance` may cross-check a small
fixed set of representative states, but it cannot provide training signals,
online events, or bulk bank-construction features.

## Immutable geometry schema

`TwoPhaseGeometry` is built once on the host from the authoritative XML/model
and then contains only immutable numeric arrays/scalars suitable as JAX static
inputs:

```text
robot geom ids, names, types, sizes, body ids
collision participation flags
wheel/body partitions
obstacle geom id, front/back/top/side bounds
floor height
contact/support tolerances
joint/root/body ids needed by extraction
```

A geometry manifest records, for every XML geom:

```text
geom name
geom type
collision participation
body ownership
robot/terrain/wheel classification
supported JAX support-boundary formula or visual-only reason
```

Every collision-relevant robot geom must map to a supported analytic formula.
The authoritative model currently uses collision boxes, cylinders, and
ellipsoids; visual meshes have `contype=conaffinity=0`. Box support uses the
absolute rotation matrix times half extents. Cylinder support combines its
rotated axis half-length and radial support. Ellipsoid support uses the norm of
the world query direction transformed into the ellipsoid frame and scaled by
its radii. Sphere and capsule formulas are implemented if the audited manifest
contains those collision types. A collision mesh is a hard construction error
unless support is implemented; a noncolliding visual mesh is recorded but
excluded with proof from its collision flags.

The name `full_structure_clearance` is emitted only after the manifest validator
proves complete formula coverage. It is the minimum world-z support boundary of
all collision-relevant robot geoms minus the authoritative obstacle top. It is
not root height or CoM height.

`robot_frontmost_x` is the maximum world-x support boundary over the same
complete collision-relevant robot geom set. The fixed sign convention is:

```text
obstacle_relative_x = obstacle_front_x - robot_frontmost_x
> 0: robot front has not reached the obstacle front
= 0: robot front aligns with the obstacle front
< 0: robot front has passed the obstacle front
```

Representative host audits report JAX clearance, MuJoCo geom distance,
absolute difference, sign agreement, and nearest robot/terrain geom pair. A
missing coverage proof or sign disagreement sets Gate B to `gate_pause`.

## Physical signals

`extract_apex_band_signals` derives stable-airborne status from deployable
airborne history, CoM/root vertical velocity, full-structure clearance,
roll/pitch, angular-speed norm, forward velocity, obstacle-relative x,
geometry/contact illegality, and physical-failure state. It consumes neither
reward, legacy oracle phase, matcher distance, reference time, nor reference
index.

`extract_recovery_signals` derives current legal wheel support, landing-region
validity, absence of body contact, pose/rate/speed values, the previous
two-phase recovery-hold count, and physical failure. Wheel support and body
legality use immutable geometry plus deployable IMU/state history so the
default Warp/IMU runtime does not depend on private contact buffers. The Gate A
`advance_recovery_hold_count` function owns the consecutive-count transition;
one instantaneous support tick cannot satisfy recovery.

`TwoPhaseEventState` contains only two-phase event latches/counters. The pure
`extract_two_phase_events` function returns a new value without writing into
the environment state. Its declared event order is:

```text
jump_window_entered -> liftoff_seen -> stable_airborne -> ascending
-> apex_band_entered -> descending -> pre_landing -> first_valid_contact
-> impact_absorbing -> stable_recovery
```

Events use current and previous physical signals. Legacy phase ids may be
retained inside the v4 restore payload but never infer a formal two-phase label.

## Guideline thresholds and provenance

`dvgc.two_phase_guideline` loads `data/reference_jump.csv`, validates its
columns/timing, reconstructs deterministic physical proposals under the
authoritative XML, and extracts fixed physical slices for the launch window,
Apex pre/nearest/post, early descent, first legal contact, and recovery hold.
It does not use success labels, continuation labels, feasibility scores, soft
Tube membership, final-policy results, or rollout-tuned acceptance.

The guideline CSV vertical coordinate is an envelope-relative legacy frame,
not the authoritative XML free-root coordinate. Reconstruction uses the one
fixed mapping

```text
root_z = nominal_base_z_ground + (reference_z - reference_initial_z)
```

and records both origins and the formula. Ground Phase U proposals and the
event-trace initial state then undergo MuJoCo-native vertical support placement
without changing posture. Placement must remain inside the existing solver
limit, provide wheel support, and produce no body-terrain contact. There is no
rollout search over vertical offsets or alternative initial rows.

Threshold selection is a deterministic function of raw reconstructed
guideline extrema, immutable model geometry, existing fixed physical limits,
and named engineering margins. Fixed slices and margins are part of the input
contract; the CLI cannot retry with relaxed values when samples fail. The
threshold manifest contains:

```text
contract version
XML and guideline paths plus SHA-256
resolved config SHA-256
action-mapping version
extraction code version and SHA-256
geometry-manifest SHA-256
physical feature definitions and units
reference anchors and fixed slices
raw physical extrema
selected Apex/Recovery thresholds
named engineering margins and source category
controller provenance
creation seed
canonical manifest hash
```

Canonical JSON serialization with sorted keys and stable float encoding makes
the manifest hash reproducible.

## Guideline controller and bank construction

`cli.build_two_phase_guideline_banks` is the single Gate B entrypoint. The
reference open-loop action sequence is recorded only as controller provenance
and reference rollout source. It is never named an expert, `pi_up`, `pi_down`,
trained policy, or learned controller.

The Phase U bank contains deterministic front/middle/back launch-window states
plus a small declared set of physically legal perturbations. The Phase D bank
contains Apex pre/nearest/post and early-descent states with transition-band
thickness. No large random expansion occurs.

Each record is a full v4 timing-explicit snapshot. Its top-level legacy phase
is retained solely for current `SnapshotBank`/restore compatibility. The
authoritative method identity is exclusively:

```text
two_phase_context.source_phase = propulsion_ascent | descent_recovery
```

New two-phase code accepts that explicit field and never derives it from the
top-level `takeoff`, `flight`, or `landing` value. Reference time/index is
provenance only.

Before capture, every proposal advances through consecutive real control ticks
`t-2 -> t-1 -> t`. The resulting actor packet FIFO, pre/current/post history,
last action, control timing, actor observation, and field ticks are produced by
the environment. Copying a current frame, constructing history from isolated
CSV rows, or mixing post-update history with the current frame is forbidden.
Each row passes `validate_snapshot_v4` and Gate A
`validate_phase_snapshot` before saving.

Outputs live under `runs/two_phase/gate_b_<run_id>/` and are ignored by Git:

```text
run_manifest.json
geometry_manifest.json
threshold_manifest.json
phase_up_guideline_bank.pkl
phase_down_guideline_bank.pkl
guideline_event_report.json
geometry_cross_audit.json
snapshot_roundtrip_report.json
gate_b_report.json
```

The run manifest is written before dynamics and records purpose, inputs,
maximum nontraining interaction cost, stopping conditions, and output path.

## Timing-explicit round-trip

Round-trip uses only
`restore_snapshot_mode(..., observation_mode="timing_explicit_independent_reconstruction")`.
The compatibility `restore_snapshot` fallback is forbidden.

At least `pre`, `nearest`, `post`, and `boundary` representatives are selected
from each applicable bank. Original and restored branches use the same
snapshot, PRNG seed, action sequence, and one-to-three control ticks. Reports
compare:

```text
qpos, qvel, ctrl, last action
actor observation and privileged observation
three-frame packet/history state
contact/support and event latches
termination and truncation
two-phase signals and event state
```

Continuous fields report maximum absolute difference against named tolerances;
discrete fields require exact equality. Reports include snapshot id, parent
trajectory id, failed fields, and both branch traces. No qpos/qvel-only result
can pass.

## Gate outcomes and validation

The guideline trace reports first control tick for every declared event, Apex
band consecutive width, pre/nearest/post counts, stable-recovery hold, and all
missing or inverted events. If the guideline cannot satisfy the approved Gate
A physical contract, the result is `gate_pause`; thresholds are not weakened
and legacy phase ids are not substituted. The CLI writes geometry, threshold,
support, event, and interaction-cost evidence before stopping, and it does not
create either bank after an event-trace failure.

Required source verification is:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_two_phase_semantics.py tests/test_feasibility.py \
  tests/test_training_budget.py tests/test_two_phase_runtime.py \
  tests/test_two_phase_guideline.py tests/test_two_phase_snapshot_roundtrip.py
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate
```

The runtime gate's 64+32 transition PPO is only compile/update/resume smoke.
The report separates actual runtime-smoke transitions from formal training
transitions, which remain zero.

Gate B passes only if watchdog state remains disabled/inactive, geometry
coverage and cross-audit pass, guideline event order is valid, manifests are
reproducible, both banks are nonempty, Apex pre/nearest/post exist, all rows
pass both snapshot validators, full round-trip passes, all test/preflight gates
pass, runtime gate passes, and formal training transitions equal zero. Any core
failure produces `gate_pause`. Completion updates `docs/EXPERIMENT_STATE.md`,
commits only source/tests/docs, and stops without entering Gate C.
