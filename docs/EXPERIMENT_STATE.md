# DVGC Experiment State

## Current method

The approved research direction is the two-phase learned soft-feasibility-Tube
method defined in `PROJECT.md` and `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`:
`Propulsion-Ascent` -> Apex transition band -> `Descent-Recovery`, followed by
`V_up`/`V_down`, soft-Tube guidance, one unified Tube-RSI PPO, and independent
frozen-policy JCE/JEL evaluation.

Gate A static contracts and the Gate B external pure-JAX runtime adapter,
geometry manifest/audit, deterministic guideline threshold/bank builder, and
timing-explicit round-trip verifier are implemented. Gate B is nevertheless
frozen at `gate_pause` because the authoritative guideline open-loop sequence
does not satisfy the required real-environment event order. Existing
five-stage code and results remain legacy migration sources only.

This method is not yet implemented end-to-end: there are no trained phase
experts, feasibility models, learned soft Tubes, or unified two-phase policy.

## Current branch and commit

- Branch: `agent/two-phase-soft-tube`
- Gate B baseline: `5331896bee08a920321a9b39b496f66c7b9b0879`
- Gate B implementation head: `387ae59`
- Failure-video audit implementation head: `5b8fe73`
- Gate A implementation: `5e5da3b`
- Cleanup baseline: `main@b7bb815`
- Dependency design: `27f0aa3`
- Clarified design policy: `3cbd6c1`
- Validated cleanup commits through `3d6a6b5`; resolve the containing final
  validation commit with `git rev-parse HEAD`.

## Last validated gate

- Current reusable runtime gate: PASS at source fingerprint
  `186dcfa5383632a4227bc0e56fe0d7df98350b46bec5d77c1f2e9567996e1345`.
- It used the fresh ignored work directory
  `runs/runtime_gate_failure_video_exact_closure_20260803/` and took 99.345
  seconds.
- The gate's 64+32 = 96 transitions are compile/update/resume smoke evidence
  only, not expert, pilot, or formal training.

## Current active run

None. Gate B stopped at its fixed guideline event gate. The authoritative
event probe used 12 environment transitions; the runtime integrity gate used
96 smoke transitions. Expert, pilot, learnability, and formal training
transitions remain exactly 0.

## Pipeline automation safety interlock

On 2026-08-03 the live user-systemd state was inspected before any two-phase
implementation work. `dvgc-pipeline-watchdog.timer` was still enabled and
active, and `runs/ACTIVE_PIPELINE.json` still had status `ACTIVE` for
`scripts/start_corrected_apex_unified_rsi_followons.sh`. The watchdog source
also retains its legacy Descent-Tube fallback.

The timer was stopped and disabled without changing or deleting its unit files,
watchdog source, or referenced legacy scripts. The active pointer was preserved
byte-for-byte as
`runs/ACTIVE_PIPELINE.legacy-disabled-20260803T134842+0800.json` (SHA-256
`6546dc01d7d2579d733fcbf39b3544942467232dfec2a25319e078a24b3bfae4`), and
the default `runs/ACTIVE_PIPELINE.json` path is absent. The complete ignored
operations record is under
`runs/operations/watchdog_deactivation_20260803T134842+0800/`.

Restoring the retired automation requires an explicit decision followed, from
the repository root, by:

```bash
mv -- runs/ACTIVE_PIPELINE.legacy-disabled-20260803T134842+0800.json runs/ACTIVE_PIPELINE.json
systemctl --user enable --now dvgc-pipeline-watchdog.timer
```

## Current inputs and hashes

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c`
- Payload: 4.0 kg
- Hip/knee limits: +/-50 N m
- Action order: `[steer, rear-wheel drive, hip, knee]`
- Runtime: `/home/qy/mujoco_playground/.venv/bin/python`

## Latest result

Gate B is `gate_pause`, not passed. The formal path is the external pure-JAX
adapter over `state.data`/`state.info` plus immutable XML geometry; no latch was
added to `env.step`. All collision-relevant robot geoms are covered by the
geometry manifest, representative host `mj_geomDistance` cross-audit passed,
and the fixed threshold manifest was produced reproducibly at:

```text
runs/two_phase/gate_b_20260803_authoritative_pause/threshold_manifest.json
file SHA-256: c2a1cf6b17910d8912b5b9de282a4ddfe601467c77f267780d22b068451b0e8f
canonical hash: f8b06896f46c3da68c861058e203fb25752b60a581539185f7012fd97bee493a
```

The real guideline event trace began from a MuJoCo-audited grounded state
(wheel support present, no body-terrain contact), then terminated after 12
control ticks with `end_code=9` (`prelaunch_airborne`). All ten required events
were missing; Apex consecutive width and stable-recovery hold were both 0.
The open-loop sequence becomes airborne before the legal jump-window entry, so
the CLI stopped before bank admission. Consequently:

- no authoritative Phase U bank was written;
- no authoritative Phase D bank was written;
- no Gate B bank round-trip report was run or written;
- the implemented builder and round-trip contracts remain covered by dynamic
  tests but are not promotion evidence.

## Dynamic failure video evidence

Gate B failures now preserve videos after the physical event report is closed
and before the original Gate-pause exception is raised. Rendering consumes
captured qpos/qvel/ctrl only and cannot advance or influence environment
dynamics. A renderer failure is recorded separately and cannot mask or weaken
the original physical failure.

The current ignored audit directory is:

```text
runs/two_phase/gate_b_20260803_failure_videos_exact_timing/
```

- `full_guideline_prelaunch_airborne.mp4`: 190,794 bytes, SHA-256
  `81ef246869bb585c8ea00ef00aec30af1060d062c123cc16788234c3272c7b48`,
  12 environment transitions, terminal `prelaunch_airborne`.
- `full_guideline_prelaunch_airborne.states.npz`: SHA-256
  `580153b43956eae2e2b4f43cad6a757b581e3cdd966e88befa3e308af5796aee`.
- `launch_history_airborne_before_window.mp4`: 158,374 bytes, SHA-256
  `dc18b0af241b320a811723a8fc366e74389f74106f70813b1dd5d5d6a6117954`,
  8 environment transitions, nonterminal contract failure.
- `launch_history_airborne_before_window.states.npz`: SHA-256
  `77d1a0dd70b3d07202f87c73e86a6c84a1f7b8ba1a758c8ab1851dbaf2a66a96`.

The launch-history audit uses the formal timing contract exactly: state origin
83, initial `ctrl`/`last_action` from reference index 73, then control actions
83, 93, and 103. The earlier manually rendered launch-history video used a
one-action-shifted schedule and is superseded; it is not evidence.

At the first root-position window sample (tick 6), the host contact audit still
sees one wheel contact, but the deployable IMU/support estimator is false and
the jump latch remains false. The frontmost-geometry event latch follows at
tick 7 and liftoff is observed at tick 8. Both contact signals are shown
independently in the overlay; the failure is not described as simple geometric
loss of all contact.

Final source validation on 2026-08-03:

- `python -m compileall dvgc cli`: passed.
- Gate A/B targeted files: 229 passed.
- Failure-video/guideline/round-trip targeted files: 57 passed.
- Full pytest: 791 passed, with one existing JAXopt deprecation warning.
- `bash scripts/local_preflight.sh`: passed and repeated 791 tests.
- Fresh runtime gate: PASS, 96 smoke transitions.
- Guideline event audit: 12 environment transitions, `gate_pause`.
- Formal training transitions: 0.

Legacy five-stage experimental outcomes are not evidence for the dynamic
two-phase method and must not be promoted retrospectively.

## Known blockers

The Gate B physical blocker is the mismatch between the guideline open-loop
sequence and the authoritative 4 kg, +/-50 N m model: the robot becomes
airborne before the legal jump-window entry. The approved contract forbids
lowering physical thresholds, substituting legacy phases, or searching rollout
offsets until the trace passes.

There are still no authoritative Gate B Phase U/Phase D banks, trained/frozen
phase experts, continuation-labeled two-phase snapshots, `V_up`/`V_down`,
learned soft Tubes, unified two-phase PPO, or independent frozen-final-policy
evaluation. The watchdog is disabled/inactive, its service is inactive, and
`runs/ACTIVE_PIPELINE.json` is absent.

## Next permitted action

Stop at Gate B `gate_pause`. The next permitted action is a separately reviewed
decision about the non-expert guideline/controller mismatch. Gate C, expert
training, snapshot labeling, feasibility training, soft-Tube construction, and
unified PPO are not authorized from this marker.

## Closed routes

- Landing -> Flight -> Takeoff -> Approach sequential shared-Actor bootstrap
- exhaustive H1/C_L A/B
- roll-targeted shared-Actor retention
- sequential Flight retention repair
- final-shared v1/v2 and corrected Apex follow-ons
- five-stage phase-balanced unified RSI controllers
- universal 4 -> 8 -> 16/32 requirement for training Tubes
