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

## Current branch and commit

- Branch: `agent/two-phase-soft-tube`
- Gate B baseline: `5331896bee08a920321a9b39b496f66c7b9b0879`
- Gate B implementation head: `387ae59`
- Gate A implementation: `5e5da3b`
- Cleanup baseline: `main@b7bb815`
- Dependency design: `27f0aa3`
- Clarified design policy: `3cbd6c1`
- Validated cleanup commits through `3d6a6b5`; resolve the containing final
  validation commit with `git rev-parse HEAD`.

## Last validated gate

- Current reusable runtime gate: PASS at source fingerprint
  `84f61c8fe3be360cc0982f2ec75de0e7a3e1b33721b49eed8262511900301ce6`.
- It used the fresh ignored work directory
  `runs/runtime_gate_gate_b_20260803/` and took 99.318 seconds.
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

Final source validation on 2026-08-03:

- `python -m compileall dvgc cli`: passed.
- Gate A/B targeted files: 229 passed.
- Full pytest: 781 passed, with one existing JAXopt deprecation warning.
- `bash scripts/local_preflight.sh`: passed and repeated 781 tests.
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
