# DVGC Experiment State

## Current method

The approved research direction is the two-phase learned soft-feasibility-Tube
method defined in `PROJECT.md` and `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`:
`Propulsion-Ascent` -> Apex transition band -> `Descent-Recovery`, followed by
`V_up`/`V_down`, soft-Tube guidance, one unified Tube-RSI PPO, and independent
frozen-policy JCE/JEL evaluation.

Gate A static contracts and the Gate B external pure-JAX runtime adapter,
geometry manifest/audit, deterministic guideline threshold builder, natural
reset audit, and timing-explicit snapshot/round-trip contracts are implemented.
The revised Gate B no longer requires a complete guideline open-loop event
sequence or guideline-generated banks. `data/reference_jump.csv` is a
kinematic guideline and weak prior, not an expert or authoritative controller.
Existing five-stage code and results remain legacy migration sources only.

This method is not yet implemented end-to-end: there are no trained phase
experts, feasibility models, learned soft Tubes, or unified two-phase policy.

## Current branch and commit

- Branch: `agent/two-phase-soft-tube`
- Gate B baseline: `5331896bee08a920321a9b39b496f66c7b9b0879`
- Gate B implementation head: `387ae59`
- Failure-video audit implementation head: `5b8fe73`
- Prelaunch-continuation design: `bac2a93`, `691ad8e`, `ec90d6d`
- Prelaunch-continuation implementation: `23a746e`, `6ed2cdc`, `0b86435`
- Gate A implementation: `5e5da3b`
- Cleanup baseline: `main@b7bb815`
- Dependency design: `27f0aa3`
- Clarified design policy: `3cbd6c1`
- Validated cleanup commits through `3d6a6b5`; resolve the containing final
  validation commit with `git rev-parse HEAD`.

## Last validated gate

- Current reusable runtime gate: PASS at source fingerprint
  `fb5839508d93696bc782c87ebf218c622c8fd8955a4da682e1bc228a88ce0fc7`.
- It used the fresh ignored work directory
  `runs/runtime_gate_prelaunch_final_20260803/` and took 95.976
  seconds.
- The gate's 64+32 = 96 transitions are compile/update/resume smoke evidence
  only, not expert, pilot, or formal training.

## Current active run

None. The historical Gate B guideline event probe used 17 environment
transitions; its two outcome-video diagnostics used 25 transitions in total.
Those runs are retained provenance, not active work or a revised Gate B pass
condition. The runtime integrity gate used
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

Gate B has been revised and accepted as a runtime/state-foundation gate. The
formal path is the external pure-JAX adapter over `state.data`/`state.info` plus
immutable XML geometry; no latch was added to `env.step`. All
collision-relevant robot geoms are covered by the geometry manifest,
representative host `mj_geomDistance` cross-audit passed, the natural start is
valid, and the fixed threshold manifest was produced reproducibly at:

```text
runs/two_phase/gate_b_20260803_prelaunch_continuation/threshold_manifest.json
file SHA-256: eb7b517e5fb0bd4d49f90fd93f9223d4056718f9329e07b5400c35aa49119387
canonical hash: 2886802483c77a6d13817cbce889e9fec24807be2b536ba1429cdb7c6aeff900
```

The historical real guideline event trace began from a MuJoCo-audited grounded state
(wheel support present, no body-terrain contact). Early airborne was retained
as telemetry and did not terminate the rollout. The environment's monotonic
jump latch became true on the terminal tick, but the rollout then triggered the
unchanged roll safety limit after 17 control ticks with `end_code=4`
(`roll_limit`). The external two-phase adapter does not admit an event on a
physical-failure tick, so no valid Apex or stable recovery occurred. This
result proves that full open-loop dynamic compatibility with the current 4 kg,
+/-50 N m model was not demonstrated; it does not block expert training. The
CLI stopped before bank admission. Consequently:

- no authoritative Phase U bank was written;
- no authoritative Phase D bank was written;
- no real Gate B bank round-trip report was run or written;
- the implemented builder and round-trip contracts remain covered by dynamic
  tests but are not claims that an expert bank exists.

## Dynamic failure video evidence

Gate B failures now preserve videos after the physical event report is closed
and before the original Gate-pause exception is raised. Rendering consumes
captured qpos/qvel/ctrl only and cannot advance or influence environment
dynamics. A renderer failure is recorded separately and cannot mask or weaken
the original physical failure.

The current ignored audit directory is:

```text
runs/two_phase/gate_b_20260803_prelaunch_continuation/failure_videos/
```

- `full_guideline_continuation.mp4`: 238,955 bytes, SHA-256
  `08df9e45f7c711ed63cade78dd8aa8d25e5da4e384a55cf9723b7951a45a0ce5`,
  17 environment transitions, terminal `roll_limit` (`end_code=4`).
- `full_guideline_continuation.states.npz`: SHA-256
  `2109bacfbecdaabdf63d833e61a574be03ea0bf2e2347fd5641b9548ee267823`.
- `launch_history_window_latch.mp4`: 157,826 bytes, SHA-256
  `1ea9ebb58a438d922cf7f2c2144f9f8385ea2e03b25532f0965e9e1c3bb62e02`,
  8 environment transitions, nonterminal outcome diagnostic.
- `launch_history_window_latch.states.npz`: SHA-256
  `77d1a0dd70b3d07202f87c73e86a6c84a1f7b8ba1a758c8ab1851dbaf2a66a96`.
- `failure_video_manifest.json`: SHA-256
  `e8f4fc62f7370b165da4c56a4e895d6f8cb5e617aaaeac67940584ecef742e84`.

The run was executed at producer HEAD
`0b8643521797eac2d38ff52b8f108cda5fb6d283`. Because the pause-path manifest
version used by that run recorded only input hashes, the ignored
`producer_provenance.json` is an explicitly labeled post-run execution record
that binds that HEAD's env, reward, failure-video, runtime, guideline, and
builder hashes to the event-report, threshold-manifest, video-manifest, and
video-status hashes. It is not presented as an in-run signed record. The
current builder closes these producer hashes directly in all future pause
manifests.

The launch-history audit uses the formal timing contract exactly: state origin
83, initial `ctrl`/`last_action` from reference index 73, then control actions
83, 93, and 103. The earlier manually rendered launch-history video used a
one-action-shifted schedule and is superseded; it is not evidence.

At the first root-position window sample (tick 6), the host contact audit still
sees one wheel contact, but the deployable IMU/support estimator is false. The
root-position window event enters the legal window and the environment jump
latch becomes true without requiring wheel support; it remains true after the
window.
Liftoff is observed at tick 7. Both contact signals are shown independently in
the overlay. This diagnostic is nonterminal and does not assert Phase U success.

Final source validation on 2026-08-03:

- `python -m compileall dvgc cli`: passed.
- Prelaunch semantics/reward/failure/guideline/round-trip targeted files:
  115 passed.
- Full pytest: 811 passed, with one existing JAXopt deprecation warning.
- `bash scripts/local_preflight.sh`: passed and repeated 811 tests.
- Fresh runtime gate: PASS, 96 smoke transitions.
- Three source-fingerprint refreshes were required during review closure: 288
  cumulative engineering smoke transitions this round, all excluded from the
  formal-training total.
- Guideline event audit: 17 environment transitions, `gate_pause` at retained
  `roll_limit`.
- Outcome-video diagnostics: 25 environment transitions, rendering PASS and
  original Gate status still `gate_pause`.
- Formal training transitions: 0.

Legacy five-stage experimental outcomes are not evidence for the dynamic
two-phase method and must not be promoted retrospectively.

## Known blockers

The retained `END_ROLL_LIMIT=4` outcome is not a revised Gate B blocker. It
shows only that the reference open-loop actions were not proven dynamically
compatible with the current model. No further reference-action repair is
planned, and roll/pitch/contact/nonfinite safety limits remain unchanged.

There are still no authoritative Gate B Phase U/Phase D banks, trained/frozen
phase experts, continuation-labeled two-phase snapshots, `V_up`/`V_down`,
learned soft Tubes, unified two-phase PPO, or independent frozen-final-policy
evaluation. The watchdog is disabled/inactive, its service is inactive, and
`runs/ACTIVE_PIPELINE.json` is absent.

Phase U must train from audited natural starts and earn the complete Apex-band
success contract. Phase D preliminary candidates require the physical seed
validation protocol and cannot be called reachable or safe. Its formal reset
distribution must be sourced primarily from real frozen-`pi_up` Apex and early
descent rollouts. Neither phase-expert training entrypoint exists yet.

## Next permitted action

Stop after the Gate B revision and Gate C expert-entry design commit. The next
permitted action is review and implementation of Gate C1's stable Phase U CLI
and PPO smoke capability. This marker does not authorize running smoke, a
102,400-transition pilot, 500,000-transition formal training, Phase D training,
snapshot labeling, feasibility training, Soft Tube construction, or unified
PPO.

## Closed routes

- Landing -> Flight -> Takeoff -> Approach sequential shared-Actor bootstrap
- exhaustive H1/C_L A/B
- roll-targeted shared-Actor retention
- sequential Flight retention repair
- final-shared v1/v2 and corrected Apex follow-ons
- five-stage phase-balanced unified RSI controllers
- universal 4 -> 8 -> 16/32 requirement for training Tubes
