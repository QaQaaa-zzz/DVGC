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

The method contract was revised on 2026-08-10 to make the phase experts local
continuation controllers and state-distribution generators rather than final
deployment outputs. Expert training and feasibility-data acquisition now
overlap: successful checkpoints may start real online candidate acquisition
and policy-bound continuation diagnostics while expert training continues.
Formal feasibility data later re-labels all accumulated candidates under the
selected frozen phase expert. An expert trajectory is never itself a Tube.

Gate C1's stable Phase U smoke capability is now implemented at
`cli/train_phase_expert.py` and `dvgc/phase_expert_training.py`. The first
single-run smoke authorization was consumed on 2026-08-10 and entered
`gate_pause` during Brax's trace-only initial evaluation because reset and step
published different `state.done` dtypes. No PPO rollout, optimizer update,
checkpoint, fixed evaluation, or failure trajectory was produced. The dtype
contract is corrected at `87c3f4d`, but the failed authorization was not reused
and no automatic retry was performed.

After a real Brax-wrapper regression reproduced and closed that defect, one
new run-bound replacement smoke completed at source HEAD `88d074d`. It executed
exactly one 1,600-transition PPO rollout block, the fixed 1,600-transition Brax
evaluation protocol, and 216 transitions across eight external fixed
evaluations. All fixed evaluations ended as `other_failure` with
`takeoff_missed_liftoff_deadline`; there were no successes, physical failures,
or timeouts. This is engineering smoke evidence only and does not authorize a
learnability pilot or establish `pi_up`.

The current task authorizes a new Phase U run with a maximum of 1,000,000 total
training transitions, requested checkpoints at 0/100k/250k/500k/750k/1M, and
separate evaluation/acquisition/continuation accounting. This is an upper
bound, not permission to ignore gate-pause conditions or a requirement to wait
until 1M before acquiring candidate states. The run has not started yet. First,
the revised reward, physical evaluation, aligned checkpoint, truthful resume,
and acquisition-hook contracts must pass red-green implementation and a fresh
bounded smoke.

## Current branch and commit

- Branch: `agent/two-phase-soft-tube`
- Gate B baseline: `5331896bee08a920321a9b39b496f66c7b9b0879`
- Gate B implementation head: `387ae59`
- Failure-video audit implementation head: `5b8fe73`
- Prelaunch-continuation design: `bac2a93`, `691ad8e`, `ec90d6d`
- Prelaunch-continuation implementation: `23a746e`, `6ed2cdc`, `0b86435`
- Gate A implementation: `5e5da3b`
- Gate C1 run contracts: `07a435c`, `24b6217`, `1221b9a`
- Gate C1 Phase U adapter and smoke runtime: `0e4f718`, `b36cfec`
- Gate C1 failure-video archive support: `74723a5`
- Post-pause dtype correction: `87c3f4d`
- Brax-wrapper dtype regression: `88d074d`
- Completed-interaction accounting closure: `55b47d1`
- Cleanup baseline: `main@b7bb815`
- Dependency design: `27f0aa3`
- Clarified design policy: `3cbd6c1`
- Validated cleanup commits through `3d6a6b5`; resolve the containing final
  validation commit with `git rev-parse HEAD`.

## Last validated gate

- Current reusable runtime gate: PASS at source fingerprint
  `568a5982ab698e42e7c6a976f5bdfe66ba64b5a5ecbfb0ee301674580a3c0e51`.
- It used the fresh ignored work directory
  `runs/runtime_gate_gate_c1_20260810/` and took 97.672 seconds.
- The gate's 64+32 = 96 transitions are compile/update/resume smoke evidence
  only, not expert, pilot, or formal training.

## Current active run

None. The original Gate C1 smoke attempt is retained at
`runs/two_phase/phase_experts/gate_c1_phase_u_smoke_20260810_seed710001/`
with status `gate_pause`. Its actual training, Brax evaluation, fixed
evaluation, and combined environment-transition counts are all 0. The failure
occurred during JAX type tracing, so no dynamic frames existed and a failure
video was not applicable. The run was not retried.

The separately authorized replacement run is complete at:

```text
runs/two_phase/phase_experts/gate_c1_phase_u_replacement_smoke_20260810_seed710001/
```

Its actual interaction accounting is 1,600 training + 1,600 Brax evaluation
+ 216 fixed evaluation = 3,416 environment transitions, within the authorized
4,800 ceiling. The run wrote a full-state Orbax checkpoint at transition 1,600
and a validated recursive-identity sidecar. It did not authorize promotion.

The historical Gate B guideline event probe used 17 environment
transitions; its two outcome-video diagnostics used 25 transitions in total.
Those runs are retained provenance, not active work or a revised Gate B pass
condition. Gate C1 used one adapter integration diagnostic transition, 96
runtime-integrity transitions, and 3,416 replacement-smoke interactions.
Phase U learnability-pilot, formal expert, Phase D, feasibility, Soft Tube, and
unified-policy training transitions remain exactly 0.

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

Gate C1 validation on 2026-08-10:

- `python -m compileall dvgc cli`: passed.
- Gate C1/two-phase/repository focused tests: passed.
- Full pytest after the final runtime implementation: 845 passed, with the
  existing JAXopt deprecation warning only.
- `bash scripts/local_preflight.sh`: passed and repeated all 845 tests.
- Fresh managed runtime gate: PASS, 96 engineering-integrity transitions.
- Static threshold refresh used only kinematic reconstruction and immutable
  geometry, with 0 environment and 0 training transitions; canonical hash
  `603ce888e40dae0d15a9cc6c6bf0704af538a62183d343e639e73c430743a881`.
- Real adapter integration probe: 1 environment transition, nonterminal, no
  success, no physical failure.
- Authorized Phase U PPO smoke attempt: `gate_pause` during trace-only initial
  evaluation; actual environment transitions 0 and no checkpoint.
- Automatic retry: none. Phase U pilot/formal authorization: none.
- A real Brax training-wrapper regression was added after the first pause. It
  reproduced the original `lax.scan` dtype error when the cast was removed and
  passed with the corrected contract.
- Replacement Phase U smoke: completed one 1,600-transition rollout block;
  Brax evaluation used 1,600 transitions; external fixed evaluation used 216
  transitions.
- Fixed evaluation outcomes: success 0, physical failure 0, timeout 0,
  other failure 8. All eight terminal reasons were
  `takeoff_missed_liftoff_deadline` (`end_code=12`).
- Jump-window entry occurred at tick 19 in every fixed rollout; no legal
  liftoff, stable-airborne, ascending, or Apex event was observed.
- Eight MP4 videos and eight timing-aligned state traces were saved under the
  replacement run's `failure_videos/` directory.
- Orbax transition-1600 checkpoint sidecar validation passed; recursive
  checkpoint SHA-256:
  `43adb2f97740c7a6348588df9b15f89eb461fe0c602f484c78a79d35d2a4d6b4`.
- Promotion, pilot, and formal authorization remain false.

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

The stable Gate C1 CLI and Phase U adapter now complete an engineering PPO
smoke, but the smoke checkpoint has zero fixed-evaluation success and is not a
trained/frozen `pi_up`. Phase U must train from audited natural starts and earn
the complete Apex-band success contract. Phase D preliminary candidates require the physical seed
validation protocol and cannot be called reachable or safe. Its formal reset
distribution must be sourced primarily from real frozen-`pi_up` Apex and early
descent rollouts. Phase D execution remains blocked at Gate C1.

## Next permitted action

Implement and verify the approved reward-only Phase U hypothesis plus
per-checkpoint physical evaluation, aligned checkpoint, truthful warm-start
resume, and candidate-harvesting hooks. Then execute one newly authorized
bounded smoke. If smoke integrity passes, Codex may issue a new run-bound
authorization and launch one persistent Phase U process capped at 1,000,000
training transitions without waiting for another manual approval. The run may
automatically begin real candidate acquisition and bounded policy-dependent
continuation diagnostics only after their evidence gates pass.

This marker does not authorize formal `V_up`, a learned Soft Tube declaration,
Phase D expert training, unified PPO, or JCE/JEL. Provisional acquisition aids
must remain clearly labeled and cannot shape Phase U reward or reset sampling.

## Closed routes

- Landing -> Flight -> Takeoff -> Approach sequential shared-Actor bootstrap
- exhaustive H1/C_L A/B
- roll-targeted shared-Actor retention
- sequential Flight retention repair
- final-shared v1/v2 and corrected Apex follow-ons
- five-stage phase-balanced unified RSI controllers
- universal 4 -> 8 -> 16/32 requirement for training Tubes
