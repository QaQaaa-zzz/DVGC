# DVGC Experiment State

## Current method

The approved research direction is the two-phase learned soft-feasibility-Tube
method defined in `PROJECT.md` and `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`:
`Propulsion-Ascent` -> Apex transition band -> `Descent-Recovery`, followed by
`V_up`/`V_down`, soft-Tube guidance, one unified Tube-RSI PPO, and independent
frozen-policy JCE/JEL evaluation.

This method is not yet implemented. Existing five-stage code and results are
legacy migration sources only.

## Current branch and commit

- Branch: `agent/repo-cleanup-two-phase`
- Cleanup baseline: `main@b7bb815`
- Dependency design: `27f0aa3`
- Clarified design policy: `3cbd6c1`
- Validated cleanup commits through `3d6a6b5`; resolve the containing final
  validation commit with `git rev-parse HEAD`.

## Last validated gate

- Current reusable runtime gate: PASS at source fingerprint
  `3dcbfff04d02b9b68e00f136e59b46e2a8afe3b1df80b7919b1f1f11756f3a76`.
- `cli.runtime_gate --check-only` confirmed the current XML, config, and source
  fingerprint on 2026-08-03.
- The gate's 64+32 timestep PPO is compile/update/resume smoke evidence only.

## Current active run

None authorized. Repository cleanup must not start formal PPO, a learnability
pilot, or the two-phase pipeline.

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

The dependency-closure cleanup is implemented. From the 571-path baseline, 28
closed-route files were deleted and the root legacy summary was replaced by one
concise archive note, for 29 net baseline-path removals. The final ledger keeps
56 paths, deletes 28, archives one summary, and defers 486 paths. The retained
tree has 547 tracked files after adding stable tests/contracts and current
documentation.

Final validation: `compileall dvgc cli` passed; full pytest reported 552 passed;
`scripts/local_preflight.sh` passed on the configured GPU runtime; and the
matching runtime-gate report was verified current without running a new PPO
block.

Legacy five-stage experimental outcomes are not evidence for the unimplemented
two-phase method and must not be promoted retrospectively.

## Known blocker

The two expert objectives, Apex transition-band contract, snapshot label schema,
`V_up`/`V_down`, learned soft Tubes, unified two-phase PPO, and final pipeline
entrypoint still require a separate method-implementation task.

## Next permitted action

Stop after this cleanup. The only next permitted action is a separately
authorized two-phase method-implementation task. Do not begin that work or any
formal training from this marker.

## Closed routes

- Landing -> Flight -> Takeoff -> Approach sequential shared-Actor bootstrap
- exhaustive H1/C_L A/B
- roll-targeted shared-Actor retention
- sequential Flight retention repair
- final-shared v1/v2 and corrected Apex follow-ons
- five-stage phase-balanced unified RSI controllers
- universal 4 -> 8 -> 16/32 requirement for training Tubes
