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
- Resolve the containing commit of this file with `git rev-parse HEAD`; no run
  artifact is keyed to these documentation-only commits.

## Last validated gate

- Latest recorded pre-cleanup runtime gate: PASS at source fingerprint
  `3dcbfff04d02b9b68e00f136e59b46e2a8afe3b1df80b7919b1f1f11756f3a76`.
- Treat that result as reusable only if `cli.runtime_gate --check-only` confirms
  the current XML, config, and source fingerprint.
- The gate's 64+32 timestep PPO is compile/update/resume smoke evidence only.

## Current active run

None authorized. Repository cleanup must not start formal PPO, a learnability
pilot, or the two-phase pipeline.

## Current inputs and hashes

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c`
- Payload: 4.0 kg
- Hip/knee limits: +/-50 N m
- Action order: `[steer, rear-wheel drive, hip, knee]`
- Runtime: `/home/qy/mujoco_playground/.venv/bin/python`

## Latest result

The dependency-closure cleanup design is approved. It identifies 55 retained
paths, 43 provisional deletion candidates, one archive summary, and 472 paths
deferred by default at the 571-path baseline. No source has yet been deleted.

Legacy five-stage experimental outcomes are not evidence for the unimplemented
two-phase method and must not be promoted retrospectively.

## Known blocker

The two expert objectives, Apex transition-band contract, snapshot label schema,
`V_up`/`V_down`, learned soft Tubes, unified two-phase PPO, and final pipeline
entrypoint still require a separate method-implementation task.

## Next permitted action

Complete dependency-safe repository cleanup: switch documentation truth,
perform fresh reference/systemd checks, remove only approved dependency-free
legacy entrypoints, migrate reusable test contracts, validate the retained
repository, and stop. Do not begin two-phase implementation or long training.

## Closed routes

- Landing -> Flight -> Takeoff -> Approach sequential shared-Actor bootstrap
- exhaustive H1/C_L A/B
- roll-targeted shared-Actor retention
- sequential Flight retention repair
- final-shared v1/v2 and corrected Apex follow-ons
- five-stage phase-balanced unified RSI controllers
- universal 4 -> 8 -> 16/32 requirement for training Tubes
