# OrangeBike DVGC

DVGC studies how a single-track robot can learn a complete jump despite the
early-failure bottleneck that makes later flight and recovery states difficult
to reach from a natural start.

## Current research direction

The approved concise RA-L method uses two experts:

1. `Propulsion-Ascent` reaches an Apex transition band.
2. `Descent-Recovery` continues from that band to landing and stable recovery.

Frozen experts provide event-aligned snapshots and continuation labels for two
feasibility models, `V_up` and `V_down`. Their learned soft feasibility Tubes
guide one final Tube-RSI PPO. A soft Tube is not a certified safe set. The final
empirical Jump Capability Envelope is measured only by independent evaluation
of the frozen unified policy.

The method is specified in `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`.

## Implementation status

Implemented and reusable today: the authoritative model loader, action mapping,
MJX environment, observations/history, snapshot round-trip, `SnapshotBank`, PPO
training/resume/inference, policy bundles, rollout, rewards, reference parsing,
provenance, seed tracking, runtime validation, and generic evaluation.

Not yet implemented: final two-phase expert semantics, `V_up`/`V_down`, learned
soft-Tube construction, two-phase unified Tube-RSI PPO, and a new two-phase
pipeline CLI. Existing five-stage code is retained temporarily as a legacy
migration source, not advertised as the active method.

## Environment

Use the existing configured interpreter without reinstalling or upgrading it:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.prepare_project \
  --xml assets/orange_bike_4kg_horizontal.xml \
  --reference data/reference_jump.csv
```

The only authoritative model is
`assets/orange_bike_4kg_horizontal.xml`: 4 kg payload, hip/knee force limits
`+/-50 N m`, and action order `[steer, rear-wheel drive, hip, knee]`. Cleanup
does not modify XML, meshes, collision geometry, obstacles, matcher radii, or
the configured MuJoCo Playground environment.

## Minimum preflight

```bash
bash scripts/local_preflight.sh
```

For the complete runtime smoke gate:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate
```

The gate includes only a 64+32 timestep PPO compile/update/resume smoke test;
it is not formal training or a learnability pilot.

## Existing runnable entrypoints

- `python -m cli.prepare_project`: audit configuration, XML, and reference data.
- `python -m cli.runtime_gate`: validate model load, reset/step, snapshot,
  checkpoint resume, deterministic inference, and bounded PPO plumbing.
- `python -m cli.build_candidates`, `cli.train`, `cli.certify`, `cli.audit`, and
  `cli.evaluate`: reusable generic infrastructure retained for migration.
- `python -m pytest -q`: repository verification.

There is currently no formal two-phase pipeline command. No placeholder command
is provided.

## Repository layout

- `dvgc/`: reusable model, environment, snapshot, PPO, policy, rollout, and
  evaluation infrastructure.
- `cli/`: real runnable entrypoints plus legacy migration utilities under
  dependency review.
- `scripts/`: preflight and temporarily retained launchers under dependency
  review.
- `configs/`: default and legacy experiment configurations.
- `tests/`: stable contracts plus legacy route tests pending safe extraction.
- `docs/EXPERIMENT_STATE.md`: compact recoverable state.
- `docs/REPOSITORY_LAYOUT.md`: cleanup dependency and deletion ledger.

The cleanup branch does not start the future two-phase training pipeline. Its
next permitted action is documented in `docs/EXPERIMENT_STATE.md`.
