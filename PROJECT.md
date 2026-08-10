# DVGC Project

## Publication target

DVGC targets a concise IEEE RA-L method for learning a complete jump with two
phase experts and learned soft feasibility guidance:

```text
kinematic guideline / natural physical states
  -> local phase experts
  -> expert rollout states and local perturbations
  -> frozen-policy continuation rollouts
  -> continuation feasibility labels
  -> V_up and V_down feasibility fields
  -> learned soft feasibility tubes
  -> unified Tube-RSI PPO
  -> frozen-policy empirical Jump Capability Envelope
```

The Apex transition band connects the two phases. It is a band of admissible
handoff states, not a separately owned expert and not a certified safe set.
The input reference is a kinematic guideline and weak prior, not an expert or
authoritative dynamic controller, and complete open-loop replay is not a
prerequisite for training the phase experts.

The local experts are not deployment outputs. They are continuation
controllers and state-distribution generators used to identify where phase
completion remains possible. Expert training and feasibility-data acquisition
overlap in time; candidate states collected from early checkpoints are later
re-labeled under the selected frozen phase expert before formal feasibility
training. The only final deployment controller is `pi_unified`.

## Method contract

- `Propulsion-Ascent` learns the launch and rising-flight behavior needed to
  reach the Apex transition band.
- `Descent-Recovery` learns from the Apex transition band through landing and
  stable recovery.
- Phase-expert checkpoints generate real online candidate states throughout
  training. Physically valid local perturbations give the candidate set
  thickness; an expert trajectory itself is not a Tube.
- Frozen checkpoint policies generate policy-dependent continuation labels.
  Formal datasets re-label accumulated candidates under the selected
  `pi_up_star` or `pi_down_star` to avoid mixing authorities.
- Two learned feasibility models, `V_up` and `V_down`, estimate phase-specific
  continuation feasibility from those labels.
- Thresholded/calibrated feasibility fields define soft training tubes. A soft
  Tube is a learned training guide, not a formal safety certificate.
- One final unified PPO uses Tube-RSI, jump signals, the final task reward, and
  soft feasibility guidance.
- Only an independent evaluation of the frozen unified policy establishes the
  empirical Jump Capability Envelope (JCE/JEL).

The five-stage expert stack, sequential shared-Actor backward extension, and
universal 4 -> 8 -> 16/32 branch funnel are not the current research method.
Neither is a sequential "train both experts to convergence, then start Tube
work" route.

## Implemented today

The repository currently provides reusable infrastructure for:

- configuration loading and hashes;
- authoritative XML loading and audit;
- action mapping;
- MuJoCo/MJX environment, observations, and history;
- snapshot capture, restore, provenance, timing, and `SnapshotBank`;
- PPO training, truthful normalizer/policy/value checkpoint warm starts, and
  deterministic inference;
- policy bundles, rollout, rewards, and reference-trajectory parsing;
- seed/provenance manifests, certification utilities, runtime validation, and
  generic full-jump evaluation;
- Gate A/B two-phase semantics/runtime contracts and the Gate C1 stable
  phase-expert smoke entrypoint, including authorization, checkpoint, fixed
  evaluation, and interaction accounting;
- legacy stage/expert code that may supply utilities during migration.

The existing five-stage implementation is a legacy migration source. Its
controllers, policies, banks, and results do not demonstrate the new two-phase
method.

## Not implemented

The following work remains for separately gated implementation and experiments:

- trained/frozen `pi_up` and `pi_down` checkpoints beyond engineering smoke;
- `V_up` and `V_down` dataset/training/calibration contracts;
- learned soft feasibility Tube construction;
- two-phase Tube-RSI unified PPO;
- the new final two-phase pipeline CLI and independent JCE/JEL protocol.

No current file may claim these capabilities exist until code, tests, and
experiments validate them.

## Immutable physical contracts

- Model: `assets/orange_bike_4kg_horizontal.xml`
- Model SHA-256: `d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c`
- Payload: 4.0 kg
- Hip/knee actuator force limits: +/-50 N m
- Action order: `[steer, rear-wheel drive, hip, knee]`
- Control rate: 50 Hz
- No replacement XML, mesh/collision edits, obstacle changes, or matcher-radius
  changes are permitted by repository cleanup.

## Runtime and validation

The configured runtime is:

```text
/home/qy/mujoco_playground/.venv/bin/python
```

Repository validation uses `scripts/local_preflight.sh` and
`python -m cli.runtime_gate`. The gate's 64+32 timestep PPO sequence is only a
compile/update/checkpoint-resume smoke test. It is not formal training or a
learnability pilot.

See `docs/METHOD_TWO_PHASE_SOFT_TUBE.md` for the approved method definition,
`docs/REPOSITORY_LAYOUT.md` for cleanup decisions, and
`docs/EXPERIMENT_STATE.md` for the recoverable current state.
