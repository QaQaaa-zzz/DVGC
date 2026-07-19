# DVGC Project State

## Publication Target

The active target is a concise IEEE RA-L paper. The implementation and paper
will make three claims only:

1. Event-aligned next-stage labels expose phase reachability without requiring
   every intermediate controller to finish the whole jump.
2. Phase-conditioned reachability estimates guide proposal coverage but never
   replace frozen-policy branch certification.
3. Phase-wise Tube-RSI supports a final unified policy whose own empirical
   envelope is established by end-to-end Final-Recovery certification.

The current event filter is deployable but is not a trained Streaming GRU. The
active Tube is DVGC-Physical, not the full Physical-Belief method described in
the archived v23 research specification.

## Fixed Contracts

- Model: `assets/orange_bike_4kg_horizontal.xml`
- Model SHA-256: `d7e9f43ff8fb9e4571203f81062ce9c828acfa38692ee8c71a3e5daa15ce794c`
- Payload: 4.0 kg
- Hip/knee actuator force range: +/-50 N m
- Action order: `[steer, rear-wheel drive, hip, knee]`
- Control rate: 50 Hz
- Training order: Landing, Flight, Takeoff, Approach, natural-start evaluation
- Formal Tube label: Final Recovery within a fixed horizon before Failure
- Intermediate label: valid physical entry into the next canonical stage
- Final formal label: stable recovery under the frozen unified policy

## Artifact Semantics

- `certified_tube`: frozen-policy Final-Recovery states admitted by the locked
  certification protocol and validated by an independent audit.  Only this
  role supports safe/precision/coverage claims.
- `proposal_support_bank`: physically legal provisional-safe, boundary and
  explicitly marked active-sampling states.  It is training/search support,
  not a Tube; dead, penetrating and nonfinite states are excluded.
- Intermediate experts and the full-jump teacher generate trajectories and
  candidates.  They are frozen data generators, not the final shared policy.
- `expert_conditioned_provisional_envelope`: Final-Recovery evidence under an
  immutable expert stack.  It may provide distillation and RSI data but cannot
  be called a formal shared-policy Tube or JEL.
- `final_shared_policy_jel`: independently recertified phase-wise evidence
  under one frozen shared Actor.  This is the only artifact role permitted to
  define the paper's final empirical Jump Capability Envelope.

## Active Training Architecture

Sequential shared-Actor backward extension and Flight Landing-retention repair
are diagnostic-only.  The active seed-0 route is independently owned
`pi_L`, `pi_F`, `pi_T`, and `pi_A`, irreversible canonical-entry handoffs,
expert-conditioned provisional envelopes, phase-balanced distillation, joint
RSI PPO, and fresh certification of the final shared Actor.  `C_L` remains
immutable while `pi_F` is trained, and `pi_F` has no Landing-retention gate.

## Existing Runtime

The user's configured runtime is:

```bash
/home/qy/mujoco_playground/.venv/bin/python
```

It exists on the user's Ubuntu workstation and is not mounted in the current
cloud container. The environment must not be reinstalled or upgraded as part
of this project.

## Training Gates

No long PPO run starts until all of the following pass in the configured
runtime:

1. XML and mesh load.
2. Raw reset and 100-step zero/random action rollout.
3. Brax metric dictionary and asymmetric observation contract.
4. Snapshot save/restore round trip under identical action and seed.
5. Policy parameter save/load and deterministic inference.
6. Short PPO compile/run/resume test.
7. Failure, timeout, phase transition, Chain, and Final event audit.

## Current Git Baseline

- `6e41bbf`: imported clean research baseline.
- `5766848`: adopted the authoritative 4 kg model.
