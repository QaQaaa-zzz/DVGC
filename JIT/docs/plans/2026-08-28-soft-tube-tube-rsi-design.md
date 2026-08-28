# Learned Soft Tube and Unified Tube-RSI Design

## Goal

Turn frozen `V_up` and `V_down` scores on real TRAIN snapshots into one
phase-aware learned Soft Tube, then prove that a single unified policy runtime
can reset from both phases without using an expert router.

## Artifact boundary

`T_up` contains only upstream TRAIN snapshots scored by frozen `V_up`.
`T_down` contains only downstream TRAIN snapshots scored by frozen `V_down`.
The learned Tube is their provenance-preserving union. Cross-phase scoring,
hard feasibility admission, synthetic state interpolation, and qpos/qvel
mutation are forbidden.

Every entry binds its real snapshot path, source bank, state hash, parent group,
seed, role, source checkpoint, phase, label provenance, value-model identities,
raw value score, and `0.05 + 0.95 * value_score` sampling weight. Exact duplicate
physical states are stored once; conflicting phase, label, observation, or
provenance for one state hash is an engineering error.

The manifest records `jit_soft_tube_v1`, `certified_safe=false`,
`training_guidance_only=true`, `test_data_used=false`, zero training
transitions, zero environment interactions, all input hashes, fixed weighting,
and fixed 50/50 phase mass.

## Tube-RSI runtime

The sampler first chooses upstream/downstream with equal probability, then
chooses a real snapshot inside that phase proportional to its positive weight.
All random choices are derived from the supplied JAX key, making fixed seeds
reproducible under JIT/vmap.

The unified environment uses one Actor and one action mapping. Snapshot phase
is training/reset provenance and initializes a JAX phase state; it never picks
an expert. Existing observable Apex/descent/contact events drive the transition
between the already implemented Phase U and Phase D semantics. No value model
or phase expert is called during deployment stepping.

## Validation and rollout

Pure tests enforce split isolation, phase-specific scoring, identities,
deduplication, weighting, and deterministic sampling. GPU integration restores
eight fixed TRAIN entries per phase and performs one step each: 16 diagnostic
interactions and zero PPO transitions. Only after both Soft Tube and Tube-RSI
smoke are GO may one existing 25,600-transition PPO block start as an
engineering pilot. That pilot cannot support expert, learnability, safety, or
JCE/JEL claims.
