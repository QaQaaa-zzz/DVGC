# Two-Phase Gate A Contract Design

## Status and scope

This design covers only Gate A: static two-phase semantics and artifact
contracts. It creates no expert policy, feasibility model, soft-Tube bank,
unified policy, training run, or JCE/JEL result. Passing Gate A proves that the
interfaces are importable, internally consistent, and tested; it is not a
learnability or implementation-completion claim for later method stages.

The accepted approach is contract-first. The alternatives considered were a
Phase-U vertical slice and broad CLI scaffolding. A vertical slice would force
reward and reset decisions before the shared Apex/snapshot contracts exist;
scaffolding every CLI would create a wide but unvalidated surface. Contract-first
keeps the causal order explicit and produces a reviewable base for both experts.

## Architectural boundaries

Gate A adds three focused modules:

- `dvgc/two_phase_semantics.py` owns phase names, internal event names, Apex
  transition-band membership, and phase-local success predicates.
- `dvgc/feasibility.py` owns the two-phase snapshot overlay, continuation-label
  schema, parent-disjoint splitting, scorer-inference validation, bounded
  feasibility shaping, and learned-soft-Tube metadata contracts.
- `dvgc/training_budget.py` owns unambiguous PPO transition accounting and
  report validation while delegating Brax alignment to `dvgc.runtime`.

These modules are additive. Gate A does not rewrite `dvgc/env.py`,
`dvgc/bank.py`, `dvgc/rollout.py`, or the legacy five-stage APIs. Later gates
may call the new pure contracts from the environment and stable CLIs after
their objectives and run protocols are separately reviewed.

No version-suffixed production file is introduced. No production CLI is added
in Gate A because a CLI that cannot yet perform its declared causal stage would
be misleading scaffolding.

## Two-phase semantics

`dvgc.two_phase_semantics` defines exactly these public phase values:

```python
PHASE_UP = "propulsion_ascent"
PHASE_DOWN = "descent_recovery"
PHASES = (PHASE_UP, PHASE_DOWN)
```

The Apex transition band is not a phase value. Internal event constants cover:

```text
jump_window_entered
liftoff_seen
stable_airborne
ascending
apex_band_entered
descending
pre_landing
first_valid_contact
impact_absorbing
stable_recovery
```

The module exposes immutable signal and threshold dataclasses plus pure
functions:

```python
apex_band_components(signals, thresholds) -> dict[str, ArrayLike]
apex_band_membership(signals, thresholds) -> ArrayLike
propulsion_ascent_success(signals, thresholds) -> ArrayLike
descent_recovery_components(signals, thresholds) -> dict[str, ArrayLike]
descent_recovery_success(signals, thresholds) -> ArrayLike
phase_success(phase, signals, thresholds) -> ArrayLike
```

`ApexBandSignals` contains stable-airborne status, CoM vertical velocity,
clearance, roll, pitch, angular-speed norm, forward velocity, illegal-contact
status, and physical-failure status. `ApexBandThresholds` contains the
corresponding positive bounds. Membership is the conjunction required by the
approved method; no exact height, reference index, old matcher, or oracle phase
is accepted as a substitute.

`RecoverySignals` contains valid-contact status, physical-failure status,
roll, pitch, angular-speed norm, forward velocity, and consecutive recovery
hold ticks. `RecoveryThresholds` contains attitude/rate/speed limits and the
required hold duration. Descent-Recovery success requires every component;
entering legacy Landing does not appear in the interface.

The functions operate on Python, NumPy, or JAX scalar-like values without
converting traced JAX values to host booleans. Component functions retain
separate checks for audit reports; membership functions combine them.

Scientific threshold values are deliberately not chosen in Gate A. Tests use
explicit synthetic fixtures. The later guideline-bank task must produce the
initial threshold payload from guideline envelopes and immutable physical
constraints, with provenance, before either expert can train. Success labels
may not be used to tune those initial values.

## Snapshot semantic overlay

The existing `dvgc_physical_policy_state_v4_timing_explicit` record remains the
only authoritative replay format. Gate A does not increment or fork that
schema. Instead, a two-phase snapshot is a valid v4 record with a required
`two_phase_context` mapping containing:

```text
contract_version
source_phase
parent_trajectory_id
trajectory_id
time_index
event_names
event_position
terminated
truncated
termination_reason
source_policy_hash
source_xml_hash
source_config_hash
```

`event_position` is one of `pre`, `nearest`, or `post` for Apex-aligned rows
and `event` for other aligned events. `source_phase` must be one of the two
formal phases. `event_names` may contain only declared internal events.

```python
validate_phase_snapshot(record, **v4_validation_inputs) -> dict[str, Any]
```

first calls `dvgc.snapshot_timing.validate_snapshot_v4` and then checks the
overlay, phase/provenance consistency, terminal exclusivity, and nonempty
lineage. It never calls the deprecated compatibility restore fallback.
Validation returns named checks and failures rather than collapsing evidence
to one boolean.

Gate A validates the static record contract only. Actual same-seed/same-action
round-trip execution belongs to Gate B, where MuJoCo state is available.

## Continuation labels

A continuation label is nested under `continuation_label` and contains:

```text
contract_version
phase
num_rollouts
num_successes
empirical_rate
termination_reason_counts
physical_failure_rate
timeout_rate
label_source_policy_hash
label_protocol_hash
```

Validation requires positive rollout count, successes within the rollout
count, exact empirical-rate consistency within numeric tolerance, nonnegative
terminal counts summing to the rollout count, rates within `[0, 1]`, phase
agreement with the snapshot, and presence of the frozen labeling-policy hash as
an explicit field. Labels are described as empirical continuation evidence
under the frozen protocol, never as physical reachability or true safety
probability.

No fixed branch count is part of the schema.

## Parent-disjoint data splits

```python
split_by_parent(records, *, train_fraction, validation_fraction, seed)
    -> ParentDisjointSplit
```

The function uses only `two_phase_context.parent_trajectory_id` as the grouping
authority. It deterministically shuffles unique parents from the supplied
seed, assigns whole parents to train/validation/test, rejects fewer than three
unique parents, rejects empty partitions, and reports row and parent counts.
Adjacent snapshots from one trajectory can never cross a split.

This function provides the split contract only. Feasibility training and
calibration do not occur in Gate A.

## Feasibility scorer and shaping contracts

Gate A defines a scorer protocol without selecting or training a network:

```python
validate_scorer_inference(scorer, features, *, expected_rows) -> dict[str, Any]
```

The validator requires one finite scalar score per row and rejects NaN,
infinity, shape mismatch, or a scorer that mutates its inputs. The test scorer
is synthetic; passing this test does not claim that `V_up` or `V_down` exists.
Feature manifests must reject oracle phase, reward, success label, teacher ID,
controller ID, and reference time/index fields.

Pure shaping helpers implement only the approved mathematics:

```python
bounded_feasibility_delta(current_score, next_score, *, delta_max)
mixed_feasibility_potential(up_score, down_score, up_weight)
```

The delta is clipped symmetrically, `delta_max` must be positive, and the phase
weight must stay in `[0, 1]` with down-weight equal to `1 - up_weight`.
Observable phase-weight construction is deferred until the unified-policy
design so Gate A does not silently choose scientific transition scales.

## Learned soft-Tube artifact contract

```python
build_soft_tube_metadata(...)
validate_soft_tube_metadata(metadata) -> dict[str, Any]
```

always requires:

```text
artifact_role = learned_soft_feasibility_tube
certified_safe = false
training_guidance_only = true
```

Metadata includes phase, model hash, labeled-dataset hash, parent-split hash,
selection rule, XML hash, config hash, action-mapping version, source-policy
hashes, parent count, and layer counts for `core`, `boundary`, and
`exploration`. The validator rejects formal-certification language, missing
provenance, empty total support, one-parent concentration, inconsistent layer
counts, or a phase outside the two formal phases.

Gate A defines metadata only; it does not build a bank or choose 10/30 percent
thresholds from nonexistent model scores.

## PPO budget accounting

`dvgc.training_budget` defines `PPOBudgetReport` and:

```python
build_ppo_budget_report(
    requested_total_transitions,
    num_parallel_envs,
    episode_horizon,
    unroll_length,
    batch_size,
    num_minibatches,
    num_updates_per_batch,
    num_evals,
    experiment_level,
    wall_clock_seconds=None,
) -> PPOBudgetReport

validate_ppo_budget_report(report, *, completed) -> dict[str, Any]
```

The implementation calls existing `ppo_rollout_block_steps`,
`ppo_effective_timesteps`, and `validate_ppo_batch_layout`. It reports:

```text
requested_total_transitions
effective_total_transitions
requested_timesteps
effective_timesteps
alignment_overhead
num_parallel_envs
mean_steps_per_env
episode_horizon
episode_equivalents
ppo_rollout_block_size
ppo_rollout_blocks
ppo_optimizer_updates
num_minibatches
num_updates_per_batch
wall_clock_seconds
```

`ppo_optimizer_updates` counts minibatch optimizer applications:
`ppo_rollout_blocks * num_minibatches * num_updates_per_batch`. The validator
allows `wall_clock_seconds = None` for a pre-run report and requires a finite
nonnegative value when `completed=True`. The report carries an explicit run
level from
`static`, `smoke`, `learnability_pilot`, `formal_expert`,
`formal_unified`, or `final_evaluation` so short runs cannot be described as
formal training.

## Error handling and prohibited behavior

- Unknown phases and events raise `ValueError` with the offending value.
- Missing schema fields are reported explicitly and never inferred from old
  five-stage fields.
- Legacy reference indices, oracle phase IDs, matchers, Chain labels, and
  certified-Tube metadata cannot satisfy two-phase contracts.
- No validator mutates input records.
- No Gate A code reads or writes `runs/`, starts systemd units, loads policy
  checkpoints, executes MuJoCo, or invokes PPO.
- Existing watchdog and legacy launchers remain retained unchanged.

## Test and acceptance strategy

Tests are additive:

- `tests/test_two_phase_semantics.py` covers exact phase vocabulary, Apex
  component failures, thick-band-compatible samples, both success predicates,
  unknown phases, and JAX-jittable masks.
- `tests/test_feasibility.py` covers v4 composition, overlay failures,
  continuation arithmetic and terminal accounting, parent leakage prevention,
  scorer inference, forbidden inputs, shaping bounds, and soft-Tube metadata.
- `tests/test_training_budget.py` covers block alignment, all required report
  fields, optimizer-update accounting, invalid layouts, run-level vocabulary,
  and completed wall-clock validation.

The implementation follows red-green TDD for each contract. Final Gate A
verification is:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_two_phase_semantics.py tests/test_feasibility.py tests/test_training_budget.py
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
scripts/local_preflight.sh
```

`docs/EXPERIMENT_STATE.md` will then record only that Gate A static contracts
passed, with branch, HEAD, test evidence, no active model/policy, zero training
transitions, the remaining blocker, and Gate B as the only next permitted
action. The phase will be committed with explicit paths and no run artifacts.
