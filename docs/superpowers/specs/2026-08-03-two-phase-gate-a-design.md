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

The authorized implementation scope is exactly:

```text
dvgc/two_phase_semantics.py
dvgc/feasibility.py
dvgc/training_budget.py
tests/test_two_phase_semantics.py
tests/test_feasibility.py
tests/test_training_budget.py
docs/EXPERIMENT_STATE.md
docs/superpowers/specs/2026-08-03-two-phase-gate-a-design.md
```

Gate A does not modify `dvgc/env.py`, `dvgc/rewards.py`, `dvgc/bank.py`, PPO
logic, XML, action mapping, matchers, the configured virtual environment, or
legacy files.

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
advance_recovery_hold_count(signals, thresholds) -> ArrayLike
descent_recovery_success(signals, thresholds) -> ArrayLike
phase_success(phase, signals, thresholds) -> ArrayLike
```

`ApexBandSignals` contains stable-airborne status, CoM vertical velocity,
clearance, roll, pitch, angular-speed norm, forward velocity,
`obstacle_relative_x`, illegal-contact status, and physical-failure status.
`ApexBandThresholds` contains the corresponding positive bounds plus
`relative_x_min` and `relative_x_max`. Membership requires:

```text
relative_x_min <= obstacle_relative_x <= relative_x_max
```

This is a deployable obstacle-relative horizontal geometry window. It is not
global position tracking, reference-point tracking, or an old matcher. A robot
far from the obstacle cannot satisfy Phase-U success from height and low
vertical speed alone.

`RecoverySignals` contains `stable_wheel_support`, `landing_region_valid`,
`no_body_contact`, physical-failure status, roll, pitch, angular-speed norm,
forward velocity, and `previous_recovery_hold_count`. `RecoveryThresholds`
contains attitude/rate/speed limits and the positive integer required hold
duration. `stable_wheel_support` means legal wheel-ground support on the
current tick; it is not a historical "contact seen" latch.
`advance_recovery_hold_count` increments the previous count only when every
current wheel-support, landing-region, no-body-contact, pose/rate/speed, and
no-physical-failure gate holds, and otherwise returns zero. This pure state
transition makes continuous validity auditable and prevents a caller-supplied
historical contact count from satisfying recovery. The public signal contract
does not retain an ambiguous `valid_contact` field.

Descent-Recovery success requires every current component and a hold count at
least equal to the configured duration. One instantaneous valid contact and
entry into legacy Landing are both insufficient.

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
lineage. `terminated` and `truncated` must be actual booleans; an active row
must use termination reason `none`, while a terminal row must use a non-`none`
reason. It never calls the deprecated compatibility restore fallback.
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
outcome_counts:
  success
  physical_failure
  timeout
  other_failure
termination_reason_counts
physical_failure_rate
timeout_rate
label_source_policy_hash
label_protocol_hash
```

Validation requires positive rollout count, nonnegative integer outcome counts,
and this closed accounting:

```text
sum(outcome_counts.values()) == num_rollouts
outcome_counts.success == num_successes
empirical_rate == num_successes / num_rollouts
physical_failure_rate == outcome_counts.physical_failure / num_rollouts
timeout_rate == outcome_counts.timeout / num_rollouts
```

`termination_reason_counts` remains required as a finer diagnostic and cannot
substitute for the standard outcome categories. Any count/rate inconsistency is
rejected, and every diagnostic reason key must be a nonempty string. Validation
also requires phase agreement with the snapshot and the
frozen labeling-policy hash as an explicit field. Labels are empirical
continuation evidence under the frozen protocol, never physical reachability or
true safety probability.

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

## Deployable feature allowlist

Feature selection is allowlist-only. Gate A defines an ordered
`DEPLOYABLE_FEATURE_ALLOWLIST` and:

```python
build_feature_manifest(feature_names) -> DeployableFeatureManifest
extract_deployable_features(record, manifest) -> numpy.ndarray
build_deployable_feature_matrix(records, manifest) -> numpy.ndarray
```

The allowlist contains only deployable physical observations or estimates:

```text
actor_observation
root_linear_velocity
com_linear_velocity
roll
pitch
angular_velocity
hip_position
knee_position
hip_velocity
knee_velocity
obstacle_relative_x
obstacle_relative_height
stable_wheel_support
landing_region_valid
no_body_contact
jump_signal
observation_history_encoding
```

All unregistered names are rejected by default. The manifest orders requested
fields by their canonical allowlist order, rejects duplicates and empty
manifests, and records the resulting field order. The manifest dataclass
self-validates, and every public extraction/matrix boundary revalidates it so a
directly constructed or forged manifest cannot bypass the allowlist. Extraction reads values only
from a record's dedicated `deployable_features` mapping, flattens each field in
manifest order, requires finite numeric values, and requires stable per-field
shapes across rows. It never falls back to the record root,
`two_phase_context`, provenance, or `continuation_label`.

Consequently these fields, and every other unregistered field, are forbidden:

```text
terminated
truncated
termination_reason
event_names
event_position
source_phase
source_policy_hash
source_config_hash
parent_trajectory_id
trajectory_id
time_index
continuation_label
num_successes
empirical_rate
reward
success
teacher identity
controller identity
reference time/index
oracle phase
```

Tests must demonstrate rejection of non-allowlisted names, isolation from
metadata/result fields, stable manifest order, and the impossibility of scorer
label leakage through snapshot or continuation-result fields.

## Feasibility scorer and shaping contracts

Gate A defines a scorer protocol without selecting or training a network:

```python
validate_scorer_inference(scorer, features, *, expected_rows) -> dict[str, Any]
```

The validator receives only the extracted numeric feature matrix. It requires
one finite scalar score per row and rejects NaN, infinity, shape mismatch, or a
scorer that mutates its inputs. The test scorer is synthetic; passing this test
does not claim that `V_up` or `V_down` exists. Because no record or metadata is
passed to the scorer, result fields cannot be consumed as labels in disguise.

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
Certification/JCE/JEL/safe-Tube claim tokens are rejected recursively from
free-form string metadata, including selection rules and additional fields.

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
unroll_length
batch_size
num_minibatches
num_updates_per_batch
num_evals
wall_clock_seconds
```

`requested_timesteps` is a Brax API legacy alias emitted by the report and is
always exactly equal to `requested_total_transitions`. It is not an independent
input. Likewise, `effective_timesteps` is always exactly equal to
`effective_total_transitions`. Validation rejects any report where either
alias diverges. The primary public budget unit is total environment
transitions.

Validation uses the recorded `unroll_length`, `batch_size`, `num_minibatches`,
and `num_evals` to recompute both the Brax rollout block and the exact effective
total. It requires the effective total to be at least the requested total, so
an internally self-consistent but undercounted report is invalid.

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

## Gate B watchdog block

The operational rule is:

```text
Gate B dynamic MuJoCo/PPO work is blocked until the enabled legacy
dvgc-pipeline-watchdog.timer is explicitly stopped and disabled, or migrated
under separate authorization.
```

The timer was recorded as disabled and inactive before Gate A began, but Gate B
still requires a fresh read-only state check and separate user authorization.
Gate A completion never authorizes automatic progression to Gate B and never
authorizes a systemd state change.

## Test and acceptance strategy

Tests are additive:

- `tests/test_two_phase_semantics.py` covers exact phase vocabulary, Apex
  component failures including the relative horizontal window,
  thick-band-compatible samples, sustained recovery support/region/body-contact
  gates, both success predicates, unknown phases, and JAX-jittable masks.
- `tests/test_feasibility.py` covers v4 composition, overlay failures,
  closed continuation outcome arithmetic plus termination diagnostics, parent
  leakage prevention, allowlist-only extraction, metadata/result isolation,
  scorer inference, shaping bounds, and soft-Tube metadata.
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

The runtime PPO gate is not run because Gate A neither changes the existing
runtime-fingerprint management scope nor authorizes PPO execution. The final
report records this reason explicitly.

`docs/EXPERIMENT_STATE.md` will then record only that Gate A static contracts
passed, with branch, HEAD, test evidence, no active model/policy, zero training
transitions, the remaining blocker, and Gate B as the only next permitted
action. The phase will be committed with explicit paths and no run artifacts.
Work stops after that report and commit; Gate B is not started automatically.
