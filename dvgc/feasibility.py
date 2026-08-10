"""Static feasibility, continuation, and learned soft-Tube contracts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np
from jax import numpy as jp

from .snapshot_timing import validate_snapshot_v4
from .two_phase_semantics import INTERNAL_EVENTS, PHASES


PHASE_SNAPSHOT_CONTRACT_VERSION = 1
CONTINUATION_LABEL_CONTRACT_VERSION = 1
OUTCOME_CATEGORIES = ("success", "physical_failure", "timeout", "other_failure")
DEPLOYABLE_FEATURE_ALLOWLIST = (
    "actor_observation",
    "root_linear_velocity",
    "com_linear_velocity",
    "roll",
    "pitch",
    "angular_velocity",
    "hip_position",
    "knee_position",
    "hip_velocity",
    "knee_velocity",
    "obstacle_relative_x",
    "obstacle_relative_height",
    "stable_wheel_support",
    "landing_region_valid",
    "no_body_contact",
    "jump_signal",
    "observation_history_encoding",
)
_PHASE_CONTEXT_FIELDS = frozenset(
    {
        "contract_version",
        "source_phase",
        "parent_trajectory_id",
        "trajectory_id",
        "time_index",
        "event_names",
        "event_position",
        "terminated",
        "truncated",
        "termination_reason",
        "source_policy_hash",
        "source_xml_hash",
        "source_config_hash",
    }
)


def validate_phase_snapshot(record: Mapping[str, Any], **v4_validation_inputs: Any) -> dict[str, Any]:
    """Compose two-phase context with the authoritative timing-explicit v4 schema."""
    v4_result = validate_snapshot_v4(record, **v4_validation_inputs)
    context = record.get("two_phase_context")
    is_mapping = isinstance(context, Mapping)
    context = context if is_mapping else {}
    missing = sorted(_PHASE_CONTEXT_FIELDS - set(context))
    events = context.get("event_names")
    valid_events = (
        isinstance(events, (list, tuple))
        and bool(events)
        and all(isinstance(event, str) and event in INTERNAL_EVENTS for event in events)
    )
    has_apex_event = valid_events and "apex_band_entered" in events
    position = context.get("event_position")
    position_valid = (
        position in ("pre", "nearest", "post")
        if has_apex_event
        else position in ("pre", "event", "nearest", "post")
    )
    time_index = context.get("time_index")
    time_index_valid = (
        isinstance(time_index, int) and not isinstance(time_index, bool) and time_index >= 0
    )
    terminated = context.get("terminated")
    truncated = context.get("truncated")
    terminal_flags_valid = type(terminated) is bool and type(truncated) is bool
    is_terminal = terminal_flags_valid and (terminated or truncated)
    termination_reason = context.get("termination_reason")
    termination_reason_valid = (
        isinstance(termination_reason, str)
        and bool(termination_reason)
        and (
            (not is_terminal and termination_reason == "none")
            or (is_terminal and termination_reason != "none")
        )
    )
    provenance = record.get("provenance")
    provenance = provenance if isinstance(provenance, Mapping) else {}
    checks = {
        "v4_snapshot": bool(v4_result["valid"]),
        "two_phase_context": is_mapping,
        "context_required_fields": not missing,
        "contract_version": context.get("contract_version") == PHASE_SNAPSHOT_CONTRACT_VERSION,
        "source_phase": context.get("source_phase") in PHASES,
        "lineage": all(
            isinstance(context.get(name), str) and bool(context.get(name))
            for name in ("parent_trajectory_id", "trajectory_id")
        ),
        "time_index": time_index_valid,
        "event_names": valid_events,
        "event_position": position_valid,
        "terminal_flags": terminal_flags_valid,
        "terminal_exclusive": terminal_flags_valid and not (terminated and truncated),
        "termination_reason": termination_reason_valid,
        "context_provenance": all(
            context.get(context_name) == provenance.get(provenance_name)
            for context_name, provenance_name in (
                ("source_policy_hash", "policy_params_sha256"),
                ("source_xml_hash", "xml_sha256"),
                ("source_config_hash", "config_sha256"),
            )
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {
        "valid": not failed,
        "checks": checks,
        "failed": failed,
        "missing": missing,
        "v4": v4_result,
    }


def _nonnegative_integer_counts(
    value: Any,
    *,
    exact_keys: tuple[str, ...] | None = None,
    require_string_keys: bool = False,
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if exact_keys is not None and set(value) != set(exact_keys):
        return False
    return (not require_string_keys or all(isinstance(key, str) and key for key in value)) and all(
        isinstance(count, int) and not isinstance(count, bool) and count >= 0
        for count in value.values()
    )


def _rate_matches(actual: Any, expected: float | None) -> bool:
    if expected is None:
        return False
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def validate_continuation_label(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate closed empirical continuation outcomes under a frozen policy."""
    label = record.get("continuation_label")
    label = label if isinstance(label, Mapping) else {}
    context = record.get("two_phase_context")
    context = context if isinstance(context, Mapping) else {}
    num_rollouts = label.get("num_rollouts")
    rollouts_valid = (
        isinstance(num_rollouts, int)
        and not isinstance(num_rollouts, bool)
        and num_rollouts > 0
    )
    outcomes = label.get("outcome_counts")
    outcomes_valid = _nonnegative_integer_counts(outcomes, exact_keys=OUTCOME_CATEGORIES)
    outcomes = outcomes if outcomes_valid else {}
    reasons = label.get("termination_reason_counts")
    reasons_valid = (
        _nonnegative_integer_counts(reasons, require_string_keys=True) and bool(reasons)
    )
    reasons = reasons if reasons_valid else {}
    expected_successes = outcomes.get("success") if outcomes_valid else None
    expected_empirical = expected_successes / num_rollouts if rollouts_valid and outcomes_valid else None
    expected_physical = (
        outcomes["physical_failure"] / num_rollouts
        if rollouts_valid and outcomes_valid
        else None
    )
    expected_timeout = (
        outcomes["timeout"] / num_rollouts if rollouts_valid and outcomes_valid else None
    )
    checks = {
        "continuation_label": isinstance(record.get("continuation_label"), Mapping),
        "contract_version": label.get("contract_version")
        == CONTINUATION_LABEL_CONTRACT_VERSION,
        "phase": label.get("phase") in PHASES
        and label.get("phase") == context.get("source_phase"),
        "num_rollouts": rollouts_valid,
        "outcome_counts": outcomes_valid,
        "outcome_total": rollouts_valid
        and outcomes_valid
        and sum(outcomes.values()) == num_rollouts,
        "success_count": outcomes_valid
        and isinstance(label.get("num_successes"), int)
        and not isinstance(label.get("num_successes"), bool)
        and label.get("num_successes") == expected_successes,
        "empirical_rate": _rate_matches(label.get("empirical_rate"), expected_empirical),
        "physical_failure_rate": _rate_matches(
            label.get("physical_failure_rate"), expected_physical
        ),
        "timeout_rate": _rate_matches(label.get("timeout_rate"), expected_timeout),
        "termination_reason_counts": reasons_valid,
        "termination_reason_total": rollouts_valid
        and reasons_valid
        and sum(reasons.values()) == num_rollouts,
        "label_provenance": all(
            isinstance(label.get(name), str) and bool(label.get(name))
            for name in ("label_source_policy_hash", "label_protocol_hash")
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"valid": not failed, "checks": checks, "failed": failed}


@dataclass(frozen=True)
class ParentDisjointSplit:
    train: tuple[Mapping[str, Any], ...]
    validation: tuple[Mapping[str, Any], ...]
    test: tuple[Mapping[str, Any], ...]

    @property
    def row_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def parent_counts(self) -> dict[str, int]:
        def count(rows: tuple[Mapping[str, Any], ...]) -> int:
            return len(
                {
                    row["two_phase_context"]["parent_trajectory_id"]
                    for row in rows
                }
            )

        return {
            "train": count(self.train),
            "validation": count(self.validation),
            "test": count(self.test),
        }


def split_by_parent(
    records: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    *,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> ParentDisjointSplit:
    """Deterministically assign complete parent trajectories to three splits."""
    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    if not (
        math.isfinite(train_fraction)
        and math.isfinite(validation_fraction)
        and train_fraction > 0.0
        and validation_fraction > 0.0
        and train_fraction + validation_fraction < 1.0
    ):
        raise ValueError("train and validation fractions must be positive and sum below one")
    parent_by_row = []
    for row in records:
        context = row.get("two_phase_context")
        parent = context.get("parent_trajectory_id") if isinstance(context, Mapping) else None
        if not isinstance(parent, str) or not parent:
            raise ValueError("Every record requires a nonempty parent_trajectory_id")
        parent_by_row.append(parent)
    parents = sorted(set(parent_by_row))
    if len(parents) < 3:
        raise ValueError("Parent-disjoint splitting requires at least three unique parents")
    shuffled = list(np.random.default_rng(int(seed)).permutation(parents))
    train_count = int(math.floor(len(parents) * train_fraction))
    validation_count = int(math.floor(len(parents) * validation_fraction))
    test_count = len(parents) - train_count - validation_count
    if min(train_count, validation_count, test_count) <= 0:
        raise ValueError("Requested parent fractions produce an empty partition")
    train_parents = set(shuffled[:train_count])
    validation_parents = set(shuffled[train_count : train_count + validation_count])
    test_parents = set(shuffled[train_count + validation_count :])

    def select(selected: set[str]) -> tuple[Mapping[str, Any], ...]:
        return tuple(row for row, parent in zip(records, parent_by_row, strict=True) if parent in selected)

    return ParentDisjointSplit(
        train=select(train_parents),
        validation=select(validation_parents),
        test=select(test_parents),
    )


@dataclass(frozen=True)
class DeployableFeatureManifest:
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_feature_fields(self.fields)


def _validate_feature_fields(fields: Any) -> tuple[str, ...]:
    if not isinstance(fields, tuple) or not fields:
        raise ValueError("A feature manifest requires at least one field tuple")
    if not all(isinstance(name, str) for name in fields):
        raise ValueError("Feature manifest fields must be strings")
    if len(set(fields)) != len(fields):
        raise ValueError("A feature manifest cannot contain duplicate fields")
    unknown = [name for name in fields if name not in DEPLOYABLE_FEATURE_ALLOWLIST]
    if unknown:
        raise ValueError(f"Feature is not deployable allowlisted: {unknown}")
    return fields


def _validate_feature_manifest(
    manifest: DeployableFeatureManifest,
) -> tuple[str, ...]:
    if not isinstance(manifest, DeployableFeatureManifest):
        raise ValueError("A DeployableFeatureManifest is required")
    return _validate_feature_fields(manifest.fields)


def build_feature_manifest(feature_names: list[str] | tuple[str, ...]) -> DeployableFeatureManifest:
    """Build a stable manifest from an explicit deployable-feature allowlist."""
    requested = tuple(feature_names)
    _validate_feature_fields(requested)
    requested_set = set(requested)
    ordered = tuple(name for name in DEPLOYABLE_FEATURE_ALLOWLIST if name in requested_set)
    return DeployableFeatureManifest(fields=ordered)


def _deployable_feature_arrays(
    record: Mapping[str, Any], manifest: DeployableFeatureManifest
) -> list[np.ndarray]:
    fields = _validate_feature_manifest(manifest)
    values = record.get("deployable_features")
    if not isinstance(values, Mapping):
        raise ValueError("Record is missing deployable feature mapping")
    arrays = []
    for name in fields:
        if name not in values:
            raise ValueError(f"Record is missing deployable feature: {name}")
        try:
            array = np.asarray(values[name], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Deployable feature {name} must be numeric") from exc
        if not np.isfinite(array).all():
            raise ValueError(f"Deployable feature {name} must be finite")
        arrays.append(array)
    return arrays


def extract_deployable_features(
    record: Mapping[str, Any], manifest: DeployableFeatureManifest
) -> np.ndarray:
    """Flatten only the dedicated deployable feature mapping in manifest order."""
    arrays = _deployable_feature_arrays(record, manifest)
    return np.concatenate([array.reshape(-1) for array in arrays])


def build_deployable_feature_matrix(
    records: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    manifest: DeployableFeatureManifest,
) -> np.ndarray:
    """Build a finite feature matrix while requiring stable field shapes."""
    if not records:
        raise ValueError("Feature matrix requires at least one record")
    rows = []
    expected_shapes: tuple[tuple[int, ...], ...] | None = None
    for record in records:
        arrays = _deployable_feature_arrays(record, manifest)
        shapes = tuple(array.shape for array in arrays)
        if expected_shapes is None:
            expected_shapes = shapes
        elif shapes != expected_shapes:
            raise ValueError("Deployable features must have stable shape across records")
        rows.append(np.concatenate([array.reshape(-1) for array in arrays]))
    return np.stack(rows, axis=0)


def validate_scorer_inference(
    scorer: Any, features: Any, *, expected_rows: int
) -> dict[str, Any]:
    """Validate artifact-neutral scorer inference on numeric features only."""
    matrix = np.asarray(features, dtype=np.float64)
    before = matrix.copy()
    scorer_input = matrix.copy()
    scores = np.asarray(scorer(scorer_input), dtype=np.float64)
    checks = {
        "feature_matrix": matrix.ndim == 2 and matrix.shape[0] == int(expected_rows),
        "input_immutable": np.array_equal(scorer_input, before),
        "score_shape": scores.shape == (int(expected_rows),),
        "finite_scores": bool(np.isfinite(scores).all()),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"valid": not failed, "checks": checks, "failed": failed, "scores": scores}


def bounded_feasibility_delta(
    current_score: Any, next_score: Any, *, delta_max: float
) -> Any:
    """Return a symmetrically clipped feasibility-score improvement."""
    if not math.isfinite(float(delta_max)) or float(delta_max) <= 0.0:
        raise ValueError("delta_max must be finite and positive")
    return jp.clip(next_score - current_score, -float(delta_max), float(delta_max))


def mixed_feasibility_potential(
    up_score: Any, down_score: Any, *, up_weight: float
) -> Any:
    """Blend phase scores with complementary weights."""
    weight = float(up_weight)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise ValueError("up_weight must be finite and within [0, 1]")
    return weight * up_score + (1.0 - weight) * down_score


SOFT_TUBE_ARTIFACT_ROLE = "learned_soft_feasibility_tube"
SOFT_TUBE_LAYERS = ("core", "boundary", "exploration")
_CERTIFICATION_CLAIM_PATTERN = re.compile(
    r"\b(?:certified|certification|jce|jel)\b|\bsafe[\s_-]*tube\b",
    flags=re.IGNORECASE,
)


def _contains_certification_claim(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_CERTIFICATION_CLAIM_PATTERN.search(value))
    if isinstance(value, Mapping):
        return any(_contains_certification_claim(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_certification_claim(item) for item in value)
    return False


def build_soft_tube_metadata(
    *,
    phase: str,
    model_hash: str,
    labeled_dataset_hash: str,
    parent_split_hash: str,
    selection_rule: str,
    xml_hash: str,
    config_hash: str,
    action_mapping_version: str,
    source_policy_hashes: list[str] | tuple[str, ...],
    parent_count: int,
    layer_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build explicit noncertified learned-soft-Tube training metadata."""
    layers = {name: layer_counts.get(name) for name in SOFT_TUBE_LAYERS}
    total = sum(value for value in layers.values() if isinstance(value, int))
    return {
        "artifact_role": SOFT_TUBE_ARTIFACT_ROLE,
        "certified_safe": False,
        "training_guidance_only": True,
        "phase": phase,
        "model_hash": model_hash,
        "labeled_dataset_hash": labeled_dataset_hash,
        "parent_split_hash": parent_split_hash,
        "selection_rule": selection_rule,
        "xml_hash": xml_hash,
        "config_hash": config_hash,
        "action_mapping_version": action_mapping_version,
        "source_policy_hashes": sorted(set(source_policy_hashes)),
        "parent_count": parent_count,
        "layer_counts": layers,
        "total_records": total,
    }


def validate_soft_tube_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate learned-soft-Tube provenance, diversity, and claim boundaries."""
    layers = metadata.get("layer_counts")
    layers_valid = (
        isinstance(layers, Mapping)
        and set(layers) == set(SOFT_TUBE_LAYERS)
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
            for value in layers.values()
        )
    )
    expected_total = sum(layers.values()) if layers_valid else None
    source_policies = metadata.get("source_policy_hashes")
    checks = {
        "artifact_role": metadata.get("artifact_role") == SOFT_TUBE_ARTIFACT_ROLE,
        "claim_boundary": metadata.get("certified_safe") is False
        and metadata.get("training_guidance_only") is True,
        "claim_language": not _contains_certification_claim(metadata),
        "phase": metadata.get("phase") in PHASES,
        "provenance": all(
            isinstance(metadata.get(name), str) and bool(metadata.get(name))
            for name in (
                "model_hash",
                "labeled_dataset_hash",
                "parent_split_hash",
                "selection_rule",
                "xml_hash",
                "config_hash",
                "action_mapping_version",
            )
        ),
        "source_policies": isinstance(source_policies, (list, tuple))
        and bool(source_policies)
        and all(isinstance(value, str) and bool(value) for value in source_policies),
        "parent_diversity": isinstance(metadata.get("parent_count"), int)
        and not isinstance(metadata.get("parent_count"), bool)
        and metadata.get("parent_count") >= 2,
        "layer_counts": layers_valid,
        "layer_total": layers_valid and metadata.get("total_records") == expected_total,
        "nonempty_support": layers_valid and expected_total is not None and expected_total > 0,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    return {"valid": not failed, "checks": checks, "failed": failed}
