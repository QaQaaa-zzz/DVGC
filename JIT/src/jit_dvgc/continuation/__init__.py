"""Stable continuation-field, continuation-label, and audit-bank API."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import file_sha256
from ..policy_conditioned_continuation_field import fit_policy_conditioned_continuation_fields
from ..shared_continuation_field_refit import (
    CONFIG_SCHEMA as SHARED_REFIT_CONFIG_SCHEMA,
    fit_shared_continuation_fields,
)
from ..soft_tube import load_soft_tube
from ..unified_continuation_labels import (
    DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS,
    DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED,
    label_unified_continuations,
    validate_candidate_snapshot,
    validate_unified_boundary_catalog,
)


NEGATIVE_ACCEPTANCE_BANK_SCHEMA = "jit_repair_acceptance_negative_bank_v1"
PHASES = ("upstream", "downstream")


class AcceptanceBankReadinessError(ValueError):
    """Raised when a fresh negative acceptance bank misses phasewise minima."""

    def __init__(self, audit: Mapping[str, Any]) -> None:
        self.audit = dict(audit)
        details = ", ".join(
            f"{phase}:states={row['negative_state_count']},parents={row['negative_parent_group_count']}"
            for phase, row in self.audit["readiness"].items()
        )
        super().__init__(
            "fresh negative acceptance bank is not phasewise ready before repair training: "
            + details
        )


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def select_negative_acceptance_rows(
    labels: Sequence[Mapping[str, Any]],
    *,
    excluded_state_sha256: Sequence[str] = (),
    minimum_negative_states_per_phase: int,
    minimum_negative_parent_groups_per_phase: int,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Select all baseline-negative audit rows under a predeclared support rule.

    This function never picks a favorable subset.  Every unique baseline
    continuation-negative row outside the explicitly excluded target support is
    retained.  Phase-wise state and parent-group minima are checked before a
    replacement policy may be trained.
    """
    min_states = int(minimum_negative_states_per_phase)
    min_groups = int(minimum_negative_parent_groups_per_phase)
    if min_states <= 0 or min_groups <= 0:
        raise ValueError("negative acceptance minima must be positive")

    excluded = {str(value) for value in excluded_state_sha256}
    seen_states: set[str] = set()
    selected: list[dict[str, Any]] = []
    excluded_target_negative_count = 0

    for source in labels:
        row = dict(source)
        if row.get("split") != "train":
            raise ValueError("acceptance-bank labels must be TRAIN-only audit rows")
        phase = str(row.get("phase", ""))
        if phase not in PHASES:
            raise ValueError("acceptance-bank label has unsupported phase")
        label = int(row.get("label", -1))
        if label not in (0, 1):
            raise ValueError("acceptance-bank label must be binary")
        state_sha = str(row.get("state_sha256", ""))
        if len(state_sha) != 64:
            raise ValueError("acceptance-bank label has invalid state SHA-256")
        if state_sha in seen_states:
            raise ValueError("acceptance-bank labels contain a duplicate physical state")
        seen_states.add(state_sha)
        if label == 1:
            continue
        if state_sha in excluded:
            excluded_target_negative_count += 1
            continue
        selected.append(row)

    phase_counts = Counter(str(row["phase"]) for row in selected)
    parent_groups = {
        phase: {str(row["parent_group_id"]) for row in selected if row["phase"] == phase}
        for phase in PHASES
    }
    readiness = {
        phase: {
            "negative_state_count": int(phase_counts.get(phase, 0)),
            "negative_parent_group_count": len(parent_groups[phase]),
            "minimum_negative_states": min_states,
            "minimum_negative_parent_groups": min_groups,
            "ready": (
                int(phase_counts.get(phase, 0)) >= min_states
                and len(parent_groups[phase]) >= min_groups
            ),
        }
        for phase in PHASES
    }
    audit = {
        "selection": "all_baseline_continuation_negative_candidates",
        "input_label_count": len(labels),
        "input_negative_count": sum(int(row.get("label", -1)) == 0 for row in labels),
        "excluded_target_state_count": len(excluded),
        "excluded_target_negative_count": excluded_target_negative_count,
        "locked_negative_count": len(selected),
        "readiness": readiness,
    }
    if not all(row["ready"] for row in readiness.values()):
        raise AcceptanceBankReadinessError(audit)
    return tuple(selected), audit


def lock_negative_acceptance_bank(
    labels_path: Path,
    catalog_path: Path,
    output_path: Path,
    *,
    target_tube_path: Path,
    expected_target_tube_manifest_sha256: str,
    predeclaration_file_sha256: str,
    predeclared_protocol_sha256: str,
    minimum_negative_states_per_phase: int,
    minimum_negative_parent_groups_per_phase: int,
) -> dict[str, Any]:
    """Persist one immutable baseline-negative non-final acceptance bank."""
    labels_path = Path(labels_path)
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"acceptance bank already exists: {output_path}")

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if not isinstance(labels, list) or not labels:
        raise ValueError("acceptance-bank source labels must be a nonempty JSON array")
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict) or catalog.get("status") != "completed":
        raise ValueError("acceptance-bank source catalog must be completed")

    target_tube = load_soft_tube(Path(target_tube_path))
    if target_tube.manifest.get("manifest_sha256") != expected_target_tube_manifest_sha256:
        raise ValueError("acceptance-bank target Tube manifest drift")
    target_states = {str(row["state_sha256"]) for row in target_tube.entries}

    try:
        selected, selection_audit = select_negative_acceptance_rows(
            labels,
            excluded_state_sha256=tuple(sorted(target_states)),
            minimum_negative_states_per_phase=minimum_negative_states_per_phase,
            minimum_negative_parent_groups_per_phase=minimum_negative_parent_groups_per_phase,
        )
    except AcceptanceBankReadinessError as error:
        readiness_path = output_path.with_name("acceptance_readiness.json")
        readiness = {
            "schema": "jit_repair_acceptance_readiness_v1",
            "status": "not_ready_before_repair_training",
            "selection": "all_baseline_continuation_negative_candidates",
            **error.audit,
            "training_transitions": 0,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
        readiness_path.parent.mkdir(parents=True, exist_ok=True)
        readiness_path.write_text(
            json.dumps(readiness, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        raise

    actor_ids = {str(row["policy_actor_sha256"]) for row in selected}
    payload_ids = {str(row["policy_payload_sha256"]) for row in selected}
    label_protocols = {str(row["label_protocol_sha256"]) for row in selected}
    acquisition_protocols = {
        str(row["acquisition_protocol_sha256"]) for row in selected
    }
    if len(actor_ids) != 1 or len(payload_ids) != 1:
        raise ValueError("acceptance-bank baseline policy identity is not unique")
    if len(label_protocols) != 1 or len(acquisition_protocols) != 1:
        raise ValueError("acceptance-bank protocol identity is not unique")

    base = {
        "schema": NEGATIVE_ACCEPTANCE_BANK_SCHEMA,
        "status": "locked_before_repair_training",
        "artifact_role": "fresh_nonfinal_baseline_negative_iteration_acceptance_bank",
        "split": "train_audit_only",
        "selection": "all_baseline_continuation_negative_candidates",
        "source_labels": str(labels_path),
        "source_labels_file_sha256": file_sha256(labels_path),
        "source_catalog": str(catalog_path),
        "source_catalog_file_sha256": file_sha256(catalog_path),
        "source_catalog_root": str(catalog_path.parent),
        "source_catalog_protocol_sha256": str(catalog["protocol_sha256"]),
        "baseline_policy_actor_sha256": next(iter(actor_ids)),
        "baseline_policy_payload_sha256": next(iter(payload_ids)),
        "label_protocol_sha256": next(iter(label_protocols)),
        "acquisition_protocol_sha256": next(iter(acquisition_protocols)),
        "predeclaration_file_sha256": str(predeclaration_file_sha256),
        "predeclared_protocol_sha256": str(predeclared_protocol_sha256),
        "target_tube": str(target_tube_path),
        "target_tube_manifest_sha256": str(expected_target_tube_manifest_sha256),
        "target_tube_state_count": len(target_states),
        "entry_count": len(selected),
        "selection_audit": selection_audit,
        "entries": list(selected),
        "environment_interactions": 0,
        "training_transitions": 0,
        "expert_switching_used": False,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "nonfinal_iteration_acceptance_audit_only": True,
            "candidate_policy_outcomes_inspected": False,
            "repair_training_data": False,
            "tube_construction_data": False,
            "continuation_field_fit_data": False,
            "jce_jel_claim": False,
            "certified_safe_set_claim": False,
        },
    }
    bank_sha = _canonical_sha256(base)
    result = {**base, "bank_sha256": bank_sha}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


__all__ = [
    "fit_policy_conditioned_continuation_fields",
    "SHARED_REFIT_CONFIG_SCHEMA",
    "fit_shared_continuation_fields",
    "DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS",
    "DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED",
    "label_unified_continuations",
    "validate_candidate_snapshot",
    "validate_unified_boundary_catalog",
    "NEGATIVE_ACCEPTANCE_BANK_SCHEMA",
    "AcceptanceBankReadinessError",
    "select_negative_acceptance_rows",
    "lock_negative_acceptance_bank",
]
