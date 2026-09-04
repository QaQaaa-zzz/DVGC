"""Iteration-only compatibility for sparse parent-label TRAIN cells.

Bootstrap C0 refit evidence was curated so every parent group contained both
continuation labels.  Later automatic frontier iterations only predeclare and
gate class support at the phase level; a valid TRAIN role may therefore contain
an outcome-pure parent group.  Requiring both labels inside every parent at fit
time is an inherited bootstrap assumption, not part of the k>=1 frontier role
contract.

This module preserves every predeclared TRAIN row and extends the existing
cell-balancing rule to the cells that actually exist: every observed
``(parent_group_id, label)`` cell receives equal total loss mass.  No missing
cell is fabricated, no row is resampled, and CALIBRATION/ACCEPTANCE/TEST data are
never used for parameter fitting.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .iterative_frontier_protocol import canonical_sha256
from . import iterative_continuation_fields as iterative
from . import shared_continuation_field_refit as shared


CONTRACT_SCHEMA = "jit_iterative_observed_parent_label_weighting_compat_v1"
ARCHIVE_SCHEMA = "jit_iterative_prefit_failure_archive_v1"


def observed_cell_balanced_weights(
    groups: Sequence[str], targets: np.ndarray
) -> np.ndarray:
    """Equalize total mass across every observed parent/label cell.

    When all parents contain both labels this is exactly the historical
    ``equal_parent_label_cell_mass`` calculation.  If a parent is outcome-pure,
    its one observed cell is retained rather than rejecting the whole TRAIN
    role.  Global two-class support remains enforced by the iterative fitter.
    """
    if len(groups) != len(targets) or len(groups) <= 0:
        raise ValueError("iterative TRAIN weighting requires nonempty aligned rows")
    normalized_targets = np.asarray(targets, dtype=np.float32).reshape(-1)
    if not np.isfinite(normalized_targets).all():
        raise ValueError("iterative TRAIN weighting targets must be finite")
    labels = [int(value) for value in normalized_targets]
    if any(value not in (0, 1) for value in labels):
        raise ValueError("iterative TRAIN weighting labels must be binary")
    normalized_groups = [str(group) for group in groups]
    if any(not group for group in normalized_groups):
        raise ValueError("iterative TRAIN weighting parent group missing")

    cells = Counter(zip(normalized_groups, labels))
    weights = np.asarray(
        [1.0 / float(cells[(group, label)]) for group, label in zip(normalized_groups, labels)],
        dtype=np.float32,
    )
    weights *= np.float32(len(weights) / np.sum(weights))
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("iterative continuation field sample weights are invalid")
    return weights


def _train_cell_support(train_root: Path) -> dict[str, Any]:
    manifest, rows = iterative._load_role(Path(train_root), "train")
    phases: dict[str, Any] = {}
    for phase in ("upstream", "downstream"):
        selected = [dict(row) for row in rows if row.get("phase") == phase]
        cells = Counter(
            (str(row["parent_group_id"]), int(row["label"])) for row in selected
        )
        groups = sorted({str(row["parent_group_id"]) for row in selected})
        parent_support = {}
        for group in groups:
            parent_support[group] = {
                "positive_count": int(cells[(group, 1)]),
                "negative_count": int(cells[(group, 0)]),
                "observed_label_cell_count": int(cells[(group, 0)] > 0)
                + int(cells[(group, 1)] > 0),
            }
        positive = sum(int(row["label"]) for row in selected)
        phases[phase] = {
            "candidate_count": len(selected),
            "positive_count": positive,
            "negative_count": len(selected) - positive,
            "parent_group_count": len(groups),
            "observed_parent_label_cell_count": len(cells),
            "outcome_pure_parent_group_count": sum(
                int(value["observed_label_cell_count"] == 1)
                for value in parent_support.values()
            ),
            "parent_support": parent_support,
        }
    return {
        "train_role_manifest_sha256": str(manifest["role_manifest_sha256"]),
        "policy_actor_sha256": str(manifest["policy_actor_sha256"]),
        "policy_payload_sha256": str(manifest["policy_payload_sha256"]),
        "source_tube_manifest_sha256": str(manifest["source_tube_manifest_sha256"]),
        "phases": phases,
    }


def _write_contract(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        declared = str(existing.get("contract_sha256", ""))
        base = {key: value for key, value in existing.items() if key != "contract_sha256"}
        if len(declared) != 64 or canonical_sha256(base) != declared:
            raise ValueError("iterative weighting compatibility contract hash drift")
        if existing != payload:
            raise ValueError("iterative weighting compatibility contract content drift")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _archive_empty_prefit_failure(output_dir: Path) -> Path | None:
    """Preserve the known empty directory left by a failure before model fit."""
    output = Path(output_dir)
    if not output.exists():
        return None
    if not output.is_dir():
        raise FileExistsError(f"iterative continuation output is not a directory: {output}")
    if any(output.iterdir()):
        raise FileExistsError(
            "nonempty incomplete iterative continuation output must be preserved and "
            f"diagnosed explicitly before retry: {output}"
        )
    archive = output.with_name(output.name + "_failed_prefit_parent_cell_contract")
    if archive.exists():
        raise FileExistsError(f"prefit failure archive already exists: {archive}")
    output.rename(archive)
    marker = {
        "schema": ARCHIVE_SCHEMA,
        "status": "preserved_engineering_failure_before_model_fit",
        "original_output_dir": str(output),
        "archived_dir": str(archive),
        "original_directory_was_empty": True,
        "model_parameters_written": False,
        "training_rows_changed": False,
        "calibration_rows_changed": False,
        "acceptance_rows_changed": False,
        "test_data_used": False,
        "reason": (
            "k>=1 TRAIN passed phase-level support but inherited bootstrap cell weighting "
            "rejected an outcome-pure parent group before model fitting"
        ),
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    (archive / "engineering_failure.json").write_text(
        json.dumps(marker, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return archive


def fit_and_calibrate_observed_cells(
    *, train_root: Path, calibration_root: Path, output_dir: Path
) -> dict[str, Any]:
    """Run the existing k>=1 fitter with the iteration-compatible cell rule."""
    output = Path(output_dir)
    support = _train_cell_support(Path(train_root))
    contract_path = output.with_name(output.name + "_weighting_contract.json")
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": "engineering_contract_repair_after_prefit_structural_failure",
        "scope": "k>=1_iterative_continuation_parameter_fit_only",
        "train_root": str(train_root),
        "calibration_root": str(calibration_root),
        "output_dir": str(output),
        "weighting_rule": "equal_total_mass_for_each_observed_parent_group_label_cell",
        "historical_mixed_cell_behavior_preserved": True,
        "missing_parent_label_cells_fabricated": False,
        "train_rows_resampled": False,
        "train_role_membership_changed": False,
        "calibration_rows_used_for_parameter_fit": False,
        "acceptance_rows_used_for_parameter_fit": False,
        "test_data_used": False,
        "bootstrap_c0_strict_parent_two_class_validator_changed": False,
        "model_architecture_changed": False,
        "optimizer_changed": False,
        "calibration_contract_changed": False,
        "train_support": support,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    _write_contract(contract_path, contract)
    _archive_empty_prefit_failure(output)

    original = shared._cell_balanced_weights
    shared._cell_balanced_weights = observed_cell_balanced_weights
    try:
        return iterative.fit_and_calibrate(
            train_root=Path(train_root),
            calibration_root=Path(calibration_root),
            output_dir=output,
        )
    finally:
        shared._cell_balanced_weights = original
