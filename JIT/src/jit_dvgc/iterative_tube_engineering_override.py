"""Build Tube2 from the explicit Iteration-1 64x64 C1 engineering selection.

The generic iterative Tube builder remains strict and still requires formal
calibration PASS for both phases.  This wrapper is deliberately narrower: it
accepts only the exact self-hashed engineering-selection summary created by
``c1_engineering_selection``.  Upstream keeps its original formal calibration
FAIL while downstream must pass normally.  The resulting Tube manifest records
that an engineering override, not a formal all-phase calibration pass, was used.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import file_sha256
from .iterative_frontier_protocol import canonical_sha256
from . import iterative_tube as base
from .c1_engineering_selection import (
    SCHEMA as OVERRIDE_SCHEMA,
    SELECTED_PROFILE,
    SELECTED_UPSTREAM_AUC,
    SELECTED_UPSTREAM_CALIBRATION_SHA256,
    SELECTED_UPSTREAM_FIELD_SHA256,
    SELECTED_UPSTREAM_MANIFEST_SHA256,
)


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    if len(declared) != 64 or canonical_sha256({k: v for k, v in payload.items() if k != field}) != declared:
        raise ValueError(f"{field} self-hash drift")


def _load_engineering_fields(root: Path):
    root = Path(root)
    summary = base._read(root / "summary.json")
    if not isinstance(summary, dict) or summary.get("schema") != "jit_iterative_continuation_fields_v1":
        raise ValueError("engineering iterative Tube continuation summary schema drift")
    if summary.get("status") != "completed_engineering_selected":
        raise ValueError("engineering iterative Tube requires completed_engineering_selected C1")
    if summary.get("engineering_selection_authorized") is not True or summary.get("next_tube_construction_authorized") is not True:
        raise ValueError("engineering iterative Tube C1 selection is not authorized")
    if summary.get("formal_calibration_gate_all_phases_passed") is not False:
        raise ValueError("engineering iterative Tube must preserve formal all-phase calibration FAIL")
    _verify_self_hash(summary, "summary_sha256")

    override = base._read(root / "engineering_override.json")
    if not isinstance(override, dict) or override.get("schema") != OVERRIDE_SCHEMA:
        raise ValueError("engineering override schema drift")
    _verify_self_hash(override, "override_sha256")
    if summary.get("engineering_override_sha256") != override.get("override_sha256"):
        raise ValueError("engineering override identity drift")
    if override.get("scope") != "upstream_C_up^1_only" or override.get("user_authorized") is not True:
        raise ValueError("engineering override scope/authorization drift")
    if override.get("selected_model_profile") != SELECTED_PROFILE:
        raise ValueError("engineering override selected profile drift")
    if override.get("selected_field_file_sha256") != SELECTED_UPSTREAM_FIELD_SHA256:
        raise ValueError("engineering override upstream field drift")
    if override.get("selected_field_manifest_sha256") != SELECTED_UPSTREAM_MANIFEST_SHA256:
        raise ValueError("engineering override upstream manifest drift")
    if override.get("selected_calibration_sha256") != SELECTED_UPSTREAM_CALIBRATION_SHA256:
        raise ValueError("engineering override upstream calibration drift")
    if float(override.get("selected_observed_roc_auc")) != SELECTED_UPSTREAM_AUC:
        raise ValueError("engineering override upstream AUC drift")
    if override.get("threshold_relaxed") is not False or override.get("calibration_result_rewritten") is not False:
        raise ValueError("engineering override may not relax threshold or rewrite calibration result")
    if override.get("acceptance_data_used") is not False or override.get("test_data_used") is not False or override.get("final_evaluation_data_used") is not False:
        raise ValueError("engineering override touched forbidden evidence")

    phases = {}
    for phase in ("upstream", "downstream"):
        manifest = base._read(root / phase / "manifest.json")
        calibration = base._read(root / phase / "calibration.json")
        _verify_self_hash(manifest, "manifest_sha256")
        _verify_self_hash(calibration, "calibration_sha256")
        if manifest.get("model_profile") != SELECTED_PROFILE or manifest.get("architecture") != "76->64_tanh->64_tanh->1":
            raise ValueError(f"engineering iterative Tube {phase} architecture drift")
        if int(manifest.get("parameter_count", -1)) != 9153:
            raise ValueError(f"engineering iterative Tube {phase} parameter-count drift")
        if calibration.get("field_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"engineering iterative Tube {phase} field/calibration drift")
        field_path = root / phase / "field.npz"
        if file_sha256(field_path) != manifest["field_file_sha256"]:
            raise ValueError(f"engineering iterative Tube {phase} field file drift")
        if phase == "upstream":
            if calibration.get("calibration_passed") is not False:
                raise ValueError("engineering upstream must preserve original formal calibration FAIL")
            if manifest.get("field_file_sha256") != SELECTED_UPSTREAM_FIELD_SHA256:
                raise ValueError("engineering upstream selected field identity drift")
            if manifest.get("manifest_sha256") != SELECTED_UPSTREAM_MANIFEST_SHA256:
                raise ValueError("engineering upstream selected manifest identity drift")
            if calibration.get("calibration_sha256") != SELECTED_UPSTREAM_CALIBRATION_SHA256:
                raise ValueError("engineering upstream selected calibration identity drift")
            if float(calibration.get("metrics", {}).get("roc_auc")) != SELECTED_UPSTREAM_AUC:
                raise ValueError("engineering upstream selected AUC drift")
            if int(calibration.get("accepted_negative_count", -1)) != 0:
                raise ValueError("engineering upstream accepted-negative drift")
            parent_support = calibration.get("parent_support")
            if not isinstance(parent_support, Mapping) or any(
                not isinstance(value, Mapping) or int(value.get("accepted_positive_count", 0)) <= 0
                for value in parent_support.values()
            ):
                raise ValueError("engineering upstream parent coverage drift")
        else:
            if calibration.get("calibration_passed") is not True:
                raise ValueError("downstream still requires formal calibration PASS")
        phases[phase] = {"manifest": manifest, "calibration": calibration, "field_path": field_path}
    return summary, phases


def build_iterative_tube_engineering_override(*, source_tube: Path, train_root: Path, fields_root: Path, output_dir: Path):
    original = base._load_fields
    base._load_fields = _load_engineering_fields
    try:
        result = base.build_iterative_tube(
            source_tube=Path(source_tube),
            train_root=Path(train_root),
            fields_root=Path(fields_root),
            output_dir=Path(output_dir),
        )
    finally:
        base._load_fields = original

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    manifest = base._read(manifest_path)
    fields_summary = base._read(Path(fields_root) / "summary.json")
    override = base._read(Path(fields_root) / "engineering_override.json")
    manifest.update(
        {
            "continuation_selection_mode": "explicit_engineering_override",
            "formal_calibration_gate_all_phases_passed": False,
            "engineering_override_used": True,
            "engineering_override_sha256": override["override_sha256"],
            "upstream_formal_auc_gate_passed": False,
            "downstream_formal_calibration_passed": True,
            "continuation_fields_status": fields_summary["status"],
        }
    )
    manifest.pop("manifest_sha256", None)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    base._write(manifest_path, manifest)
    result["manifest"] = manifest
    result["engineering_override_used"] = True
    result["formal_calibration_gate_all_phases_passed"] = False
    return result
