"""Finalize the user-authorized Iteration-1 64x64 C1 engineering selection.

This is intentionally not a rewrite of the original calibration result.  The
selected upstream 64x64 field keeps ``calibration_passed=false`` because its ROC
AUC is 0.6903137789904502 versus the pre-existing 0.70 formal gate.  The user
has explicitly authorized that exact field as the engineering C_up^1 selection
because calibration contains only six negatives while recall, zero-accepted-
negative, and per-parent accepted-positive checks all pass; the larger 128x128
comparison degraded sharply.

The finalizer copies that exact upstream artifact, fits/calibrates C_down^1 with
the same 64x64 architecture on the already-declared TRAIN/CALIBRATION roles,
and writes a self-hashed continuation summary whose Tube2 authorization is an
explicit engineering override rather than a fabricated formal calibration PASS.
TEST/JCE/JEL and ACCEPTANCE remain untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from .config import file_sha256
from .iterative_frontier_protocol import canonical_sha256
from . import iterative_continuation_fields as iterative
from . import shared_continuation_field_refit as shared
from .iterative_weighting_compat import observed_cell_balanced_weights


SCHEMA = "jit_c1_engineering_selection_v1"
SELECTED_PROFILE = "standard_mlp_64x64_tanh"
SELECTED_UPSTREAM_FIELD_SHA256 = "94528aed8bfb4e6db5c01a2bd4231297a0cd3252f198a889c929bca4ee8aac07"
SELECTED_UPSTREAM_MANIFEST_SHA256 = "f010f1cafd17dc7e981f1c0c0d62f55dbeda75ea8d65185568041ac8f199955d"
SELECTED_UPSTREAM_CALIBRATION_SHA256 = "3b1de2557eba250fdaa1df6e4b6f05082e2f808e9979a3c10ce843d53469cdf3"
SELECTED_UPSTREAM_CALIBRATION_ROLE_SHA256 = "d36f0675cb7196a877a3ab7424390a862d5a562dfce8a4587d29b800ff110bdc"
SELECTED_UPSTREAM_AUC = 0.6903137789904502
SELECTED_UPSTREAM_RECALL = 0.5934515688949522
SELECTED_UPSTREAM_THRESHOLD = 0.9835533512239714


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _verify_self_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    if len(declared) != 64 or canonical_sha256({k: v for k, v in payload.items() if k != field}) != declared:
        raise ValueError(f"{field} self-hash drift")


def _verify_selected_upstream(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(root)
    manifest = _read(root / "upstream" / "manifest.json")
    calibration = _read(root / "upstream" / "calibration.json")
    _verify_self_hash(manifest, "manifest_sha256")
    _verify_self_hash(calibration, "calibration_sha256")
    field_path = root / "upstream" / "field.npz"
    if file_sha256(field_path) != SELECTED_UPSTREAM_FIELD_SHA256:
        raise ValueError("selected upstream 64x64 field identity drift")
    if manifest.get("manifest_sha256") != SELECTED_UPSTREAM_MANIFEST_SHA256:
        raise ValueError("selected upstream 64x64 manifest identity drift")
    if calibration.get("calibration_sha256") != SELECTED_UPSTREAM_CALIBRATION_SHA256:
        raise ValueError("selected upstream 64x64 calibration identity drift")
    if calibration.get("calibration_role_manifest_sha256") != SELECTED_UPSTREAM_CALIBRATION_ROLE_SHA256:
        raise ValueError("selected upstream calibration role identity drift")
    if manifest.get("model_profile") != SELECTED_PROFILE or calibration.get("model_profile") != SELECTED_PROFILE:
        raise ValueError("selected upstream model profile drift")
    if manifest.get("architecture") != "76->64_tanh->64_tanh->1" or int(manifest.get("parameter_count", -1)) != 9153:
        raise ValueError("selected upstream architecture drift")
    if calibration.get("calibration_passed") is not False:
        raise ValueError("selected upstream artifact must preserve the original formal FAIL")
    metrics = calibration.get("metrics")
    if not isinstance(metrics, Mapping) or float(metrics.get("roc_auc")) != SELECTED_UPSTREAM_AUC:
        raise ValueError("selected upstream AUC drift")
    if float(calibration.get("positive_recall_at_threshold")) != SELECTED_UPSTREAM_RECALL:
        raise ValueError("selected upstream recall drift")
    if float(calibration.get("acceptance_threshold_exclusive")) != SELECTED_UPSTREAM_THRESHOLD:
        raise ValueError("selected upstream threshold drift")
    if int(calibration.get("accepted_negative_count", -1)) != 0:
        raise ValueError("selected upstream accepted-negative drift")
    counts = calibration.get("calibration_counts")
    if counts != {"candidate_count": 739, "negative_count": 6, "parent_group_count": 3, "positive_count": 733}:
        raise ValueError("selected upstream calibration support drift")
    parent_support = calibration.get("parent_support")
    if not isinstance(parent_support, Mapping) or len(parent_support) != 3:
        raise ValueError("selected upstream parent support drift")
    if any(int(value.get("accepted_positive_count", 0)) <= 0 for value in parent_support.values() if isinstance(value, Mapping)):
        raise ValueError("selected upstream no longer has accepted-positive support in every parent")
    return manifest, calibration


def finalize_c1_engineering_selection(
    *,
    train_root: Path,
    calibration_root: Path,
    selected_upstream_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    train_manifest, train_rows = iterative._load_role(Path(train_root), "train")
    calibration_manifest, calibration_rows = iterative._load_role(Path(calibration_root), "calibration")
    for field in ("iteration", "policy_actor_sha256", "policy_payload_sha256", "source_tube_manifest_sha256", "plan_sha256"):
        if train_manifest.get(field) != calibration_manifest.get(field):
            raise ValueError(f"TRAIN/calibration {field} mismatch")
    iteration = int(train_manifest["iteration"])
    if iteration != 1:
        raise ValueError("this explicit engineering selection is locked to Iteration 1")

    upstream_manifest, upstream_calibration = _verify_selected_upstream(Path(selected_upstream_root))
    if upstream_manifest.get("source_train_role_manifest_sha256") != train_manifest["role_manifest_sha256"]:
        raise ValueError("selected upstream TRAIN role identity drift")
    if upstream_calibration.get("calibration_role_manifest_sha256") != calibration_manifest["role_manifest_sha256"]:
        raise ValueError("selected upstream CALIBRATION role identity drift")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"engineering-selected C1 output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    try:
        shutil.copytree(Path(selected_upstream_root) / "upstream", output / "upstream")

        model = iterative._model_config(SELECTED_PROFILE, iteration)
        phase_train, train_counts = iterative._rows_for_phase(train_rows, "downstream")
        if train_counts["positive_count"] < 20 or train_counts["negative_count"] < 20 or train_counts["parent_group_count"] < 3:
            raise ValueError(f"downstream TRAIN support insufficient: {train_counts}")

        original_weighting = shared._cell_balanced_weights
        shared._cell_balanced_weights = observed_cell_balanced_weights
        try:
            iterative._fit_standard_mlp_phase(phase_train, phase="downstream", model_cfg=model, output=output)
        finally:
            shared._cell_balanced_weights = original_weighting

        downstream_manifest_path = output / "downstream" / "manifest.json"
        downstream_manifest = _read(downstream_manifest_path)
        downstream_manifest.update(
            {
                "field_name": "C_down^1",
                "iteration": 1,
                "policy_name": "pi_1",
                "policy_actor_sha256": train_manifest["policy_actor_sha256"],
                "policy_payload_sha256": train_manifest["policy_payload_sha256"],
                "source_train_role_manifest_sha256": train_manifest["role_manifest_sha256"],
                "train_counts": train_counts,
                "model_profile": SELECTED_PROFILE,
                "post_failure_architecture_revision": True,
            }
        )
        downstream_manifest.pop("manifest_sha256", None)
        downstream_manifest["manifest_sha256"] = canonical_sha256(downstream_manifest)
        _write(downstream_manifest_path, downstream_manifest)

        phase_calibration, calibration_counts = iterative._rows_for_phase(calibration_rows, "downstream")
        scores = iterative._score(output / "downstream" / "field.npz", phase_calibration)
        downstream_calibration = iterative._calibrate("downstream", phase_calibration, scores)
        downstream_calibration.update(
            {
                "iteration": 1,
                "field_name": "C_down^1",
                "field_manifest_sha256": downstream_manifest["manifest_sha256"],
                "field_file_sha256": downstream_manifest["field_file_sha256"],
                "calibration_role_manifest_sha256": calibration_manifest["role_manifest_sha256"],
                "calibration_counts": calibration_counts,
                "model_profile": SELECTED_PROFILE,
                "model_parameters_refit_on_calibration": False,
                "calibration_rows_may_enter_tube": False,
                "calibration_reused_after_architecture_revision": True,
                "independent_architecture_selection_claim": False,
                "test_data_used": False,
                "final_evaluation_data_used": False,
            }
        )
        downstream_calibration["calibration_sha256"] = canonical_sha256(downstream_calibration)
        _write(output / "downstream" / "calibration.json", downstream_calibration)
        if downstream_calibration["calibration_passed"] is not True:
            raise ValueError("downstream 64x64 continuation calibration did not pass; preserve output and inspect before Tube2")

        override = {
            "schema": SCHEMA,
            "scope": "upstream_C_up^1_only",
            "user_authorized": True,
            "reason": "sample_limited_calibration_negative_support_with_all_non_auc_gates_passing",
            "formal_minimum_roc_auc_unchanged": 0.70,
            "selected_observed_roc_auc": SELECTED_UPSTREAM_AUC,
            "calibration_negative_count": 6,
            "positive_recall": SELECTED_UPSTREAM_RECALL,
            "accepted_negative_count": 0,
            "accepted_positive_in_every_parent": True,
            "selected_model_profile": SELECTED_PROFILE,
            "selected_field_file_sha256": SELECTED_UPSTREAM_FIELD_SHA256,
            "selected_field_manifest_sha256": SELECTED_UPSTREAM_MANIFEST_SHA256,
            "selected_calibration_sha256": SELECTED_UPSTREAM_CALIBRATION_SHA256,
            "threshold_relaxed": False,
            "calibration_result_rewritten": False,
            "acceptance_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
        override["override_sha256"] = canonical_sha256(override)
        _write(output / "engineering_override.json", override)

        summary = {
            "schema": iterative.SUMMARY_SCHEMA,
            "status": "completed_engineering_selected",
            "iteration": 1,
            "policy_name": "pi_1",
            "policy_actor_sha256": train_manifest["policy_actor_sha256"],
            "policy_payload_sha256": train_manifest["policy_payload_sha256"],
            "source_tube_manifest_sha256": train_manifest["source_tube_manifest_sha256"],
            "model_profile": SELECTED_PROFILE,
            "architecture": model["architecture"],
            "model_family": model["family"],
            "parameter_count_per_phase": int(model["parameter_count"]),
            "train_role_manifest_sha256": train_manifest["role_manifest_sha256"],
            "calibration_role_manifest_sha256": calibration_manifest["role_manifest_sha256"],
            "formal_calibration_gate_all_phases_passed": False,
            "engineering_selection_authorized": True,
            "engineering_override_sha256": override["override_sha256"],
            "phases": {
                "upstream": {
                    "field_manifest_sha256": upstream_manifest["manifest_sha256"],
                    "field_file_sha256": upstream_manifest["field_file_sha256"],
                    "calibration_sha256": upstream_calibration["calibration_sha256"],
                    "acceptance_threshold_exclusive": upstream_calibration["acceptance_threshold_exclusive"],
                    "calibration_metrics": upstream_calibration["metrics"],
                    "calibration_positive_recall": upstream_calibration["positive_recall_at_threshold"],
                    "formal_calibration_passed": False,
                    "engineering_override_authorized": True,
                },
                "downstream": {
                    "field_manifest_sha256": downstream_manifest["manifest_sha256"],
                    "field_file_sha256": downstream_manifest["field_file_sha256"],
                    "calibration_sha256": downstream_calibration["calibration_sha256"],
                    "acceptance_threshold_exclusive": downstream_calibration["acceptance_threshold_exclusive"],
                    "calibration_metrics": downstream_calibration["metrics"],
                    "calibration_positive_recall": downstream_calibration["positive_recall_at_threshold"],
                    "formal_calibration_passed": True,
                    "engineering_override_authorized": False,
                },
            },
            "training_transitions": 0,
            "environment_interactions": 0,
            "validation_rows_used_for_weight_fit": 0,
            "acceptance_rows_used_for_weight_fit": 0,
            "test_data_used": False,
            "final_evaluation_data_used": False,
            "next_tube_construction_authorized": True,
            "claim_boundary": {
                "policy_conditioned_empirical_continuation_fields": True,
                "engineering_mainline_only": True,
                "formal_upstream_auc_gate_pass_claim": False,
                "fresh_independent_architecture_selection_claim": False,
                "certified_probability_claim": False,
                "certified_safe_set_claim": False,
                "jce_jel_claim": False,
            },
        }
        summary["summary_sha256"] = canonical_sha256(summary)
        _write(output / "summary.json", summary)
        return summary
    except BaseException:
        # Preserve partial nonempty evidence for diagnosis exactly as the rest of
        # the iteration pipeline does; never silently clean failed output.
        raise
