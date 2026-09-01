"""Reconstruct a matched upstream TRAIN panel without spending interactions.

The three-checkpoint CV mixed a heterogeneous legacy transition_4988928
acquisition family with two newer checkpoint domains collected under one
matched panel.  Before collecting anything else, this diagnostic reconstructs
which legacy transition_4988928 labels already came from the exact locked
{durations 4/8/16} x {strengths .025/.1} x four-axis x +/- panel.

It reads TRAIN artifacts and candidate catalogs only.  It never reads consumed
validation rows/predictions, steps an environment, fits a model, constructs a
Tube, or trains a policy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .iteration_train_evidence import canonical_sha256
from .upstream_checkpoint_train_evidence import (
    checkpoint_domain,
    load_frozen_upstream_checkpoint_train_evidence,
)

CONFIG_SCHEMA = "jit_upstream_matched_panel_audit_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_matched_panel_audit_protocol_v1"
SUMMARY_SCHEMA = "jit_upstream_matched_panel_audit_summary_v1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(value: Any, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256")
    return text


def load_upstream_matched_panel_audit_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported upstream matched-panel audit config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("upstream matched-panel audit protocol drift")
    if protocol.get("status") != "predeclared_after_checkpoint_domain_cv_fail_before_more_interactions":
        raise ValueError("upstream matched-panel audit status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("upstream matched-panel audit policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_checkpoint_train_manifest_sha256",
        "frozen_checkpoint_train_labels_sha256",
        "prior_checkpoint_cv_summary_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("target_checkpoint_domain") != "transition_4988928":
        raise ValueError("upstream matched-panel audit target domain drift")
    if protocol.get("expected_target_domain") != {
        "candidate_count": 571,
        "positive_count": 545,
        "negative_count": 26,
        "parent_group_count": 5,
    }:
        raise ValueError("upstream matched-panel audit target-domain counts drift")
    if protocol.get("matched_panel") != {
        "action_names": ["steer", "rear_wheel_drive", "hip", "knee"],
        "signs": [-1, 1],
        "strengths": [0.025, 0.1],
        "durations": [4, 8, 16],
        "expected_cells_per_parent": 48,
        "expected_total_cells": 240,
    }:
        raise ValueError("upstream matched-panel definition drift")
    roots = protocol.get("catalog_search_roots")
    names = protocol.get("catalog_filenames")
    if not isinstance(roots, list) or not roots or not isinstance(names, list) or not names:
        raise ValueError("upstream matched-panel catalog search contract missing")
    if protocol.get("data_policy") != {
        "train_rows_only": True,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "environment_interactions": 0,
        "training_transitions": 0,
    }:
        raise ValueError("upstream matched-panel data policy drift")
    if protocol.get("claim_boundary") != {
        "diagnostic_only": True,
        "matched_panel_reconstruction_only": True,
        "continuation_field_reselected": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("upstream matched-panel claim boundary drift")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("upstream matched-panel protocol SHA drift")
    return config


def _validate_prior_checkpoint_cv(path: Path, expected_sha: str) -> None:
    summary = _read_object(path)
    declared = str(summary.get("summary_sha256", ""))
    payload = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if canonical_sha256(payload) != declared or declared != expected_sha:
        raise ValueError("prior checkpoint-domain CV summary identity drift")
    if summary.get("status") != "completed":
        raise ValueError("prior checkpoint-domain CV is not completed")
    if summary.get("checkpoint_domain_generalization_supported") is not False:
        raise ValueError("matched-panel audit requires the prior checkpoint CV to have failed")
    if summary.get("fresh_validation_predeclaration_authorized") is not False:
        raise ValueError("prior checkpoint CV unexpectedly authorized fresh validation")
    if summary.get("tube_1_authorized") is not False:
        raise ValueError("prior checkpoint CV unexpectedly authorized Tube_1")
    if summary.get("consumed_validation_rows_reused") is not False:
        raise ValueError("prior checkpoint CV reused consumed validation rows")
    if summary.get("consumed_validation_predictions_reused") is not False:
        raise ValueError("prior checkpoint CV reused consumed validation predictions")


def _catalog_paths(roots: Sequence[str], filenames: Sequence[str]) -> list[Path]:
    allowed = set(str(name) for name in filenames)
    paths: set[Path] = set()
    for raw_root in roots:
        root = Path(raw_root)
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            if path.name in allowed:
                paths.add(path)
    return sorted(paths)


def _candidate_metadata_by_state(
    target_rows: Sequence[Mapping[str, Any]],
    catalog_paths: Sequence[Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    target = {str(row["state_sha256"]): dict(row) for row in target_rows}
    matches: dict[str, dict[str, Any]] = {}
    scanned = 0
    relevant_catalogs: set[str] = set()
    protocol_counts: Counter[str] = Counter()
    for path in catalog_paths:
        try:
            catalog = _read_object(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        entries = catalog.get("entries")
        if not isinstance(entries, list):
            continue
        protocol_sha = str(catalog.get("protocol_sha256", ""))
        if len(protocol_sha) != 64:
            continue
        scanned += 1
        catalog_relevant = False
        for source in entries:
            if not isinstance(source, Mapping):
                continue
            state = str(source.get("state_sha256", ""))
            base = target.get(state)
            if base is None or source.get("phase") != "upstream":
                continue
            if str(base.get("acquisition_protocol_sha256", "")) != protocol_sha:
                continue
            if str(source.get("parent_group_id", "")) != str(base.get("parent_group_id", "")):
                raise ValueError("matched-panel catalog/base parent identity drift")
            perturbation = source.get("perturbation")
            if not isinstance(perturbation, Mapping):
                continue
            record = {
                "catalog_path": str(path),
                "acquisition_protocol_sha256": protocol_sha,
                "candidate_id": str(source.get("candidate_id", "")),
                "parent_group_id": str(base["parent_group_id"]),
                "state_sha256": state,
                "label": int(base["label"]),
                "action_name": str(perturbation.get("action_name", "")),
                "sign": int(perturbation.get("sign", 0)),
                "strength": float(perturbation.get("strength", np.nan)),
                "duration": int(perturbation.get("duration", -1)),
            }
            if not np.isfinite(record["strength"]):
                continue
            previous = matches.get(state)
            if previous is not None:
                comparable = {key: value for key, value in record.items() if key != "catalog_path"}
                prior = {key: value for key, value in previous.items() if key != "catalog_path"}
                if comparable != prior:
                    raise ValueError("one TRAIN state maps to conflicting acquisition metadata")
                continue
            matches[state] = record
            protocol_counts[protocol_sha] += 1
            catalog_relevant = True
        if catalog_relevant:
            relevant_catalogs.add(str(path))
    return matches, {
        "catalog_json_files_scanned": scanned,
        "relevant_catalog_paths": sorted(relevant_catalogs),
        "reconstructed_by_acquisition_protocol_sha256": dict(sorted(protocol_counts.items())),
    }


def _is_target_panel(row: Mapping[str, Any], panel: Mapping[str, Any]) -> bool:
    return (
        str(row["action_name"]) in set(panel["action_names"])
        and int(row["sign"]) in set(int(x) for x in panel["signs"])
        and any(abs(float(row["strength"]) - float(value)) <= 1.0e-9 for value in panel["strengths"])
        and int(row["duration"]) in set(int(x) for x in panel["durations"])
    )


def audit_upstream_matched_panel(config_path: Path) -> dict[str, Any]:
    config = load_upstream_matched_panel_audit_config(Path(config_path))
    protocol = config["protocol"]
    root = Path(str(protocol["frozen_checkpoint_train_evidence"]))
    manifest, rows = load_frozen_upstream_checkpoint_train_evidence(root)
    if manifest["manifest_sha256"] != protocol["frozen_checkpoint_train_manifest_sha256"]:
        raise ValueError("matched-panel frozen TRAIN manifest drift")
    if manifest["labels_sha256"] != protocol["frozen_checkpoint_train_labels_sha256"]:
        raise ValueError("matched-panel frozen TRAIN labels drift")
    if manifest["policy_actor_sha256"] != protocol["policy_actor_sha256"]:
        raise ValueError("matched-panel actor identity drift")
    if manifest["policy_payload_sha256"] != protocol["policy_payload_sha256"]:
        raise ValueError("matched-panel payload identity drift")
    _validate_prior_checkpoint_cv(
        Path(str(protocol["prior_checkpoint_cv_summary"])),
        str(protocol["prior_checkpoint_cv_summary_sha256"]),
    )

    target_domain = str(protocol["target_checkpoint_domain"])
    target_rows = [dict(row) for row in rows if checkpoint_domain(str(row["parent_group_id"])) == target_domain]
    positive = sum(int(row["label"]) for row in target_rows)
    groups = sorted({str(row["parent_group_id"]) for row in target_rows})
    expected = protocol["expected_target_domain"]
    if (
        len(target_rows) != int(expected["candidate_count"])
        or positive != int(expected["positive_count"])
        or len(target_rows) - positive != int(expected["negative_count"])
        or len(groups) != int(expected["parent_group_count"])
    ):
        raise ValueError("matched-panel target-domain semantic counts drift")

    catalogs = _catalog_paths(protocol["catalog_search_roots"], protocol["catalog_filenames"])
    metadata, scan = _candidate_metadata_by_state(target_rows, catalogs)
    panel = protocol["matched_panel"]
    matched = [row for row in metadata.values() if _is_target_panel(row, panel)]
    matched.sort(
        key=lambda row: (
            row["parent_group_id"], row["duration"], row["strength"], row["action_name"], row["sign"]
        )
    )

    cell_counts: Counter[tuple[Any, ...]] = Counter()
    parent_stats: dict[str, dict[str, Any]] = {}
    for row in matched:
        cell = (
            str(row["parent_group_id"]),
            int(row["duration"]),
            round(float(row["strength"]), 9),
            str(row["action_name"]),
            int(row["sign"]),
        )
        cell_counts[cell] += 1
    duplicate_cells = [cell for cell, count in cell_counts.items() if count != 1]

    expected_cells: set[tuple[Any, ...]] = set()
    for group in groups:
        for duration in panel["durations"]:
            for strength in panel["strengths"]:
                for action_name in panel["action_names"]:
                    for sign in panel["signs"]:
                        expected_cells.add(
                            (group, int(duration), round(float(strength), 9), str(action_name), int(sign))
                        )
    missing_cells = sorted(expected_cells.difference(cell_counts))
    extra_cells = sorted(set(cell_counts).difference(expected_cells))

    by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched:
        by_parent[str(row["parent_group_id"])].append(row)
    for group in groups:
        subset = by_parent[group]
        pos = sum(int(row["label"]) for row in subset)
        parent_stats[group] = {
            "candidate_count": len(subset),
            "positive_count": pos,
            "negative_count": len(subset) - pos,
            "both_labels_present": bool({int(row["label"]) for row in subset} == {0, 1}),
        }

    matched_positive = sum(int(row["label"]) for row in matched)
    complete = bool(
        len(matched) == int(panel["expected_total_cells"])
        and not missing_cells
        and not extra_cells
        and not duplicate_cells
        and all(
            stats["candidate_count"] == int(panel["expected_cells_per_parent"])
            for stats in parent_stats.values()
        )
    )
    both_labels_all = bool(all(stats["both_labels_present"] for stats in parent_stats.values()))
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "protocol_sha256": str(config["expected_protocol_sha256"]),
        "target_checkpoint_domain": target_domain,
        "target_domain_candidate_count": len(target_rows),
        "target_domain_reconstructed_candidate_count": len(metadata),
        "target_domain_unreconstructed_candidate_count": len(target_rows) - len(metadata),
        "matched_panel_candidate_count": len(matched),
        "matched_panel_positive_count": matched_positive,
        "matched_panel_negative_count": len(matched) - matched_positive,
        "matched_panel_parent_group_count": len(parent_stats),
        "matched_panel_parent_stats": dict(sorted(parent_stats.items())),
        "matched_panel_complete": complete,
        "matched_panel_both_labels_in_every_parent": both_labels_all,
        "missing_cell_count": len(missing_cells),
        "extra_cell_count": len(extra_cells),
        "duplicate_cell_count": len(duplicate_cells),
        "missing_cells_preview": [list(cell) for cell in missing_cells[:20]],
        "catalog_scan": scan,
        "matched_state_sha256": [str(row["state_sha256"]) for row in matched],
        "environment_interactions": 0,
        "training_transitions": 0,
        "consumed_validation_rows_read": False,
        "consumed_validation_predictions_read": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "matched_panel_freeze_authorized": bool(complete and both_labels_all),
        "fresh_validation_bank_predeclared": False,
        "tube_1_authorized": False,
        "next_scientific_gate": (
            "if matched_panel_freeze_authorized, freeze the reconstructed transition_4988928 matched subset "
            "with the already matched transition_7987200/9977856 TRAIN rows and rerun leave-one-checkpoint-domain-out; "
            "otherwise collect only the missing transition_4988928 matched-panel cells before any model change"
        ),
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
