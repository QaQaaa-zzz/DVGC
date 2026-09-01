"""TRAIN-only fair leave-one-checkpoint-domain-out diagnostic for C_up^0.

This wrapper uses the same fold fitter, model family, optimizer, and gate as the
existing checkpoint-domain diagnostic. The only change is the matched 240-cell
panel per checkpoint domain, removing the legacy acquisition-family confound.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .iteration_train_evidence import canonical_sha256
from .upstream_checkpoint_domain_cv import (
    _fit_fold,
    _validate_rows,
    classify_checkpoint_domain_folds,
)
from .upstream_checkpoint_train_evidence import load_frozen_upstream_checkpoint_train_evidence

CONFIG_SCHEMA = "jit_upstream_checkpoint_domain_logo_config_v1"
PROTOCOL_SCHEMA = "jit_upstream_checkpoint_domain_logo_protocol_v1"
SUMMARY_SCHEMA = "jit_upstream_checkpoint_domain_logo_summary_v1"
STATUS = "predeclared_after_matched_panel_freeze_before_checkpoint_cv"


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


def load_matched_checkpoint_domain_cv_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported matched checkpoint-domain LOGO config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("matched checkpoint-domain LOGO protocol drift")
    if protocol.get("status") != STATUS:
        raise ValueError("matched checkpoint-domain LOGO status drift")
    if int(protocol.get("iteration", -1)) != 0 or protocol.get("policy_name") != "pi_0":
        raise ValueError("matched checkpoint-domain LOGO policy drift")
    for field in (
        "policy_actor_sha256",
        "policy_payload_sha256",
        "frozen_upstream_train_freeze_protocol_sha256",
    ):
        _sha(protocol.get(field), field=field)
    if protocol.get("expected_combined") != {
        "candidate_count": 720,
        "positive_count": 639,
        "negative_count": 81,
        "parent_group_count": 15,
        "checkpoint_domain_count": 3,
    }:
        raise ValueError("matched checkpoint-domain expected counts drift")
    if protocol.get("required_domains") != [
        "transition_4988928",
        "transition_7987200",
        "transition_9977856",
    ]:
        raise ValueError("matched checkpoint-domain required domain order drift")
    if protocol.get("model") != {
        "family": "linear_logistic",
        "input": "unified_actor_observation",
        "observation_size": 76,
        "normalization": "fold_train_only_zscore_clip10",
        "sample_weighting": "equal_parent_label_cell_mass",
        "l2_weight": 0.01,
        "optimizer": "adam_full_batch_fixed_schedule",
        "steps": 4000,
        "learning_rate": 0.01,
        "seed_base": 846000,
    }:
        raise ValueError("matched checkpoint-domain model contract drift")
    if protocol.get("diagnostic_gate") != {
        "minimum_mean_logo_roc_auc": 0.70,
        "minimum_worst_logo_roc_auc": 0.60,
        "require_positive_score_gap_in_every_fold": True,
    }:
        raise ValueError("matched checkpoint-domain gate drift")
    if protocol.get("data_policy") != {
        "train_rows_only": True,
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("matched checkpoint-domain data policy drift")
    if protocol.get("interpretation") != {
        "checkpoint_domain_is_parent_group_transition_prefix": True,
        "all_three_domains_use_same_locked_matched_panel": True,
        "acquisition_family_confound_removed": True,
        "failure_can_be_interpreted_as_checkpoint_domain_or_representation_generalization_limit": True,
    }:
        raise ValueError("matched checkpoint-domain interpretation drift")
    if protocol.get("claim_boundary") != {
        "diagnostic_only": True,
        "continuation_field_reselected": False,
        "fresh_validation_bank_predeclared": False,
        "tube_1_constructed": False,
        "pi_1_trained": False,
        "jce_jel_claim": False,
        "certified_safe_set_claim": False,
    }:
        raise ValueError("matched checkpoint-domain claim boundary drift")
    if not str(protocol.get("frozen_upstream_train_evidence", "")):
        raise ValueError("matched checkpoint-domain frozen evidence missing")
    if not str(config.get("output_dir", "")):
        raise ValueError("matched checkpoint-domain output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("matched checkpoint-domain protocol SHA drift")
    return config


def run_matched_checkpoint_domain_cv(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_matched_checkpoint_domain_cv_config(config_path)
    protocol = dict(config["protocol"])
    manifest, raw_rows = load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["frozen_upstream_train_evidence"]))
    )
    if manifest.get("freeze_protocol_sha256") != protocol["frozen_upstream_train_freeze_protocol_sha256"]:
        raise ValueError("matched checkpoint-domain frozen TRAIN protocol drift")
    if manifest.get("policy_actor_sha256") != protocol["policy_actor_sha256"]:
        raise ValueError("matched checkpoint-domain frozen actor drift")
    if manifest.get("policy_payload_sha256") != protocol["policy_payload_sha256"]:
        raise ValueError("matched checkpoint-domain frozen payload drift")
    if manifest.get("matched_panel") is not True:
        raise ValueError("matched checkpoint-domain requires a matched-panel frozen artifact")
    rows = _validate_rows(raw_rows, protocol)
    domains = list(protocol["required_domains"])
    folds: list[dict[str, Any]] = []
    for index, heldout_domain in enumerate(domains):
        heldout = [row for row in rows if row["parent_domain_id"] == heldout_domain]
        train = [row for row in rows if row["parent_domain_id"] != heldout_domain]
        folds.append(
            _fit_fold(
                train,
                heldout,
                heldout_domain=heldout_domain,
                model_cfg=protocol["model"],
                seed=int(protocol["model"]["seed_base"]) + index,
            )
        )
    classification = classify_checkpoint_domain_folds(folds, protocol["diagnostic_gate"])
    supported = bool(classification["checkpoint_domain_generalization_supported"])
    diagnosis = (
        "current_linear_model_generalizes_across_three_matched_train_checkpoint_domains"
        if supported
        else "current_linear_model_does_not_generalize_across_three_matched_train_checkpoint_domains"
    )
    next_gate = (
        "freeze this matched checkpoint-domain CV result and predeclare one fresh independent upstream validation bank; "
        "do not reuse consumed validation and do not construct Tube_1 yet"
        if supported
        else
        "acquisition-family confounding is removed; do not reuse consumed validation; revise the TRAIN-side upstream "
        "continuation representation/model before any fresh validation bank is predeclared"
    )
    output = Path(str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "artifact_role": "train_only_upstream_matched_checkpoint_domain_generalization_diagnostic",
        "iteration": 0,
        "policy_name": "pi_0",
        "policy_actor_sha256": str(protocol["policy_actor_sha256"]),
        "policy_payload_sha256": str(protocol["policy_payload_sha256"]),
        "protocol_sha256": str(config["expected_protocol_sha256"]),
        "frozen_upstream_train_manifest_sha256": str(manifest["manifest_sha256"]),
        "candidate_count": len(rows),
        "positive_count": sum(int(row["label"]) for row in rows),
        "negative_count": sum(1 - int(row["label"]) for row in rows),
        "parent_group_count": len({str(row["parent_group_id"]) for row in rows}),
        "checkpoint_domains": domains,
        "fold_count": len(folds),
        "folds": folds,
        **classification,
        "diagnosis": diagnosis,
        "interpretation": dict(protocol["interpretation"]),
        "environment_interactions": 0,
        "training_transitions": 0,
        "supervised_optimizer_steps": len(folds) * int(protocol["model"]["steps"]),
        "consumed_validation_rows_reused": False,
        "consumed_validation_predictions_reused": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "fresh_validation_predeclaration_authorized": supported,
        "tube_1_authorized": False,
        "next_scientific_gate": next_gate,
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
