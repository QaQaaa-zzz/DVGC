"""Core-retaining policy-conditioned Soft-Tube iteration construction.

Tube_1 is the union of the immutable Tube_0 core and expansion TRAIN states
that are both continuation-positive and strictly above the fresh-validation-
calibrated phase threshold under the frozen C_up^0/C_down^0 fields.

Fresh validation rows are never copied into the Tube. Validation contributes
only the already-frozen phase thresholds and artifact identities.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .config import file_sha256
from .iteration_train_evidence import canonical_sha256, load_frozen_iteration_train_evidence
from .soft_tube import PHASE_MIXTURE, SOFT_TUBE_SCHEMA, WEIGHT_FLOOR, WEIGHT_SCALE, SoftTubeArtifact, load_soft_tube
from .upstream_checkpoint_train_evidence import load_frozen_upstream_checkpoint_train_evidence
from .upstream_matched_checkpoint_domain_cv import _sigmoid

CONFIG_SCHEMA = "jit_core_retaining_tube_iteration_config_v1"
PROTOCOL_SCHEMA = "jit_core_retaining_tube_iteration_protocol_v1"
STATUS = "predeclared_after_fresh_validation_pass_before_tube1"


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _self_hash(payload: Mapping[str, Any], field: str) -> str:
    return canonical_sha256({key: value for key, value in payload.items() if key != field})


def load_core_retaining_tube_config(path: Path) -> dict[str, Any]:
    config = _read_object(Path(path))
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unsupported core-retaining Tube config")
    protocol = config.get("protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("core-retaining Tube protocol drift")
    if protocol.get("status") != STATUS:
        raise ValueError("core-retaining Tube protocol status drift")
    if int(protocol.get("iteration", -1)) != 1 or int(protocol.get("source_iteration", -1)) != 0:
        raise ValueError("core-retaining Tube iteration drift")
    if protocol.get("policy_name") != "pi_0":
        raise ValueError("core-retaining Tube policy drift")
    if protocol.get("thresholds") != {
        "upstream": 0.9333483934566058,
        "downstream": 0.8721734129976408,
        "decision_rule": "TRAIN_label_positive_and_score_strictly_greater_than_fresh_validation_threshold",
    }:
        raise ValueError("core-retaining Tube calibrated thresholds drift")
    if protocol.get("selection") != {
        "require_train_split": True,
        "require_positive_continuation_label": True,
        "strict_score_greater_than_phase_threshold": True,
        "retain_all_source_tube_core_entries": True,
        "core_wins_on_physical_state_duplicate": True,
        "deduplicate_expansion_physical_state": True,
        "no_quota_or_replacement": True,
    }:
        raise ValueError("core-retaining Tube selection contract drift")
    if protocol.get("sampling") != {
        "phase_mixture": {"upstream": 0.5, "downstream": 0.5},
        "mapping": "sampling_weight = 0.05 + 0.95 * score",
        "core_score_source": "preserve_existing_V_up_V_down_bootstrap_scores",
        "expansion_score_source": "C_up^0_or_C_down^0_policy_conditioned_continuation_score",
    }:
        raise ValueError("core-retaining Tube sampling contract drift")
    if protocol.get("data_policy") != {
        "tube_entries_from_train_only": True,
        "fresh_validation_rows_may_enter_tube": False,
        "fresh_validation_used_only_for_phase_threshold_calibration": True,
        "consumed_validation_rows_may_enter_tube": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "environment_interactions": 0,
        "training_transitions": 0,
    }:
        raise ValueError("core-retaining Tube data policy drift")
    if protocol.get("claim_boundary") != {
        "soft_tube_training_guidance_only": True,
        "certified_safe_set_claim": False,
        "tube_1_constructed_after_run": True,
        "pi_1_trained": False,
        "jce_jel_claim": False,
    }:
        raise ValueError("core-retaining Tube claim boundary drift")
    if protocol.get("snapshot_search_roots") != [
        "JIT/runs/pi_unified_transition_band",
        "JIT/runs/pi_unified_upstream_parent_diversity",
    ]:
        raise ValueError("core-retaining Tube snapshot roots drift")
    if not str(config.get("output_dir", "")):
        raise ValueError("core-retaining Tube output_dir missing")
    if canonical_sha256(protocol) != config.get("expected_protocol_sha256"):
        raise ValueError("core-retaining Tube protocol SHA drift")
    return config


def _load_field(root: Path, phase: str, protocol: Mapping[str, Any]) -> dict[str, np.ndarray]:
    manifest = _read_object(root / phase / "manifest.json")
    expected_manifest = protocol[f"{phase}_field_manifest_sha256"]
    expected_file = protocol[f"{phase}_field_file_sha256"]
    if manifest.get("manifest_sha256") != expected_manifest:
        raise ValueError(f"Tube_1 {phase} field manifest drift")
    field_path = root / phase / "field.npz"
    if file_sha256(field_path) != expected_file or manifest.get("field_file_sha256") != expected_file:
        raise ValueError(f"Tube_1 {phase} field file drift")
    if manifest.get("architecture") != "76->8_tanh->1" or manifest.get("parameter_count") != 625:
        raise ValueError(f"Tube_1 {phase} field architecture drift")
    with np.load(field_path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _scores(field: Mapping[str, np.ndarray], observations: np.ndarray) -> np.ndarray:
    mean = np.asarray(field["mean"], dtype=np.float32)
    std = np.asarray(field["std"], dtype=np.float32)
    x = np.clip((observations - mean) / std, -10.0, 10.0).astype(np.float32)
    hidden = np.tanh(x @ np.asarray(field["w1"], dtype=np.float32) + np.asarray(field["b1"], dtype=np.float32))
    logits = hidden @ np.asarray(field["w2"], dtype=np.float32) + float(np.asarray(field["b2"]))
    return _sigmoid(np.asarray(logits, dtype=np.float64))


def _verify_fresh_validation(protocol: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(protocol["fresh_validation_root"]))
    summary = _read_object(root / "summary.json")
    if summary.get("summary_sha256") != protocol["fresh_validation_summary_sha256"]:
        raise ValueError("Tube_1 fresh validation summary identity drift")
    if _self_hash(summary, "summary_sha256") != summary.get("summary_sha256"):
        raise ValueError("Tube_1 fresh validation summary self-hash drift")
    if summary.get("declared_protocol_sha256") != protocol["fresh_validation_declared_protocol_sha256"]:
        raise ValueError("Tube_1 fresh validation protocol drift")
    if summary.get("fresh_validation_passed") is not True or summary.get("tube_1_construction_authorized") is not True:
        raise ValueError("Tube_1 fresh validation did not authorize construction")
    if summary.get("fresh_validation_rows_may_enter_train_or_tube") is not False:
        raise ValueError("Tube_1 validation split isolation drift")
    if summary.get("test_data_used") is not False or summary.get("final_evaluation_data_used") is not False:
        raise ValueError("Tube_1 fresh validation used TEST/final data")
    for phase in ("upstream", "downstream"):
        calibration = _read_object(root / phase / "calibration.json")
        if calibration.get("phase") != phase or calibration.get("calibration_passed") is not True:
            raise ValueError(f"Tube_1 {phase} calibration did not pass")
        if float(calibration.get("acceptance_threshold_exclusive")) != float(protocol["thresholds"][phase]):
            raise ValueError(f"Tube_1 {phase} threshold drift")
    return summary


def _verify_sources(protocol: Mapping[str, Any]):
    source_tube = load_soft_tube(Path(str(protocol["source_tube"])))
    if source_tube.manifest.get("manifest_sha256") != protocol["source_tube_manifest_sha256"]:
        raise ValueError("Tube_1 source Tube manifest drift")
    if len(source_tube.entries) != int(protocol["source_tube_entry_count"]):
        raise ValueError("Tube_1 source Tube entry count drift")
    up_manifest, up_rows = load_frozen_upstream_checkpoint_train_evidence(Path(str(protocol["upstream_train_evidence"])))
    if up_manifest.get("manifest_sha256") != protocol["upstream_train_manifest_sha256"]:
        raise ValueError("Tube_1 upstream TRAIN manifest drift")
    down_manifest, down_rows = load_frozen_iteration_train_evidence(Path(str(protocol["downstream_train_evidence"])))
    if down_manifest.get("manifest_sha256") != protocol["downstream_train_manifest_sha256"]:
        raise ValueError("Tube_1 downstream TRAIN manifest drift")
    up_rows = tuple(dict(row) for row in up_rows if row.get("phase") == "upstream")
    down_rows = tuple(dict(row) for row in down_rows if row.get("phase") == "downstream")
    if not up_rows or not down_rows or any(row.get("split") != "train" for row in (*up_rows, *down_rows)):
        raise ValueError("Tube_1 expansion evidence split drift")
    root = Path(str(protocol["shared_refit_root"]))
    shared = _read_object(root / "summary.json")
    if shared.get("summary_sha256") != protocol["shared_refit_summary_sha256"]:
        raise ValueError("Tube_1 shared refit summary drift")
    architecture = _read_object(root / "architecture_manifest.json")
    if architecture.get("architecture_manifest_sha256") != protocol["architecture_manifest_sha256"]:
        raise ValueError("Tube_1 shared architecture drift")
    fields = {phase: _load_field(root, phase, protocol) for phase in ("upstream", "downstream")}
    return source_tube, up_rows, down_rows, fields


def _scan_unified_snapshots(search_roots: Sequence[Path], target_states: set[str]) -> dict[str, tuple[Path, ...]]:
    found: dict[str, list[Path]] = defaultdict(list)
    for root in search_roots:
        if not root.is_dir():
            raise FileNotFoundError(f"Tube_1 snapshot search root missing: {root}")
        for identity_path in root.rglob("identity.json"):
            try:
                identity = json.loads(identity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if identity.get("schema") != "jit_unified_envelope_snapshot_v1":
                continue
            state = str(identity.get("physical_state_sha256", ""))
            if state in target_states:
                found[state].append(identity_path.parent.resolve())
    return {state: tuple(sorted(paths)) for state, paths in found.items()}


def _select_phase(rows: Sequence[Mapping[str, Any]], *, phase: str, field: Mapping[str, np.ndarray], threshold: float):
    observations = np.asarray([row["actor_observation"] for row in rows], dtype=np.float32)
    if observations.ndim != 2 or observations.shape[1] != 76:
        raise ValueError(f"Tube_1 {phase} TRAIN observation shape drift")
    scores = _scores(field, observations)
    selected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for row, score in zip(rows, scores, strict=True):
        if int(row.get("label", -1)) != 1:
            counts["negative_label"] += 1
            continue
        if not float(score) > float(threshold):
            counts["positive_below_or_equal_threshold"] += 1
            continue
        selected.append({**dict(row), "_continuation_score": float(score)})
        counts["positive_above_threshold"] += 1
    return selected, dict(sorted(counts.items()))


def build_core_retaining_tube(config_path: Path) -> SoftTubeArtifact:
    config_path = Path(config_path)
    config = load_core_retaining_tube_config(config_path)
    protocol = config["protocol"]
    _verify_fresh_validation(protocol)
    source_tube, up_rows, down_rows, fields = _verify_sources(protocol)
    selected_up, up_selection = _select_phase(up_rows, phase="upstream", field=fields["upstream"], threshold=float(protocol["thresholds"]["upstream"]))
    selected_down, down_selection = _select_phase(down_rows, phase="downstream", field=fields["downstream"], threshold=float(protocol["thresholds"]["downstream"]))
    selected = [*selected_up, *selected_down]
    target_states = {str(row["state_sha256"]) for row in selected}
    snapshot_index = _scan_unified_snapshots([Path(str(value)) for value in protocol["snapshot_search_roots"]], target_states)
    missing = sorted(target_states.difference(snapshot_index))
    if missing:
        raise FileNotFoundError(f"Tube_1 cannot resolve {len(missing)} selected TRAIN unified snapshots; first={missing[0]}")

    from .unified_envelope_snapshot import load_unified_envelope_snapshot, physical_state_sha256 as unified_state_sha256

    core_rows = [dict(row) for row in source_tube.entries]
    core_states = {str(row["state_sha256"]) for row in core_rows}
    if len(core_states) != len(core_rows):
        raise ValueError("Tube_1 source core contains duplicate physical states")
    expansion_rows: list[dict[str, Any]] = []
    seen = set(core_states)
    dedup = Counter()
    alias_counts = Counter()
    for row in sorted(selected, key=lambda value: (str(value["phase"]), str(value["parent_group_id"]), str(value["state_sha256"]))):
        state = str(row["state_sha256"])
        if state in core_states:
            dedup["core_wins"] += 1
            continue
        if state in seen:
            dedup["expansion_duplicate"] += 1
            continue
        aliases = snapshot_index[state]
        alias_counts[len(aliases)] += 1
        snapshot_path = aliases[0]
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        if unified_state_sha256(snapshot) != state:
            raise ValueError("Tube_1 resolved snapshot physical identity drift")
        phase = str(row["phase"])
        expected_phase_index = 0 if phase == "upstream" else 1
        if int(snapshot.active_phase) != expected_phase_index:
            raise ValueError("Tube_1 expansion snapshot phase drift")
        score = float(row["_continuation_score"])
        field_name = "C_up^0" if phase == "upstream" else "C_down^0"
        expansion_rows.append({
            "candidate_id": f"tube1_{phase}_{state}",
            "source_candidate_id": str(row["candidate_id"]),
            "candidate_kind": str(row.get("candidate_kind", "policy_conditioned_train_probe")),
            "split": "train",
            "phase": phase,
            "phase_index": expected_phase_index,
            "snapshot_schema": "jit_unified_envelope_snapshot_v1",
            "snapshot": str(snapshot_path),
            "state_sha256": state,
            "parent_group_id": str(row["parent_group_id"]),
            "parent_state_sha256": str(row["parent_state_sha256"]),
            "label_target": 1.0,
            "continuation_label": 1,
            "value_score": score,
            "sampling_weight": WEIGHT_FLOOR + WEIGHT_SCALE * score,
            "value_model_target": field_name,
            "score_source": {
                "kind": "policy_conditioned_continuation_field",
                "field_name": field_name,
                "field_manifest_sha256": str(protocol[f"{phase}_field_manifest_sha256"]),
                "field_file_sha256": str(protocol[f"{phase}_field_file_sha256"]),
                "acceptance_threshold_exclusive": float(protocol["thresholds"][phase]),
                "selection_rule": "TRAIN_label_positive_and_score_strictly_greater_than_threshold",
            },
            "source_train_manifest_sha256": str(protocol["upstream_train_manifest_sha256" if phase == "upstream" else "downstream_train_manifest_sha256"]),
            "policy_actor_sha256": str(row["policy_actor_sha256"]),
            "policy_payload_sha256": str(row["policy_payload_sha256"]),
        })
        seen.add(state)
    if not expansion_rows:
        raise ValueError("Tube_1 selection produced no expansion states")

    entries = tuple([*core_rows, *expansion_rows])
    phase_counts = Counter(str(row["phase"]) for row in entries)
    expansion_phase_counts = Counter(str(row["phase"]) for row in expansion_rows)
    if set(phase_counts) != {"upstream", "downstream"}:
        raise ValueError("Tube_1 lost phase support")
    diagnostics = {
        "schema": "jit_core_retaining_tube_iteration_diagnostics_v1",
        "status": "completed",
        "source_core_count": len(core_rows),
        "source_core_phase_counts": dict(sorted(Counter(str(row["phase"]) for row in core_rows).items())),
        "upstream_train_candidate_count": len(up_rows),
        "downstream_train_candidate_count": len(down_rows),
        "selection_counts": {"upstream": up_selection, "downstream": down_selection},
        "selected_before_core_dedup_count": len(selected),
        "dedup_counts": dict(sorted(dedup.items())),
        "snapshot_alias_multiplicity_counts": {str(key): value for key, value in sorted(alias_counts.items())},
        "expansion_count": len(expansion_rows),
        "expansion_phase_counts": dict(sorted(expansion_phase_counts.items())),
        "tube_1_entry_count": len(entries),
        "tube_1_phase_counts": dict(sorted(phase_counts.items())),
        "validation_rows_embedded": 0,
        "test_rows_embedded": 0,
    }
    manifest = {
        "schema": SOFT_TUBE_SCHEMA,
        "status": "completed",
        "artifact_role": "policy_conditioned_core_retaining_soft_tube_iteration",
        "iteration": 1,
        "source_iteration": 0,
        "training_guidance_only": True,
        "certified_safe": False,
        "test_data_used": False,
        "validation_data_used": False,
        "validation_calibration_used": True,
        "validation_rows_embedded": False,
        "training_transitions": 0,
        "environment_interactions": 0,
        "source_tube_manifest_sha256": str(protocol["source_tube_manifest_sha256"]),
        "source_tube_entry_count": len(core_rows),
        "core_retained_count": len(core_rows),
        "expansion_count": len(expansion_rows),
        "entry_count": len(entries),
        "upstream_count": int(phase_counts["upstream"]),
        "downstream_count": int(phase_counts["downstream"]),
        "phase_mixture": PHASE_MIXTURE,
        "weighting": {
            "mapping": "sampling_weight = 0.05 + 0.95 * score",
            "floor": WEIGHT_FLOOR,
            "scale": WEIGHT_SCALE,
            "monotonic": True,
            "nonzero_support": True,
            "core_score_source": "preserved Tube_0 V_up/V_down bootstrap score",
            "expansion_score_source": "frozen C_up^0/C_down^0 continuation score",
            "validation_tuned_model_parameters": False,
        },
        "shared_continuation_fields": {
            "architecture_manifest_sha256": str(protocol["architecture_manifest_sha256"]),
            "upstream_field_manifest_sha256": str(protocol["upstream_field_manifest_sha256"]),
            "upstream_field_file_sha256": str(protocol["upstream_field_file_sha256"]),
            "downstream_field_manifest_sha256": str(protocol["downstream_field_manifest_sha256"]),
            "downstream_field_file_sha256": str(protocol["downstream_field_file_sha256"]),
        },
        "calibration": {
            "fresh_validation_summary_sha256": str(protocol["fresh_validation_summary_sha256"]),
            "fresh_validation_declared_protocol_sha256": str(protocol["fresh_validation_declared_protocol_sha256"]),
            "upstream_threshold_exclusive": float(protocol["thresholds"]["upstream"]),
            "downstream_threshold_exclusive": float(protocol["thresholds"]["downstream"]),
        },
        "train_evidence": {
            "upstream_manifest_sha256": str(protocol["upstream_train_manifest_sha256"]),
            "downstream_manifest_sha256": str(protocol["downstream_train_manifest_sha256"]),
        },
        "selection_rule": "retain all Tube_0 core; add TRAIN continuation-positive states whose frozen phase-field score is strictly above the fresh-validation-calibrated phase threshold; core wins duplicates; no quota/replacement",
        "score_semantics": "core preserves bootstrap value weights; expansion uses policy-conditioned continuation scores",
        "protocol_sha256": str(config["expected_protocol_sha256"]),
        "config_file_sha256": file_sha256(config_path),
        "claim_boundary": dict(protocol["claim_boundary"]),
    }
    manifest["entries_sha256"] = canonical_sha256({"entries": list(entries)})
    manifest["diagnostics_sha256"] = canonical_sha256(diagnostics)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    output = Path(str(config["output_dir"]))
    if output.exists():
        raise FileExistsError(f"Tube_1 output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write_json(temporary / "entries.json", list(entries))
        _write_json(temporary / "diagnostics.json", diagnostics)
        _write_json(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return SoftTubeArtifact(output, manifest, entries, diagnostics)
