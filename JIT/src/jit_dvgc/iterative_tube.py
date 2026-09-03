"""Build Tube_(k+1) from Tube_k and calibrated C^k for k >= 1."""
from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import numpy as np

from .config import file_sha256
from .iterative_continuation_fields import _score
from .iterative_frontier_protocol import ROLE_SCHEMA, canonical_sha256
from .soft_tube import PHASE_MIXTURE, SOFT_TUBE_SCHEMA, WEIGHT_FLOOR, WEIGHT_SCALE, load_soft_tube
from .unified_envelope_snapshot import load_unified_envelope_snapshot, physical_state_sha256


SCHEMA = "jit_iterative_core_retaining_tube_v1"


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_hash(payload: Mapping[str, Any], field: str) -> None:
    declared = str(payload.get(field, ""))
    base = {key: value for key, value in payload.items() if key != field}
    if len(declared) != 64 or canonical_sha256(base) != declared:
        raise ValueError(f"{field} self-hash drift")


def _load_train_role(root: Path):
    root = Path(root)
    manifest = _read(root / "role_manifest.json")
    if not isinstance(manifest, dict) or manifest.get("schema") != ROLE_SCHEMA:
        raise ValueError("iterative Tube TRAIN role schema drift")
    if manifest.get("status") != "completed" or manifest.get("role") != "train":
        raise ValueError("iterative Tube requires completed TRAIN frontier role")
    _verify_hash(manifest, "role_manifest_sha256")
    labels_path = root / "logical_labels.json"
    if file_sha256(labels_path) != manifest["logical_labels_file_sha256"]:
        raise ValueError("iterative Tube TRAIN labels file drift")
    payload = _read(labels_path)
    if not isinstance(payload, dict) or payload.get("role") != "train":
        raise ValueError("iterative Tube logical TRAIN labels drift")
    _verify_hash(payload, "labels_sha256")
    rows = payload.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("iterative Tube TRAIN labels empty")
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("iterative Tube can embed TRAIN rows only")
    return manifest, tuple(dict(row) for row in rows)


def _load_fields(root: Path):
    root = Path(root)
    summary = _read(root / "summary.json")
    if not isinstance(summary, dict) or summary.get("schema") != "jit_iterative_continuation_fields_v1":
        raise ValueError("iterative Tube continuation summary schema drift")
    if summary.get("status") != "completed_calibrated" or summary.get("next_tube_construction_authorized") is not True:
        raise ValueError("iterative Tube continuation fields are not calibrated/authorized")
    _verify_hash(summary, "summary_sha256")
    phases = {}
    for phase in ("upstream", "downstream"):
        manifest = _read(root / phase / "manifest.json")
        calibration = _read(root / phase / "calibration.json")
        _verify_hash(manifest, "manifest_sha256")
        _verify_hash(calibration, "calibration_sha256")
        if calibration.get("calibration_passed") is not True:
            raise ValueError(f"iterative Tube {phase} calibration did not pass")
        if calibration.get("field_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"iterative Tube {phase} field/calibration drift")
        field_path = root / phase / "field.npz"
        if file_sha256(field_path) != manifest["field_file_sha256"]:
            raise ValueError(f"iterative Tube {phase} field file drift")
        phases[phase] = {
            "manifest": manifest,
            "calibration": calibration,
            "field_path": field_path,
        }
    return summary, phases


def build_iterative_tube(*, source_tube: Path, train_root: Path, fields_root: Path, output_dir: Path):
    source = load_soft_tube(Path(source_tube))
    train_manifest, train_rows = _load_train_role(Path(train_root))
    fields_summary, fields = _load_fields(Path(fields_root))
    source_iteration = int(source.manifest.get("iteration", 0))
    iteration = source_iteration + 1
    if source_iteration < 1:
        raise ValueError("generic iterative Tube builder is for Tube_k with k>=1")
    for name, value in (
        ("train iteration", int(train_manifest["iteration"])),
        ("field iteration", int(fields_summary["iteration"])),
    ):
        if value != source_iteration:
            raise ValueError(f"{name} must equal source Tube iteration")
    if train_manifest["source_tube_manifest_sha256"] != source.manifest["manifest_sha256"]:
        raise ValueError("iterative Tube TRAIN/source Tube identity drift")
    if fields_summary["source_tube_manifest_sha256"] != source.manifest["manifest_sha256"]:
        raise ValueError("iterative Tube fields/source Tube identity drift")
    if fields_summary["train_role_manifest_sha256"] != train_manifest["role_manifest_sha256"]:
        raise ValueError("iterative Tube field TRAIN role drift")

    source_states = {str(row["state_sha256"]) for row in source.entries}
    if len(source_states) != len(source.entries):
        raise ValueError("source Tube contains duplicate physical states")
    acquisition_catalog = Path(str(train_manifest["source_acquisition_catalog"]))
    snapshot_root = acquisition_catalog.parent

    selected = []
    selection_counts = {}
    for phase in ("upstream", "downstream"):
        rows = [row for row in train_rows if row.get("phase") == phase]
        if not rows:
            raise ValueError(f"iterative Tube has no {phase} TRAIN rows")
        scores = _score(fields[phase]["field_path"], rows)
        threshold = float(fields[phase]["calibration"]["acceptance_threshold_exclusive"])
        counts = Counter()
        for row, score in zip(rows, scores, strict=True):
            if int(row["label"]) != 1:
                counts["negative_label"] += 1
                continue
            if not float(score) > threshold:
                counts["positive_below_or_equal_threshold"] += 1
                continue
            selected.append((dict(row), float(score)))
            counts["positive_above_threshold"] += 1
        selection_counts[phase] = dict(sorted(counts.items()))
    if not selected:
        raise ValueError("iterative Tube produced no expansion states")

    entries = [dict(row) for row in source.entries]
    seen = set(source_states)
    expansion = []
    dedup = Counter()
    for row, score in sorted(
        selected,
        key=lambda item: (
            str(item[0]["phase"]),
            str(item[0]["parent_group_id"]),
            str(item[0]["state_sha256"]),
        ),
    ):
        state = str(row["state_sha256"])
        if state in seen:
            dedup["source_tube_or_expansion_duplicate"] += 1
            continue
        snapshot_path = snapshot_root / str(row["source_bank"]) / str(row["snapshot"])
        if not snapshot_path.is_dir():
            raise FileNotFoundError(f"iterative Tube snapshot missing: {snapshot_path}")
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        if physical_state_sha256(snapshot) != state:
            raise ValueError("iterative Tube snapshot physical-state drift")
        phase = str(row["phase"])
        phase_index = 0 if phase == "upstream" else 1
        if int(snapshot.active_phase) != phase_index:
            raise ValueError("iterative Tube snapshot active-phase drift")
        field_name = f"C_{'up' if phase == 'upstream' else 'down'}^{source_iteration}"
        expansion_row = {
            "candidate_id": f"tube{iteration}_{phase}_{state}",
            "source_candidate_id": str(row["candidate_id"]),
            "candidate_kind": "policy_conditioned_iterative_train_probe",
            "split": "train",
            "phase": phase,
            "phase_index": phase_index,
            "snapshot_schema": "jit_unified_envelope_snapshot_v1",
            "snapshot": str(snapshot_path.resolve()),
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
                "field_manifest_sha256": fields[phase]["manifest"]["manifest_sha256"],
                "field_file_sha256": fields[phase]["manifest"]["field_file_sha256"],
                "acceptance_threshold_exclusive": float(
                    fields[phase]["calibration"]["acceptance_threshold_exclusive"]
                ),
                "selection_rule": "TRAIN_label_positive_and_score_strictly_greater_than_threshold",
                "threshold_source": "disjoint_calibration_role",
            },
            "source_train_role_manifest_sha256": train_manifest["role_manifest_sha256"],
            "policy_actor_sha256": str(row["policy_actor_sha256"]),
            "policy_payload_sha256": str(row["policy_payload_sha256"]),
        }
        expansion.append(expansion_row)
        entries.append(expansion_row)
        seen.add(state)
    if not expansion:
        raise ValueError("iterative Tube expansion vanished after deduplication")

    phase_counts = Counter(str(row["phase"]) for row in entries)
    expansion_phase_counts = Counter(str(row["phase"]) for row in expansion)
    if set(phase_counts) != {"upstream", "downstream"}:
        raise ValueError("iterative Tube lost phase support")
    diagnostics = {
        "schema": "jit_iterative_tube_diagnostics_v1",
        "status": "completed",
        "source_tube_entry_count": len(source.entries),
        "source_tube_phase_counts": dict(sorted(Counter(str(row["phase"]) for row in source.entries).items())),
        "selection_counts": selection_counts,
        "selected_before_dedup_count": len(selected),
        "dedup_counts": dict(sorted(dedup.items())),
        "expansion_count": len(expansion),
        "expansion_phase_counts": dict(sorted(expansion_phase_counts.items())),
        "entry_count": len(entries),
        "phase_counts": dict(sorted(phase_counts.items())),
        "calibration_rows_embedded": 0,
        "acceptance_rows_embedded": 0,
        "test_rows_embedded": 0,
    }
    manifest = {
        "schema": SOFT_TUBE_SCHEMA,
        "status": "completed",
        "artifact_role": "policy_conditioned_core_retaining_soft_tube_iteration",
        "iteration": iteration,
        "source_iteration": source_iteration,
        "training_guidance_only": True,
        "certified_safe": False,
        "test_data_used": False,
        "validation_data_used": False,
        "calibration_data_used": True,
        "calibration_rows_embedded": False,
        "acceptance_rows_embedded": False,
        "training_transitions": 0,
        "environment_interactions": 0,
        "source_tube_manifest_sha256": source.manifest["manifest_sha256"],
        "source_tube_entry_count": len(source.entries),
        "core_retained_count": len(source.entries),
        "newest_expansion_count": len(expansion),
        "expansion_count": len(expansion),
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
            "core_score_source": "preserve_every_source_Tube score/weight exactly",
            "expansion_score_source": f"frozen C_up^{source_iteration}/C_down^{source_iteration} score",
            "calibration_tuned_model_parameters": False,
        },
        "continuation_fields_summary_sha256": fields_summary["summary_sha256"],
        "train_role_manifest_sha256": train_manifest["role_manifest_sha256"],
        "calibration_role_manifest_sha256": fields_summary["calibration_role_manifest_sha256"],
        "selection_rule": (
            "retain every Tube_k entry exactly; add only logical-TRAIN continuation-positive "
            "states whose frozen C^k score is strictly above its disjoint calibration threshold; "
            "source Tube wins duplicates; no quota/replacement"
        ),
        "score_semantics": "policy-conditioned empirical continuation score; not certified probability",
        "claim_boundary": {
            "soft_tube_training_guidance_only": True,
            "core_retaining": True,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    manifest["entries_sha256"] = canonical_sha256({"entries": entries})
    manifest["diagnostics_sha256"] = canonical_sha256(diagnostics)
    manifest["manifest_sha256"] = canonical_sha256(manifest)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"iterative Tube output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _write(temporary / "entries.json", entries)
        _write(temporary / "diagnostics.json", diagnostics)
        _write(temporary / "manifest.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {"schema": SCHEMA, "status": "completed", "output_dir": str(output), "manifest": manifest, "diagnostics": diagnostics}
