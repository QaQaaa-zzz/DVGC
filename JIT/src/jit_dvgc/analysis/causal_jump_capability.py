"""Build the actual causal Jump-Capability evidence set.

The raw Soft Tube is training/replay support and may contain historical RSI-only
states. The causal Jump Capability Tube is stricter: a state contributes only
when (1) it was reached from the locked natural ground start using env.step
only, and (2) continuation evaluation from that exact state succeeded.

This module joins each causal acquisition catalog with its continuation labels,
projects the reached states into the declared physical resolution, and keeps
TRAIN/CALIBRATION/ACCEPTANCE roles separate. The centerline is the ground-
reachable successful backbone. Only TRAIN-positive cells may extend curriculum
support. CALIBRATION/ACCEPTANCE positive cells remain holdout evidence.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..acquisition.causal_jump import (
    ACQUISITION_MODE,
    validate_ground_reachability_payload,
)
from ..config import load_config
from ..model import load_host_model
from ..unified_envelope_snapshot import (
    load_unified_envelope_snapshot,
    physical_state_sha256,
)
from .capability_tube import (
    RESOLUTIONS,
    ROOT_GEOMETRY_FIELDS,
    physical_coordinates_from_arrays,
    quantize_coordinates,
    resolution_contract,
)
from .nominal_jump_centerline import load_nominal_jump_centerline


SCHEMA = "jit_causal_jump_capability_evidence_v1"
CELLS_SCHEMA = "jit_causal_jump_capability_cells_v1"
ROLES = ("train", "calibration", "acceptance")


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_cell_id(phase: str, bins: Mapping[str, int]) -> str:
    return _canonical_sha256(
        {"phase": str(phase), "profile": "root_geometry_v1", "bins": dict(bins)}
    )


def _phase_for_centerline_point(point: Mapping[str, Any]) -> str:
    semantics = str(point["phase_semantics"])
    return "downstream" if semantics == "downstream" else "upstream"


def _centerline_cells(centerline: Mapping[str, Any], bundle: Any) -> list[dict[str, Any]]:
    loaded = np.load(Path(str(centerline["canonical_trace_npz"])), allow_pickle=False)
    qpos = np.asarray(loaded["qpos"])
    qvel = np.asarray(loaded["qvel"])
    result: list[dict[str, Any]] = []
    for point in centerline["points"]:
        index = int(point["frame_index"])
        phase = _phase_for_centerline_point(point)
        coords = physical_coordinates_from_arrays(qpos[index], qvel[index], bundle=bundle)
        bins = quantize_coordinates(coords, ROOT_GEOMETRY_FIELDS)
        state_sha = str(point["physical_state_sha256"])
        result.append(
            {
                "evidence_kind": "locked_nominal_centerline",
                "role": "centerline",
                "label": 1,
                "phase": phase,
                "state_sha256": state_sha,
                "root_geometry_cell_id": _root_cell_id(phase, bins),
                "root_geometry_bins": bins,
                "coordinates": coords,
                "x_bin": int(bins["root_x_m"]),
                "x_center_m": int(bins["root_x_m"])
                * float(RESOLUTIONS["root_x_m"]["resolution"]),
                "ground_reachability_proven": True,
                "continuation_viability_proven": True,
                "source": str(centerline["canonical_trace_npz"]),
            }
        )
    return result


def _role_evidence(role_root: Path, role: str, bundle: Any) -> list[dict[str, Any]]:
    root = Path(role_root)
    catalog_path = root / "acquisition" / "catalog.json"
    labels_path = root / "labels" / "labels.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if catalog.get("schema") != "jit_unified_boundary_catalog_v1" or catalog.get("status") != "completed":
        raise ValueError(f"{role} causal acquisition catalog schema/status drift")
    if catalog.get("acquisition_mode") != ACQUISITION_MODE:
        raise ValueError(f"{role} acquisition is not ground-connected causal mode")
    if catalog.get("ground_reachability_proven") is not True:
        raise ValueError(f"{role} acquisition lacks reachability proof")
    if catalog.get("rsi_used_for_reachability") is not False:
        raise ValueError(f"{role} acquisition used RSI to establish reachability")
    if not isinstance(labels, list):
        raise ValueError(f"{role} labels must be a list")

    labels_by_id = {str(row["candidate_id"]): row for row in labels}
    entries = catalog.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{role} causal catalog has no entries")
    if set(labels_by_id) != {str(row["candidate_id"]) for row in entries}:
        raise ValueError(f"{role} causal catalog/label candidate identity mismatch")

    result: list[dict[str, Any]] = []
    for row in entries:
        provenance = row.get("ground_reachability")
        if not isinstance(provenance, dict):
            raise ValueError("causal candidate missing ground_reachability object")
        declared_reachability_sha = str(provenance.get("reachability_sha256", ""))
        base = {k: v for k, v in provenance.items() if k != "reachability_sha256"}
        if len(declared_reachability_sha) != 64 or _canonical_sha256(base) != declared_reachability_sha:
            raise ValueError("causal reachability provenance self-hash drift")
        validate_ground_reachability_payload(provenance)

        label = labels_by_id[str(row["candidate_id"])]
        snapshot_path = catalog_path.parent / str(row["source_bank"]) / str(row["snapshot"])
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        if physical_state_sha256(snapshot) != str(row["state_sha256"]):
            raise ValueError("causal capability snapshot physical-state drift")
        phase = str(row["phase"])
        coords = physical_coordinates_from_arrays(snapshot.qpos, snapshot.qvel, bundle=bundle)
        if phase == "downstream":
            if float(coords["root_vz_mps"]) >= 0.0:
                raise ValueError("causal downstream evidence is not descending")
            if bool(np.asarray(snapshot.down_events["valid_contact_seen"])):
                raise ValueError("causal downstream evidence is post-contact")
        bins = quantize_coordinates(coords, ROOT_GEOMETRY_FIELDS)
        result.append(
            {
                "evidence_kind": "ground_connected_causal_probe",
                "role": role,
                "label": int(label["label"]),
                "continuation_success": bool(label["continuation_success"]),
                "phase": phase,
                "candidate_id": str(row["candidate_id"]),
                "state_sha256": str(row["state_sha256"]),
                "root_geometry_cell_id": _root_cell_id(phase, bins),
                "root_geometry_bins": bins,
                "coordinates": coords,
                "x_bin": int(bins["root_x_m"]),
                "x_center_m": int(bins["root_x_m"])
                * float(RESOLUTIONS["root_x_m"]["resolution"]),
                "ground_reachability_proven": True,
                "continuation_viability_proven": bool(label["label"]),
                "ground_reachability": provenance,
                "source_catalog": str(catalog_path),
                "source_snapshot": str(snapshot_path),
            }
        )
    return result


def _unique_cells(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {str(row["root_geometry_cell_id"]) for row in rows}


def _positive(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if int(row["label"]) == 1]


def _role_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positive = _positive(rows)
    negative = [row for row in rows if int(row["label"]) == 0]
    return {
        "candidate_count": len(rows),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "positive_unique_root_geometry_cell_count": len(_unique_cells(positive)),
        "negative_unique_root_geometry_cell_count": len(_unique_cells(negative)),
        "positive_by_phase": {
            phase: sum(row["phase"] == phase for row in positive)
            for phase in ("upstream", "downstream")
        },
        "positive_root_cells_by_x_bin": dict(
            sorted(
                Counter(str(int(row["x_bin"])) for row in positive).items(),
                key=lambda item: int(item[0]),
            )
        ),
    }


def _load_source_curriculum_cells(source_summary: Path | None) -> set[str] | None:
    if source_summary is None:
        return None
    payload = json.loads(Path(source_summary).read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "completed":
        raise ValueError("source causal capability summary schema/status drift")
    cells_path = Path(str(payload["cells_file"]))
    cells_payload = json.loads(cells_path.read_text(encoding="utf-8"))
    if cells_payload.get("schema") != CELLS_SCHEMA:
        raise ValueError("source causal capability cells schema drift")
    return set(str(cell) for cell in cells_payload["curriculum_capability_root_geometry_cell_ids"])


def build_causal_jump_capability_evidence(
    *,
    nominal_centerline: Path,
    model_config: Path,
    train_root: Path,
    calibration_root: Path,
    acceptance_root: Path,
    output_dir: Path,
    source_causal_summary: Path | None = None,
) -> dict[str, Any]:
    """Create role-separated causal Jump-Capability evidence and curriculum cells."""
    centerline = load_nominal_jump_centerline(Path(nominal_centerline))
    resolved = load_config(Path(model_config))
    bundle = load_host_model(resolved)
    centerline_rows = _centerline_cells(centerline, bundle)
    role_rows = {
        "train": _role_evidence(Path(train_root), "train", bundle),
        "calibration": _role_evidence(Path(calibration_root), "calibration", bundle),
        "acceptance": _role_evidence(Path(acceptance_root), "acceptance", bundle),
    }

    centerline_cells = _unique_cells(centerline_rows)
    train_positive = _positive(role_rows["train"])
    train_positive_cells = _unique_cells(train_positive)
    curriculum_cells = centerline_cells | train_positive_cells
    calibration_positive_cells = _unique_cells(_positive(role_rows["calibration"]))
    acceptance_positive_cells = _unique_cells(_positive(role_rows["acceptance"]))
    source_cells = _load_source_curriculum_cells(source_causal_summary)
    comparison_base = source_cells if source_cells is not None else centerline_cells
    new_train_cells = train_positive_cells.difference(comparison_base)

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"causal Jump capability output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    all_rows = [*centerline_rows, *role_rows["train"], *role_rows["calibration"], *role_rows["acceptance"]]
    evidence_path = output / "evidence_rows.json"
    evidence_path.write_text(
        json.dumps(all_rows, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cells_payload = {
        "schema": CELLS_SCHEMA,
        "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
        "resolution_sha256": str(resolution_contract()["resolution_sha256"]),
        "centerline_root_geometry_cell_ids": sorted(centerline_cells),
        "train_positive_root_geometry_cell_ids": sorted(train_positive_cells),
        "curriculum_capability_root_geometry_cell_ids": sorted(curriculum_cells),
        "calibration_positive_holdout_root_geometry_cell_ids": sorted(calibration_positive_cells),
        "acceptance_positive_holdout_root_geometry_cell_ids": sorted(acceptance_positive_cells),
    }
    cells_payload["cells_sha256"] = _canonical_sha256(cells_payload)
    cells_path = output / "cells.json"
    cells_path.write_text(
        json.dumps(cells_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    summary = {
        "schema": SCHEMA,
        "status": "completed",
        "scientific_definition": (
            "Jump capability evidence = natural-ground-forward-reachable state AND successful "
            "continuation from that exact state"
        ),
        "nominal_centerline": str(nominal_centerline),
        "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
        "natural_start_state_sha256": str(centerline["natural_start_state_sha256"]),
        "model_config": str(model_config),
        "model_config_sha256": str(resolved.config_sha256),
        "xml_sha256": str(bundle.xml_sha256),
        "resolution_contract": resolution_contract(),
        "centerline_point_count": len(centerline_rows),
        "centerline_unique_root_geometry_cell_count": len(centerline_cells),
        "roles": {role: _role_summary(rows) for role, rows in role_rows.items()},
        "curriculum_capability": {
            "definition": "locked centerline cells UNION TRAIN-positive ground-connected causal cells",
            "root_geometry_cell_count": len(curriculum_cells),
            "new_train_root_geometry_cell_count_vs_source_or_centerline": len(new_train_cells),
            "source_comparison": str(source_causal_summary) if source_causal_summary is not None else "locked_centerline_baseline",
            "calibration_cells_merged": False,
            "acceptance_cells_merged": False,
        },
        "holdout_evidence": {
            "calibration_positive_root_geometry_cell_count": len(calibration_positive_cells),
            "acceptance_positive_root_geometry_cell_count": len(acceptance_positive_cells),
        },
        "ground_reachability_verified": True,
        "rsi_used_to_establish_reachability": False,
        "continuation_rsi_used_after_forward_reachability": True,
        "raw_soft_tube_is_capability_proof": False,
        "evidence_rows_file": str(evidence_path.resolve()),
        "evidence_rows_file_sha256": _file_sha256(evidence_path),
        "cells_file": str(cells_path.resolve()),
        "cells_file_sha256": _file_sha256(cells_path),
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "empirical_ground_connected_capability_evidence": True,
            "formal_reachability_proof": False,
            "viability_kernel_claim": False,
            "certified_safe_set_claim": False,
            "physical_limit_proof": False,
        },
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    summary_path = output / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary
