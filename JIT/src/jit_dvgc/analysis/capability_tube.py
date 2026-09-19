"""Resolution-aware physical capability Tube analysis.

A Soft Tube stores exact replayable snapshots for training.  Snapshot cardinality
is not, by itself, a capability-volume metric: many exact states can live inside
the same physically meaningful neighbourhood.  This module projects Tube
snapshots into an explicitly resolved physical state space, builds sparse
occupied cells, produces longitudinal x-slices, and compares Tube generations in
cell space.

The representation is intentionally separate from the Actor observation.  FIFO
history, last action and observation-validity bits remain controller inputs, but
do not define physical capability-cell identity.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import file_sha256, load_config
from ..handoff_snapshot import load_snapshot
from ..model import ModelBundle, load_host_model
from ..soft_tube import SoftTubeArtifact, load_soft_tube
from ..unified_envelope_snapshot import load_unified_envelope_snapshot


SCHEMA = "jit_capability_tube_geometry_v1"
RESOLUTION_SCHEMA = "jit_capability_state_resolution_v1"
ENTRY_SCHEMA = "jit_capability_tube_projected_entries_v1"
CELLS_SCHEMA = "jit_capability_tube_cells_v1"
SLICES_SCHEMA = "jit_capability_tube_x_slices_v1"

# User-approved first physical resolution contract.
RESOLUTIONS: dict[str, dict[str, Any]] = {
    "root_x_m": {"resolution": 0.10, "unit": "m"},
    "root_y_m": {"resolution": 0.10, "unit": "m"},
    "root_z_m": {"resolution": 0.10, "unit": "m"},
    "root_vx_mps": {"resolution": 0.10, "unit": "m/s"},
    "root_vy_mps": {"resolution": 0.10, "unit": "m/s"},
    "root_vz_mps": {"resolution": 0.10, "unit": "m/s"},
    "roll_deg": {"resolution": 0.50, "unit": "deg"},
    "pitch_deg": {"resolution": 0.50, "unit": "deg"},
    "yaw_deg": {"resolution": 0.50, "unit": "deg"},
    "root_wx_degps": {"resolution": 2.0, "unit": "deg/s"},
    "root_wy_degps": {"resolution": 2.0, "unit": "deg/s"},
    "root_wz_degps": {"resolution": 2.0, "unit": "deg/s"},
    "steering_deg": {"resolution": 0.50, "unit": "deg"},
    "hip_deg": {"resolution": 0.50, "unit": "deg"},
    "knee_deg": {"resolution": 0.50, "unit": "deg"},
    "steering_rate_degps": {"resolution": 2.0, "unit": "deg/s"},
    "hip_rate_degps": {"resolution": 2.0, "unit": "deg/s"},
    "knee_rate_degps": {"resolution": 2.0, "unit": "deg/s"},
    "rearwheel_tangent_mps": {"resolution": 0.10, "unit": "m/s"},
    "frontwheel_tangent_mps": {"resolution": 0.10, "unit": "m/s"},
}

ROOT_GEOMETRY_FIELDS: tuple[str, ...] = (
    "root_x_m",
    "root_y_m",
    "root_z_m",
    "root_vx_mps",
    "root_vy_mps",
    "root_vz_mps",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "root_wx_degps",
    "root_wy_degps",
    "root_wz_degps",
)

FULL_PHYSICAL_FIELDS: tuple[str, ...] = ROOT_GEOMETRY_FIELDS + (
    "steering_deg",
    "hip_deg",
    "knee_deg",
    "steering_rate_degps",
    "hip_rate_degps",
    "knee_rate_degps",
    "rearwheel_tangent_mps",
    "frontwheel_tangent_mps",
)

CROSS_SECTION_FIELDS: tuple[str, ...] = (
    "root_y_m",
    "root_z_m",
    "root_vx_mps",
    "root_vy_mps",
    "root_vz_mps",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "root_wx_degps",
    "root_wy_degps",
    "root_wz_degps",
)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resolution_contract() -> dict[str, Any]:
    contract = {
        "schema": RESOLUTION_SCHEMA,
        "quantization": "nearest_grid_round_half_away_from_zero",
        "phase_is_discrete": True,
        "actor_observation_defines_capability_cells": False,
        "wheel_angle_phase_in_cell_identity": False,
        "resolutions": RESOLUTIONS,
        "root_geometry_fields": list(ROOT_GEOMETRY_FIELDS),
        "full_physical_fields": list(FULL_PHYSICAL_FIELDS),
        "longitudinal_progress_coordinate": "root_x_m",
        "x_slice_width_m": RESOLUTIONS["root_x_m"]["resolution"],
    }
    contract["resolution_sha256"] = _canonical_sha256(contract)
    return contract


def _round_half_away_from_zero(value: float) -> int:
    magnitude = math.floor(abs(float(value)) + 0.5)
    return int(magnitude if float(value) >= 0.0 else -magnitude)


def quantize_coordinates(
    coordinates: Mapping[str, float], fields: Sequence[str]
) -> dict[str, int]:
    bins: dict[str, int] = {}
    for field in fields:
        if field not in RESOLUTIONS:
            raise ValueError(f"no capability resolution declared for {field}")
        value = float(coordinates[field])
        resolution = float(RESOLUTIONS[field]["resolution"])
        if not math.isfinite(value):
            raise ValueError(f"nonfinite capability coordinate: {field}")
        bins[field] = _round_half_away_from_zero(value / resolution)
    return bins


def _cell_id(phase: str, profile: str, bins: Mapping[str, int]) -> str:
    return _canonical_sha256(
        {"phase": str(phase), "profile": str(profile), "bins": dict(bins)}
    )


def _quaternion_to_euler_deg(quaternion: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid root quaternion")
    w, x, y, z = q / norm
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = min(1.0, max(-1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    scale = 180.0 / math.pi
    return roll * scale, pitch * scale, yaw * scale


def physical_coordinates_from_arrays(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    bundle: ModelBundle,
) -> dict[str, float]:
    qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
    qvel = np.asarray(qvel, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
        raise ValueError("capability projection requires finite qpos/qvel")
    index = bundle.model_index
    q0 = int(index.root_qpos_address)
    v0 = int(index.root_dof_address)
    roll, pitch, yaw = _quaternion_to_euler_deg(qpos[q0 + 3 : q0 + 7])
    rad_to_deg = 180.0 / math.pi
    rear_radius = float(bundle.mj_model.geom_size[index.rearwheel_geom_id, 0])
    front_radius = float(bundle.mj_model.geom_size[index.frontwheel_geom_id, 0])
    result = {
        "root_x_m": float(qpos[q0]),
        "root_y_m": float(qpos[q0 + 1]),
        "root_z_m": float(qpos[q0 + 2]),
        "root_vx_mps": float(qvel[v0]),
        "root_vy_mps": float(qvel[v0 + 1]),
        "root_vz_mps": float(qvel[v0 + 2]),
        "roll_deg": float(roll),
        "pitch_deg": float(pitch),
        "yaw_deg": float(yaw),
        "root_wx_degps": float(qvel[v0 + 3] * rad_to_deg),
        "root_wy_degps": float(qvel[v0 + 4] * rad_to_deg),
        "root_wz_degps": float(qvel[v0 + 5] * rad_to_deg),
        "steering_deg": float(qpos[index.steering_qpos_address] * rad_to_deg),
        "hip_deg": float(qpos[index.hip_qpos_address] * rad_to_deg),
        "knee_deg": float(qpos[index.knee_qpos_address] * rad_to_deg),
        "steering_rate_degps": float(qvel[index.steering_dof_address] * rad_to_deg),
        "hip_rate_degps": float(qvel[index.hip_dof_address] * rad_to_deg),
        "knee_rate_degps": float(qvel[index.knee_dof_address] * rad_to_deg),
        "rearwheel_tangent_mps": float(qvel[index.rearwheel_dof_address] * rear_radius),
        "frontwheel_tangent_mps": float(qvel[index.frontwheel_dof_address] * front_radius),
    }
    return result


def _load_snapshot_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, int | None, str]:
    identity = json.loads((Path(path) / "identity.json").read_text(encoding="utf-8"))
    schema = str(identity.get("schema", ""))
    if schema == "handoff_snapshot_v1":
        snapshot = load_snapshot(path)
        return snapshot.qpos, snapshot.qvel, None, str(snapshot.xml_sha256)
    if schema == "jit_unified_envelope_snapshot_v1":
        snapshot = load_unified_envelope_snapshot(path)
        return snapshot.qpos, snapshot.qvel, int(snapshot.active_phase), str(snapshot.xml_sha256)
    raise ValueError(f"unsupported Tube snapshot schema for capability projection: {schema}")


def _project_artifact(artifact: SoftTubeArtifact, *, bundle: ModelBundle) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for global_index, row in enumerate(artifact.entries):
        phase = str(row.get("phase", ""))
        if phase not in {"upstream", "downstream"}:
            raise ValueError("Tube entry phase drift during capability projection")
        snapshot_path = Path(str(row.get("snapshot", "")))
        if not snapshot_path.is_dir():
            raise FileNotFoundError(f"Tube snapshot missing: {snapshot_path}")
        qpos, qvel, active_phase, xml_sha = _load_snapshot_arrays(snapshot_path)
        if xml_sha != bundle.xml_sha256:
            raise ValueError("Tube snapshot XML differs from capability-analysis model")
        if active_phase is not None:
            expected = 0 if phase == "upstream" else 1
            if int(active_phase) != expected:
                raise ValueError("Tube snapshot active phase differs from Tube entry phase")
        coords = physical_coordinates_from_arrays(qpos, qvel, bundle=bundle)
        root_bins = quantize_coordinates(coords, ROOT_GEOMETRY_FIELDS)
        full_bins = quantize_coordinates(coords, FULL_PHYSICAL_FIELDS)
        x_bin = int(root_bins["root_x_m"])
        projected.append(
            {
                "global_index": int(global_index),
                "phase": phase,
                "state_sha256": str(row["state_sha256"]),
                "parent_group_id": str(row.get("parent_group_id", "")),
                "snapshot": str(snapshot_path),
                "coordinates": coords,
                "root_geometry_cell_id": _cell_id(phase, "root_geometry_v1", root_bins),
                "root_geometry_bins": root_bins,
                "full_physical_cell_id": _cell_id(phase, "full_physical_v1", full_bins),
                "full_physical_bins": full_bins,
                "x_bin": x_bin,
                "x_center_m": x_bin * float(RESOLUTIONS["root_x_m"]["resolution"]),
            }
        )
    return projected


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize empty capability values")
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q25", "median", "q75", "max"),
            np.quantile(array, (0.0, 0.25, 0.5, 0.75, 1.0)),
        )
    }


def _cell_records(entries: Sequence[Mapping[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    id_field = f"{kind}_cell_id"
    bins_field = f"{kind}_bins"
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in entries:
        groups[str(row[id_field])].append(row)
    result = []
    for cell_id, rows in sorted(groups.items()):
        first = rows[0]
        result.append(
            {
                "cell_id": cell_id,
                "phase": str(first["phase"]),
                "bins": dict(first[bins_field]),
                "occupancy_count": len(rows),
                "representative_global_index": int(first["global_index"]),
                "representative_state_sha256": str(first["state_sha256"]),
                "representative_coordinates": dict(first["coordinates"]),
            }
        )
    return result


def _x_slices(
    entries: Sequence[Mapping[str, Any]],
    root_cells: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    raw_by_x = Counter(int(row["x_bin"]) for row in entries)
    cells_by_x: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in root_cells:
        x_bin = int(cell["bins"]["root_x_m"])
        cells_by_x[x_bin].append(cell)
    slices = []
    width = float(RESOLUTIONS["root_x_m"]["resolution"])
    for x_bin in sorted(cells_by_x):
        cells = cells_by_x[x_bin]
        phases = Counter(str(cell["phase"]) for cell in cells)
        statistics = {
            field: _quantiles(
                [float(cell["representative_coordinates"][field]) for cell in cells]
            )
            for field in CROSS_SECTION_FIELDS
        }
        slices.append(
            {
                "x_bin": int(x_bin),
                "x_center_m": float(x_bin * width),
                "x_interval_m": [float((x_bin - 0.5) * width), float((x_bin + 0.5) * width)],
                "raw_snapshot_count": int(raw_by_x[x_bin]),
                "unique_root_geometry_cell_count": len(cells),
                "phase_cell_counts": dict(sorted(phases.items())),
                "cross_section_statistics_basis": "unique_root_geometry_cells",
                "cross_section_quantiles": statistics,
            }
        )
    return slices


def _density_summary(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [int(cell["occupancy_count"]) for cell in cells]
    return {
        "cell_count": len(values),
        "occupancy_count_quantiles": _quantiles(values),
        "singleton_cell_count": sum(value == 1 for value in values),
        "multi_snapshot_cell_count": sum(value > 1 for value in values),
    }


def _nearest_resolution_distance_summary(
    target_cells: Sequence[Mapping[str, Any]],
    source_cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if not target_cells or not source_cells:
        return None
    source_by_phase: dict[str, list[np.ndarray]] = defaultdict(list)
    for cell in source_cells:
        source_by_phase[str(cell["phase"])].append(
            np.asarray([int(cell["bins"][field]) for field in ROOT_GEOMETRY_FIELDS], dtype=np.float64)
        )
    distances: list[float] = []
    linf: list[float] = []
    for cell in target_cells:
        phase = str(cell["phase"])
        source = source_by_phase.get(phase, [])
        if not source:
            continue
        point = np.asarray([int(cell["bins"][field]) for field in ROOT_GEOMETRY_FIELDS], dtype=np.float64)
        matrix = np.stack(source)
        delta = matrix - point[None, :]
        l2_values = np.sqrt(np.sum(delta * delta, axis=1))
        linf_values = np.max(np.abs(delta), axis=1)
        distances.append(float(np.min(l2_values)))
        linf.append(float(np.min(linf_values)))
    if not distances:
        return None
    return {
        "metric_units": "per-dimension declared resolution units",
        "nearest_l2_quantiles": _quantiles(distances),
        "nearest_linf_quantiles": _quantiles(linf),
    }


def _expansion_comparison(
    target_entries: Sequence[Mapping[str, Any]],
    source_entries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target_root = _cell_records(target_entries, kind="root_geometry")
    source_root = _cell_records(source_entries, kind="root_geometry")
    target_full = _cell_records(target_entries, kind="full_physical")
    source_full = _cell_records(source_entries, kind="full_physical")
    source_root_ids = {str(cell["cell_id"]) for cell in source_root}
    source_full_ids = {str(cell["cell_id"]) for cell in source_full}
    new_root = [cell for cell in target_root if str(cell["cell_id"]) not in source_root_ids]
    new_full = [cell for cell in target_full if str(cell["cell_id"]) not in source_full_ids]
    new_root_ids = {str(cell["cell_id"]) for cell in new_root}
    raw_growth = len(target_entries) - len(source_entries)
    raw_entries_in_new_root = sum(
        str(row["root_geometry_cell_id"]) in new_root_ids for row in target_entries
    )
    by_phase = {}
    for phase in ("upstream", "downstream"):
        by_phase[phase] = {
            "new_root_geometry_cell_count": sum(cell["phase"] == phase for cell in new_root),
            "new_full_physical_cell_count": sum(cell["phase"] == phase for cell in new_full),
        }
    return {
        "source_raw_snapshot_count": len(source_entries),
        "target_raw_snapshot_count": len(target_entries),
        "raw_snapshot_growth": raw_growth,
        "source_unique_root_geometry_cell_count": len(source_root),
        "target_unique_root_geometry_cell_count": len(target_root),
        "new_root_geometry_cell_count": len(new_root),
        "source_unique_full_physical_cell_count": len(source_full),
        "target_unique_full_physical_cell_count": len(target_full),
        "new_full_physical_cell_count": len(new_full),
        "new_root_geometry_cells_per_raw_added_snapshot": (
            float(len(new_root) / raw_growth) if raw_growth > 0 else None
        ),
        "raw_added_snapshots_landing_in_new_root_geometry_cells": raw_entries_in_new_root,
        "by_phase": by_phase,
        "new_root_geometry_nearest_source_distance": _nearest_resolution_distance_summary(
            new_root, source_root
        ),
        "claim_boundary": {
            "raw_entry_growth_is_capability_volume_growth": False,
            "new_root_geometry_cells_are_resolution_aware_coverage_growth": True,
            "cell_growth_is_physical_feasibility_volume_proof": False,
        },
    }


def _write_csv(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "global_index",
        "phase",
        "state_sha256",
        "root_geometry_cell_id",
        "full_physical_cell_id",
        *FULL_PHYSICAL_FIELDS,
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in entries:
            values = {
                "global_index": row["global_index"],
                "phase": row["phase"],
                "state_sha256": row["state_sha256"],
                "root_geometry_cell_id": row["root_geometry_cell_id"],
                "full_physical_cell_id": row["full_physical_cell_id"],
            }
            values.update(row["coordinates"])
            writer.writerow(values)


def _write_visualizations(
    output_dir: Path, root_cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # optional presentation dependency
        return {"status": "not_generated", "reason": f"matplotlib unavailable: {exc}"}

    rows = list(root_cells)
    if not rows:
        return {"status": "not_generated", "reason": "no root geometry cells"}

    generated: list[str] = []
    for y_field, filename, ylabel in (
        ("root_z_m", "tube_x_z_cells.png", "root z [m]"),
        ("root_vx_mps", "tube_x_vx_cells.png", "root vx [m/s]"),
        ("pitch_deg", "tube_x_pitch_cells.png", "pitch [deg]"),
    ):
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        for phase in ("upstream", "downstream"):
            selected = [row for row in rows if row["phase"] == phase]
            ax.scatter(
                [row["representative_coordinates"]["root_x_m"] for row in selected],
                [row["representative_coordinates"][y_field] for row in selected],
                s=10,
                alpha=0.55,
                label=phase,
            )
        ax.set_xlabel("root x [m]")
        ax.set_ylabel(ylabel)
        ax.set_title("Resolution-aware capability Tube occupied cells")
        ax.legend()
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for phase in ("upstream", "downstream"):
        selected = [row for row in rows if row["phase"] == phase]
        ax.scatter(
            [row["representative_coordinates"]["root_x_m"] for row in selected],
            [row["representative_coordinates"]["root_z_m"] for row in selected],
            [row["representative_coordinates"]["root_vx_mps"] for row in selected],
            s=8,
            alpha=0.45,
            label=phase,
        )
    ax.set_xlabel("root x [m]")
    ax.set_ylabel("root z [m]")
    ax.set_zlabel("root vx [m/s]")
    ax.set_title("Capability Tube: x-z-vx occupied cells")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "tube_x_z_vx_cells_3d.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    generated.append(str(path))
    return {
        "status": "generated",
        "basis": "unique_root_geometry_cells_not_raw_snapshot_density",
        "files": generated,
    }


def build_capability_tube_geometry(
    *,
    tube: Path,
    model_config: Path,
    output_dir: Path,
    source_tube: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"capability Tube output already exists: {output_dir}")
    resolved = load_config(Path(model_config))
    bundle = load_host_model(resolved)
    artifact = load_soft_tube(Path(tube))
    target_entries = _project_artifact(artifact, bundle=bundle)
    source_entries = None
    source_artifact = None
    if source_tube is not None:
        source_artifact = load_soft_tube(Path(source_tube))
        source_entries = _project_artifact(source_artifact, bundle=bundle)

    root_cells = _cell_records(target_entries, kind="root_geometry")
    full_cells = _cell_records(target_entries, kind="full_physical")
    slices = _x_slices(target_entries, root_cells)
    output_dir.mkdir(parents=True, exist_ok=False)

    entries_payload = {
        "schema": ENTRY_SCHEMA,
        "tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "resolution_sha256": resolution_contract()["resolution_sha256"],
        "entries": target_entries,
    }
    entries_payload["entries_sha256"] = _canonical_sha256(entries_payload)
    entries_path = output_dir / "entries.json"
    entries_path.write_text(json.dumps(entries_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    cells_payload = {
        "schema": CELLS_SCHEMA,
        "tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "resolution_sha256": resolution_contract()["resolution_sha256"],
        "root_geometry_cells": root_cells,
        "full_physical_cells": full_cells,
    }
    cells_payload["cells_sha256"] = _canonical_sha256(cells_payload)
    cells_path = output_dir / "cells.json"
    cells_path.write_text(json.dumps(cells_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")

    slices_payload = {
        "schema": SLICES_SCHEMA,
        "tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "resolution_sha256": resolution_contract()["resolution_sha256"],
        "slices": slices,
    }
    slices_payload["slices_sha256"] = _canonical_sha256(slices_payload)
    slices_path = output_dir / "x_slices.json"
    slices_path.write_text(json.dumps(slices_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    csv_path = output_dir / "projected_points.csv"
    _write_csv(csv_path, target_entries)
    visualization = _write_visualizations(output_dir, root_cells)

    phases = {}
    for phase in ("upstream", "downstream"):
        phase_rows = [row for row in target_entries if row["phase"] == phase]
        phases[phase] = {
            "raw_snapshot_count": len(phase_rows),
            "unique_root_geometry_cell_count": len(
                {row["root_geometry_cell_id"] for row in phase_rows}
            ),
            "unique_full_physical_cell_count": len(
                {row["full_physical_cell_id"] for row in phase_rows}
            ),
            "x_bin_count": len({int(row["x_bin"]) for row in phase_rows}),
        }

    summary = {
        "schema": SCHEMA,
        "status": "completed",
        "tube": str(tube),
        "tube_iteration": int(artifact.manifest.get("iteration", 0)),
        "tube_manifest_sha256": str(artifact.manifest["manifest_sha256"]),
        "model_config": str(model_config),
        "model_config_sha256": str(resolved.config_sha256),
        "xml_sha256": str(bundle.xml_sha256),
        "resolution_contract": resolution_contract(),
        "raw_snapshot_count": len(target_entries),
        "unique_root_geometry_cell_count": len(root_cells),
        "unique_full_physical_cell_count": len(full_cells),
        "root_geometry_duplicate_fraction": 1.0 - (len(root_cells) / len(target_entries)),
        "full_physical_duplicate_fraction": 1.0 - (len(full_cells) / len(target_entries)),
        "root_geometry_density": _density_summary(root_cells),
        "full_physical_density": _density_summary(full_cells),
        "x_slice_count": len(slices),
        "x_min_center_m": min(row["x_center_m"] for row in slices),
        "x_max_center_m": max(row["x_center_m"] for row in slices),
        "by_phase": phases,
        "source_tube": str(source_tube) if source_tube is not None else None,
        "source_tube_manifest_sha256": (
            str(source_artifact.manifest["manifest_sha256"])
            if source_artifact is not None
            else None
        ),
        "expansion_vs_source": (
            _expansion_comparison(target_entries, source_entries)
            if source_entries is not None
            else None
        ),
        "entries_file": str(entries_path),
        "entries_file_sha256": file_sha256(entries_path),
        "cells_file": str(cells_path),
        "cells_file_sha256": file_sha256(cells_path),
        "x_slices_file": str(slices_path),
        "x_slices_file_sha256": file_sha256(slices_path),
        "projected_points_csv": str(csv_path),
        "projected_points_csv_sha256": file_sha256(csv_path),
        "visualization": visualization,
        "claim_boundary": {
            "raw_snapshot_cardinality_is_capability_expansion": False,
            "resolution_aware_cell_occupancy_is_primary_coverage_metric": True,
            "tube_geometry_is_empirical": True,
            "tube_geometry_is_certified_safe_set": False,
            "physical_feasibility_limit_proved": False,
            "actor_observation_space_used_as_physical_metric_space": False,
        },
        "training_transitions": 0,
        "environment_interactions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    summary["summary_sha256"] = _canonical_sha256(summary)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def load_projected_capability_entries(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("schema") != SCHEMA:
        raise ValueError("unsupported capability Tube summary")
    if summary.get("status") != "completed":
        raise ValueError("capability Tube summary is incomplete")
    declared = str(summary.get("summary_sha256", ""))
    base = {key: value for key, value in summary.items() if key != "summary_sha256"}
    if len(declared) != 64 or _canonical_sha256(base) != declared:
        raise ValueError("capability Tube summary self-hash drift")
    entries_path = Path(str(summary["entries_file"]))
    if file_sha256(entries_path) != summary.get("entries_file_sha256"):
        raise ValueError("capability Tube entries file hash drift")
    payload = json.loads(entries_path.read_text(encoding="utf-8"))
    if payload.get("schema") != ENTRY_SCHEMA:
        raise ValueError("capability Tube entries schema drift")
    entries_hash = str(payload.get("entries_sha256", ""))
    payload_base = {key: value for key, value in payload.items() if key != "entries_sha256"}
    if len(entries_hash) != 64 or _canonical_sha256(payload_base) != entries_hash:
        raise ValueError("capability Tube entries self-hash drift")
    return summary, [dict(row) for row in payload["entries"]]
