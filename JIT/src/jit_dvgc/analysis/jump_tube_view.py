"""Task-semantic Jump-Tube view over a resolution-aware physical Tube.

The raw Soft Tube remains an immutable replay/provenance artifact.  This module
constructs the capability view relevant to one jump: only the longitudinal
corridor around a successful nominal trajectory is counted, and downstream
support must still be descending.  Late recovery is therefore excluded from
Jump-Tube expansion metrics without deleting historical replay states.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt

from .capability_tube import load_projected_capability_entries
from .nominal_jump_centerline import load_nominal_jump_centerline


SCHEMA = "jit_jump_tube_view_v1"


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible(row: Mapping[str, Any], *, x_min: float, x_max: float) -> bool:
    coords = row["coordinates"]
    x = float(coords["root_x_m"])
    if x < x_min - 1.0e-9 or x > x_max + 1.0e-9:
        return False
    phase = str(row["phase"])
    if phase == "downstream" and not float(coords["root_vz_mps"]) < 0.0:
        return False
    return phase in {"upstream", "downstream"}


def _unique_root_cells(entries: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen = set()
    result = []
    for row in entries:
        cell = str(row["root_geometry_cell_id"])
        if cell in seen:
            continue
        seen.add(cell)
        result.append(row)
    return result


def _counts(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = _unique_root_cells(entries)
    by_phase = {}
    for phase in ("upstream", "downstream"):
        phase_rows = [row for row in entries if row["phase"] == phase]
        phase_cells = _unique_root_cells(phase_rows)
        by_phase[phase] = {
            "raw_snapshot_count": len(phase_rows),
            "unique_root_geometry_cell_count": len(phase_cells),
            "x_bin_count": len({int(row["x_bin"]) for row in phase_cells}),
        }
    by_x = Counter(int(row["x_bin"]) for row in cells)
    return {
        "raw_snapshot_count": len(entries),
        "unique_root_geometry_cell_count": len(cells),
        "x_bin_count": len(by_x),
        "by_phase": by_phase,
        "root_cells_by_x_bin": {str(key): int(value) for key, value in sorted(by_x.items())},
    }


def _plot(entries: Sequence[Mapping[str, Any]], output: Path) -> None:
    cells = _unique_root_cells(entries)
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    for phase in ("upstream", "downstream"):
        rows = [row for row in cells if row["phase"] == phase]
        ax.scatter(
            [float(row["coordinates"]["root_x_m"]) for row in rows],
            [float(row["coordinates"]["root_z_m"]) for row in rows],
            s=16,
            alpha=0.7,
            label=phase,
        )
    ax.set_xlabel("root x [m]")
    ax.set_ylabel("root z [m]")
    ax.set_title("Trajectory-centered Jump Tube: x-z occupied cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "jump_tube_x_z_cells.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for phase in ("upstream", "downstream"):
        rows = [row for row in cells if row["phase"] == phase]
        ax.scatter(
            [float(row["coordinates"]["root_x_m"]) for row in rows],
            [float(row["coordinates"]["root_z_m"]) for row in rows],
            [float(row["coordinates"]["root_vx_mps"]) for row in rows],
            s=12,
            alpha=0.65,
            label=phase,
        )
    ax.set_xlabel("root x [m]")
    ax.set_ylabel("root z [m]")
    ax.set_zlabel("root vx [m/s]")
    ax.set_title("Trajectory-centered Jump Tube: x-z-vx occupied cells")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "jump_tube_x_z_vx_cells_3d.png", dpi=180)
    plt.close(fig)


def build_jump_tube_view(
    *,
    capability_geometry_summary: Path,
    nominal_centerline: Path,
    output_dir: Path,
    source_capability_geometry_summary: Path | None = None,
) -> dict[str, Any]:
    geometry_summary, projected = load_projected_capability_entries(Path(capability_geometry_summary))
    centerline = load_nominal_jump_centerline(Path(nominal_centerline))
    x_min = float(centerline["x_min_m"])
    x_max = float(centerline["effective_centerline_max_x_m"])
    kept = [row for row in projected if _eligible(row, x_min=x_min, x_max=x_max)]
    if not kept:
        raise ValueError("Jump-Tube semantic filter removed every projected state")

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Jump-Tube view output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    counts = _counts(kept)
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "completed",
        "capability_geometry_summary": str(capability_geometry_summary),
        "capability_geometry_summary_file_sha256": _file_sha256(capability_geometry_summary),
        "capability_geometry_summary_sha256": str(geometry_summary["summary_sha256"]),
        "nominal_centerline": str(nominal_centerline),
        "nominal_centerline_sha256": str(centerline["centerline_sha256"]),
        "jump_corridor": {
            "x_min_m": x_min,
            "x_max_m": x_max,
            "x_step_m": float(centerline["x_step_m"]),
            "hard_x_max_m": float(centerline["x_hard_max_m"]),
            "terminates_at_first_valid_landing": True,
        },
        "semantic_filter": {
            "upstream": "source phase upstream within nominal jump corridor",
            "downstream": "source phase downstream within nominal jump corridor AND root_vz_mps < 0",
            "post_landing_recovery_included": False,
        },
        **counts,
        "raw_control_tube_entry_count": int(geometry_summary["raw_snapshot_count"]),
        "excluded_raw_snapshot_count": int(geometry_summary["raw_snapshot_count"]) - int(counts["raw_snapshot_count"]),
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }

    if source_capability_geometry_summary is not None:
        source_summary, source_projected = load_projected_capability_entries(
            Path(source_capability_geometry_summary)
        )
        source_kept = [row for row in source_projected if _eligible(row, x_min=x_min, x_max=x_max)]
        source_cells = {str(row["root_geometry_cell_id"]) for row in source_kept}
        target_cells = {str(row["root_geometry_cell_id"]) for row in kept}
        new_cells = target_cells.difference(source_cells)
        new_by_phase = Counter(
            str(row["phase"])
            for row in _unique_root_cells(kept)
            if str(row["root_geometry_cell_id"]) in new_cells
        )
        payload["expansion_vs_source"] = {
            "source_capability_geometry_summary": str(source_capability_geometry_summary),
            "source_capability_geometry_summary_sha256": str(source_summary["summary_sha256"]),
            "source_jump_tube_root_cell_count": len(source_cells),
            "target_jump_tube_root_cell_count": len(target_cells),
            "new_jump_tube_root_geometry_cell_count": len(new_cells),
            "new_cells_by_phase": {phase: int(new_by_phase[phase]) for phase in ("upstream", "downstream")},
        }

    payload["summary_sha256"] = _canonical_sha256(payload)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "entries.json").write_text(
        json.dumps(kept, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    with (output / "projected_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["phase", "x", "z", "vx", "vz", "root_geometry_cell_id"])
        for row in kept:
            c = row["coordinates"]
            writer.writerow([
                row["phase"], c["root_x_m"], c["root_z_m"], c["root_vx_mps"], c["root_vz_mps"], row["root_geometry_cell_id"]
            ])
    _plot(kept, output)
    return payload
