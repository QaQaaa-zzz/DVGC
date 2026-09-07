"""Common-panel empirical continuation support, distinct from a training Tube.

No hull filling or feasibility interpolation. All policies see the same reached
states; a physical cell is occupied if at least one sampled context succeeds.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..jump_evidence_validation import canonical_sha256, write
from .capability_tube import (ROOT_GEOMETRY_FIELDS, FULL_PHYSICAL_FIELDS,
    RESOLUTIONS, quantize_coordinates, resolution_contract, _cell_id)


def project_row(row, coordinates, context_sha, *, x_slice_width_m=0.1):
    root = quantize_coordinates(coordinates, ROOT_GEOMETRY_FIELDS)
    full = quantize_coordinates(coordinates, FULL_PHYSICAL_FIELDS)
    return {"candidate_id": row["candidate_id"], "state_sha256": row["state_sha256"],
            "snapshot_context_sha256": context_sha, "parent_group_id": row["parent_group_id"],
            "phase": row["phase"], "coordinates": coordinates, "x_bin": (root["root_x_m"] if x_slice_width_m == RESOLUTIONS["root_x_m"]["resolution"]
                else int(np.floor(coordinates["root_x_m"] / x_slice_width_m + 0.5))),
            "root_cell": _cell_id(row["phase"], "root_geometry_v1", root),
            "full_cell": _cell_id(row["phase"], "full_physical_v1", full)}


def summarize(projected, labels, *, role, x_slice_width_m=0.1):
    """Pure set arithmetic; a missing policy/candidate never becomes a failure."""
    if role not in {"train", "calibration", "acceptance"} or not projected or not labels:
        raise ValueError("nonempty common panel and declared development role required")
    ids = [r["candidate_id"] for r in projected]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate projected candidate")
    names = list(labels)
    masks = {}
    for name, rows in labels.items():
        if len(rows) != len(projected):
            raise ValueError("incomplete common policy panel")
        for row, point in zip(rows, projected, strict=True):
            if any(row.get(k) != point[k] for k in ("candidate_id", "state_sha256")):
                raise ValueError("common panel order/identity drift")
            if row.get("phase") != point["phase"] or type(row.get("label")) is not int or row["label"] not in (0, 1):
                raise ValueError("invalid common panel phase/outcome")
            if row.get("success_criterion") != "first_valid_landing":
                raise ValueError("mixed or missing endpoint")
        masks[name] = [bool(r["label"]) for r in rows]
    masks["union"] = [any(masks[n][i] for n in names) for i in range(len(projected))]
    sets = {n: {field: {r[field] for r, ok in zip(projected, mask) if ok}
                for field in ("snapshot_context_sha256", "root_cell", "full_cell")}
            for n, mask in masks.items()}
    metrics = []
    for n, mask in masks.items():
        metric = {"policy": n, "candidate_count": len(projected), "successful_candidates": sum(mask),
                  "failed_candidates": len(mask) - sum(mask),
                  "observed_contexts": len(sets[n]["snapshot_context_sha256"]),
                  "root_cells": len(sets[n]["root_cell"]), "full_cells": len(sets[n]["full_cell"])}
        for field, key in (("snapshot_context_sha256", "unique_contexts"), ("root_cell", "unique_root_cells"), ("full_cell", "unique_full_cells")):
            others = set().union(*(sets[other][field] for other in names if other != n))
            metric[key] = len(sets[n][field] - others) if n != "union" else None
        metric["suffix_interactions"] = sum(r["environment_interactions"] for r in labels[n]) if n != "union" else sum(
            sum(r["environment_interactions"] for r in rows) for rows in labels.values())
        metrics.append(metric)
    overlap = {field: [[(len(sets[a][field] & sets[b][field]) / len(sets[a][field] | sets[b][field]))
                        if sets[a][field] | sets[b][field] else None for b in names] for a in names]
               for field in ("root_cell", "full_cell")}
    slices = []
    for phase in ("upstream", "downstream"):
        for x in sorted({r["x_bin"] for r in projected if r["phase"] == phase}):
            indices = [i for i, r in enumerate(projected) if r["phase"] == phase and r["x_bin"] == x]
            for n, mask in masks.items():
                selected = [projected[i] for i in indices if mask[i]]
                entry = {"phase": phase, "x_bin": x, "x_m": x * x_slice_width_m,
                         "policy": n, "reached_candidates": len(indices), "successful_candidates": len(selected),
                         "reached_root_cells": len({projected[i]["root_cell"] for i in indices}),
                         "root_cells": len({r["root_cell"] for r in selected}),
                         "full_cells": len({r["full_cell"] for r in selected})}
                for field in ("root_z_m", "root_vx_mps", "root_vz_mps", "pitch_deg"):
                    values = [r["coordinates"][field] for r in selected]
                    entry[field + "_min"] = min(values) if values else None
                    entry[field + "_max"] = max(values) if values else None
                slices.append(entry)
    # Vary every dimension consistently, including x, and retain phase in IDs.
    sensitivity = []
    for factor in (0.5, 1.0, 2.0):
        cells = {field: [] for field in ("root_cell", "full_cell")}
        for row in projected:
            scaled = {k: v / factor for k, v in row["coordinates"].items()}
            for key, fields in (("root_cell", ROOT_GEOMETRY_FIELDS), ("full_cell", FULL_PHYSICAL_FIELDS)):
                cells[key].append(_cell_id(row["phase"], key, quantize_coordinates(scaled, fields)))
        for n, mask in masks.items():
            sensitivity.append({"resolution_scale": factor, "policy": n,
                **{key + "_count": len({cell for cell, ok in zip(values, mask) if ok}) for key, values in cells.items()}})
    return {"schema": "jit_common_panel_policy_envelopes_v1", "status": "completed", "role": role,
            "policy_names": names, "metrics": metrics, "x_slices": slices, "jaccard": overlap,
            "resolution_sensitivity": sensitivity, "resolution": resolution_contract(),
            "x_slice_width_m": x_slice_width_m, "common_candidate_count": len(projected), "parent_group_count": len({r["parent_group_id"] for r in projected}),
            "masks": masks, "projection_interpolation_used": False, "continuous_volume_claim": False,
            "single_actor_full_prefix_suffix_claim": False, "end_to_end_cost_comparison": False,
            "interpretation": "continuation support on a shared proposer-conditioned reached panel; not each policy's own forward envelope"}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_comparison(projected, report, output, *, centerline=(), title_prefix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = report["policy_names"] + ["union"]
    colors = dict(zip(names, ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#222222"]))
    pairs = [("root_x_m", "root_z_m", "x (m)", "z (m)"),
             ("root_x_m", "root_vx_mps", "x (m)", "vx (m/s)"),
             ("root_x_m", "root_vz_mps", "x (m)", "vz (m/s)"),
             ("roll_deg", "pitch_deg", "Roll (deg)", "Pitch (deg)")]
    masks = report["masks"]
    def save(fig, stem):
        for extension in ("png", "pdf", "svg"):
            fig.savefig(output / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
        plt.close(fig)
    def display(n):
        return "Union" if n == "union" else n.replace("pi_", "π")
    def projection(selected, stem):
        fig, axes = plt.subplots(len(selected), 4, figsize=(13, 2.25 * len(selected)), squeeze=False)
        for row_index, n in enumerate(selected):
            for col, (xf, yf, xl, yl) in enumerate(pairs):
                ax = axes[row_index, col]
                x = np.array([r["coordinates"][xf] for r in projected])
                y = np.array([r["coordinates"][yf] for r in projected])
                ax.scatter(x, y, s=5, c="#d4d4d4", rasterized=True, zorder=1)
                for phase, marker in (("upstream", "o"), ("downstream", "^")):
                    valid = [i for i, r in enumerate(projected) if r["phase"] == phase and masks[n][i]]
                    ax.scatter(x[valid], y[valid], s=9, color=colors[n], marker=marker, alpha=.7, rasterized=True, zorder=2)
                points = [p for p in centerline if xf in p and yf in p]
                if points:
                    ax.plot([p[xf] for p in points], [p[yf] for p in points], "k--", lw=.8, alpha=.65)
                for axis, values in ((ax.set_xlim, x), (ax.set_ylim, y)):
                    pad = max(float(np.ptp(values)) * .06, .01)
                    axis(float(np.min(values)) - pad, float(np.max(values)) + pad)
                ax.set(xlabel=xl, ylabel=yl, title=display(n) if col == 0 else None)
                ax.grid(alpha=.15)
        fig.suptitle(f"{title_prefix}{report['role'].upper()} | Shared reached-state panel: observed first-landing support", fontsize=12)
        handles = [Line2D([], [], marker="o", color="gray", ls="", label="All reached candidates"),
                   Line2D([], [], marker="o", color="black", ls="", label="Upstream success"),
                   Line2D([], [], marker="^", color="black", ls="", label="Downstream success")]
        if centerline:
            handles.append(Line2D([], [], color="black", ls="--", lw=.8, label="Fixed π0 reference"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
        fig.tight_layout(rect=(0, .05, 1, .96))
        save(fig, stem)
    projection(names, "all_policy_projections")
    for name in names:
        projection([name], name + "_projections")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, phase in zip(axes, ("upstream", "downstream")):
        for n in names:
            rows = [r for r in report["x_slices"] if r["phase"] == phase and r["policy"] == n]
            ax.plot([r["x_m"] for r in rows], [r["root_cells"] for r in rows], marker=".", lw=1, color=colors[n], label=display(n))
        ax.set(title=phase, xlabel="x slice (m)", ylabel="Occupied root-state cells")
        ax.grid(alpha=.2)
    axes[-1].legend(frameon=False)
    fig.suptitle(title_prefix + report["role"].upper() + " | Sampled cross-section occupancy")
    fig.tight_layout()
    save(fig, "cross_section_occupancy")
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for row_i, phase in enumerate(("upstream", "downstream")):
        for ax, field, unit in zip(axes[row_i], ("root_z_m", "root_vx_mps", "root_vz_mps", "pitch_deg"), ("z (m)", "vx (m/s)", "vz (m/s)", "Pitch (deg)")):
            for i, n in enumerate(names):
                rows = [r for r in report["x_slices"] if r["phase"] == phase and r["policy"] == n and r[field + "_min"] is not None]
                x = [r["x_m"] + (i - (len(names)-1)/2)*.009 for r in rows]
                ax.vlines(x, [r[field + "_min"] for r in rows], [r[field + "_max"] for r in rows], color=colors[n], lw=1.5, label=display(n))
                ax.scatter(x, [r[field + "_min"] for r in rows], s=4, color=colors[n])
            ax.set(xlabel="x slice (m); policy offsets for visibility", ylabel=unit, title=phase)
            ax.grid(alpha=.15)
    axes[0, -1].legend(frameon=False, fontsize=8)
    fig.suptitle(title_prefix + report["role"].upper() + " | Observed min–max spans; interiors are not certified")
    fig.tight_layout()
    save(fig, "cross_section_spans")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key, unique, label in zip(axes, ("root_cells", "full_cells"), ("unique_root_cells", "unique_full_cells"), ("Root-state cells", "Full physical cells")):
        metrics = report["metrics"]
        ax.bar(range(len(names)), [m[key] for m in metrics], color=[colors[n] for n in names], alpha=.4, label="All occupied")
        ax.bar(range(len(names)-1), [m[unique] for m in metrics[:-1]], color=[colors[n] for n in names[:-1]], label="Lost if this policy is removed")
        ax.set(xticks=range(len(names)), xticklabels=[display(n) for n in names], ylabel=label)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle(title_prefix + report["role"].upper() + " | Coverage and exclusive physical contribution")
    fig.tight_layout()
    save(fig, "policy_contributions")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key in zip(axes, ("root_cell", "full_cell")):
        values = np.array([[np.nan if v is None else v for v in r] for r in report["jaccard"][key]])
        plot = ax.imshow(values, vmin=0, vmax=1, cmap="Blues")
        ticks = range(len(names)-1)
        ax.set(xticks=ticks, yticks=ticks, xticklabels=[display(n) for n in names[:-1]], yticklabels=[display(n) for n in names[:-1]], title=key.replace("_", " "))
        for i in ticks:
            for j in ticks:
                ax.text(j, i, "N/A" if np.isnan(values[i,j]) else f"{values[i,j]:.2f}", ha="center", va="center", color="white" if values[i,j] > .55 else "black")
        fig.colorbar(plot, ax=ax, shrink=.75)
    fig.suptitle(title_prefix + report["role"].upper() + " | Jaccard overlap of occupied cells")
    fig.tight_layout()
    save(fig, "policy_overlap")
    write_csv(output / "policy_metrics.csv", report["metrics"])
    write_csv(output / "x_slices.csv", report["x_slices"])
    write_csv(output / "resolution_sensitivity.csv", report["resolution_sensitivity"])
    rows = [{"candidate_id": r["candidate_id"], "phase": r["phase"], "state_sha256": r["state_sha256"],
             "snapshot_context_sha256": r["snapshot_context_sha256"], "parent_group_id": r["parent_group_id"],
             "root_cell": r["root_cell"], "full_cell": r["full_cell"],
             **({"trajectory_id": r["trajectory_id"], "trajectory_step": r["trajectory_step"]} if "trajectory_id" in r else {}),
             **r["coordinates"],
             **{n: int(masks[n][i]) for n in names}} for i, r in enumerate(projected)]
    write_csv(output / "candidate_outcomes.csv", rows)
    report["report_sha256"] = canonical_sha256({k:v for k,v in report.items() if k != "report_sha256"})
    write(output / "summary.json", report)
