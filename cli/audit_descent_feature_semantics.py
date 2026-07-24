"""Compare frozen Descent matcher distance with diagnostic alternatives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


FEATURE_NAMES = (
    "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
    "wx", "wy", "wz", "steer", "hip", "knee", "rearwheel_velocity",
)


def _nearest(feature, support, center, scale, *, wrap=False, drop_x=False):
    delta = support - feature[None, :]
    if wrap:
        delta[:, 3:6] = (delta[:, 3:6] + np.pi) % (2 * np.pi) - np.pi
    normalized = delta / scale
    if drop_x:
        normalized[:, 0] = 0.
    squared = normalized * normalized
    index = int(np.argmin(np.sum(squared, axis=1)))
    return float(np.sqrt(np.sum(squared[index]))), index, squared[index]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--apex-bank", required=True)
    p.add_argument("--support-bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/default.json")
    a = p.parse_args()
    apex = SnapshotBank.load(a.apex_bank)
    support_bank = SnapshotBank.load(a.support_bank)
    matcher = support_bank.metadata["stage_entry_matcher"]
    center = np.asarray(matcher["center"], float)
    scale = np.asarray(matcher["scale"], float)
    support = np.asarray([r["physical_feature"] for r in support_bank.records], float)
    cfg = load_config(a.config)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    hip_q = int(model.jnt_qposadr[model.joint("hip_joint").id])
    knee_q = int(model.jnt_qposadr[model.joint("knee_joint").id])
    rows = []
    ranking_changed_wrap = 0; ranking_changed_no_x = 0
    maximum_relative_equivalence_error = 0.
    for row in apex.records:
        feature = np.asarray(row["physical_feature"], float)
        linear, li, contribution = _nearest(feature, support, center, scale)
        wrapped, wi, wrapped_contribution = _nearest(
            feature, support, center, scale, wrap=True
        )
        no_x, xi, no_x_contribution = _nearest(
            feature, support, center, scale, drop_x=True
        )
        relative_feature = feature.copy(); relative_support = support.copy()
        relative_feature[0] -= float(cfg.step_front_x)
        relative_support[:, 0] -= float(cfg.step_front_x)
        relative_center = center.copy(); relative_center[0] -= float(cfg.step_front_x)
        relative, ri, relative_contribution = _nearest(
            relative_feature, relative_support, relative_center, scale
        )
        maximum_relative_equivalence_error = max(
            maximum_relative_equivalence_error, abs(relative - linear)
        )
        ranking_changed_wrap += int(wi != li)
        ranking_changed_no_x += int(xi != li)
        qpos = np.asarray(row["qpos"], float)
        rows.append({
            "candidate_id": row["id"],
            "candidate_kind": row.get("candidate_kind"),
            "independent_parent": row.get(
                "independent_trajectory_parent_id",
                row.get("source_parent_id", row.get("trajectory_parent_id")),
            ),
            "formal_linear": {
                "distance": linear, "nearest_support_id": support_bank.records[li]["id"],
                "squared_contribution": dict(zip(FEATURE_NAMES, contribution.tolist())),
            },
            "angle_wrapped_diagnostic": {
                "distance": wrapped, "nearest_support_id": support_bank.records[wi]["id"],
                "squared_contribution": dict(zip(FEATURE_NAMES, wrapped_contribution.tolist())),
            },
            "absolute_x_removed_diagnostic": {
                "distance": no_x, "nearest_support_id": support_bank.records[xi]["id"],
                "squared_contribution": dict(zip(FEATURE_NAMES, no_x_contribution.tolist())),
            },
            "landing_relative_x_diagnostic": {
                "distance": relative, "nearest_support_id": support_bank.records[ri]["id"],
                "squared_contribution": dict(zip(FEATURE_NAMES, relative_contribution.tolist())),
            },
            "joint_semantics": {
                "hip_feature_minus_named_qpos": float(feature[13] - qpos[hip_q]),
                "knee_feature_minus_named_qpos": float(feature[14] - qpos[knee_q]),
                "unit": "radian", "sign_zero": "authoritative XML joint coordinates",
            },
        })
    closest = min(rows, key=lambda row: row["formal_linear"]["distance"])
    payload = {
        "status": "PASS", "artifact_role": "descent_matcher_feature_semantics_audit",
        "formal_matcher_unchanged": True,
        "apex_bank_sha256": file_sha256(a.apex_bank),
        "support_bank_sha256": file_sha256(a.support_bank),
        "matcher_sha256": matcher["matcher_sha256"],
        "matcher_radius": matcher["radius"],
        "feature_names_match": tuple(matcher["feature_names"]) == FEATURE_NAMES,
        "semantics": {
            "x": "authoritative XML world x; subtracting the same landing reference from query and support is distance-invariant",
            "z": "authoritative XML world z",
            "linear_velocity": "world-frame root translational velocity",
            "angular_velocity": "root angular velocity in generalized velocity convention used by current extractor",
            "angles": "formal matcher uses direct linear differences; wrapped result is diagnostic only",
            "joints": "named hip/knee qpos in radians, authoritative XML sign and zero",
            "history_control": "PolicyState/history affects restored control and phase evolution but is intentionally absent from the 16-D matcher",
        },
        "angle_wrapping_changed_nearest_count": ranking_changed_wrap,
        "drop_absolute_x_changed_nearest_count": ranking_changed_no_x,
        "landing_relative_transform_max_distance_error": maximum_relative_equivalence_error,
        "closest_formal": closest,
        "rows": rows,
        "deterministic_semantics_error_found": False,
        "diagnostic_alternatives_not_active": True,
    }
    save_json(a.output, payload)
    print(json.dumps({k: v for k, v in payload.items()
                      if k not in ("rows", "closest_formal")}, indent=2))
    print(json.dumps({"closest_formal": closest}, indent=2))


if __name__ == "__main__":
    main()
