from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from jit_dvgc.analysis.nominal_jump_centerline import build_nominal_jump_centerline


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_nominal_centerline_uses_real_frames_and_stops_at_landing(tmp_path: Path) -> None:
    n = 41
    x = np.linspace(2.4, 4.4, n)
    qpos = np.zeros((n, 12), dtype=np.float64)
    qvel = np.zeros((n, 11), dtype=np.float64)
    qpos[:, 0] = x
    qpos[:, 2] = 0.3 + 0.45 * np.sin(np.linspace(0.0, np.pi, n))
    qvel[:, 0] = 2.0
    apex_index = 18  # x=3.3
    contact_index = 34  # x=4.1
    qvel[: apex_index + 1, 2] = 0.4
    qvel[apex_index + 1 :, 2] = -0.4

    apex = np.zeros(n, dtype=np.float64)
    apex[apex_index:] = 1.0
    contact = np.zeros(n, dtype=np.float64)
    contact[contact_index:] = 1.0
    npz = tmp_path / "canonical_trace.npz"
    np.savez_compressed(
        npz,
        qpos=qpos,
        qvel=qvel,
        **{
            "metric__event__slash__apex_seen": apex,
            "metric__event__slash__descent_valid_contact_seen": contact,
        },
    )
    metadata = {
        "npz_sha256": _sha(npz),
        "metric_keys": {
            "event/apex_seen": "metric__event__slash__apex_seen",
            "event/descent_valid_contact_seen": "metric__event__slash__descent_valid_contact_seen",
        },
    }
    metadata_path = tmp_path / "canonical_trace.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    report = {
        "schema": "jit_pi_unified_canonical_natural_eval_v1",
        "status": "completed",
        "canonical_rollout": {"full_recovery_success": True},
        "canonical_trace_npz": str(npz),
        "canonical_trace_metadata": str(metadata_path),
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    result = build_nominal_jump_centerline(
        canonical_evaluation_report=report_path,
        output_dir=tmp_path / "centerline",
    )
    assert result["real_frames_only"] is True
    assert result["qpos_qvel_interpolation_used"] is False
    assert result["post_landing_recovery_included"] is False
    assert result["effective_centerline_max_x_m"] == 4.1
    assert result["points"][0]["x_target_m"] == 2.5
    assert result["points"][-1]["x_target_m"] == 4.1
    assert all(point["x_match_error_m"] <= 0.05 + 1.0e-9 for point in result["points"])
    downstream = [point for point in result["points"] if point["phase_semantics"] == "downstream"]
    assert downstream
    assert all(point["root_vz_mps"] < 0.0 for point in downstream)
