"""Build authentic late/early Ascent banks for reverse-curriculum diagnosis."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from cli.prepare_stage_controller_pilots import aligned_reference_anchors
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


def _even(rows, count):
    if len(rows) <= count:
        return list(rows)
    return [rows[int(i)] for i in np.linspace(0, len(rows) - 1, count, dtype=int)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--flight-bank", required=True)
    p.add_argument("--output-bank", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--late-count", type=int, default=3)
    p.add_argument("--early-count", type=int, default=3)
    p.add_argument("--config", default="configs/default.json")
    p.add_argument("--reference", default="data/reference_jump.csv")
    a = p.parse_args()
    cfg = load_config(a.config)
    frame = pd.read_csv(a.reference)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    source = SnapshotBank.load(a.flight_bank)
    aligned, rejected = aligned_reference_anchors(
        [row for row in source.records if row.get("flight_subinterval") == "ascent"],
        frame, model, "ascent",
    )
    ordered = sorted(aligned, key=lambda row: int(row["reference_index"]))
    midpoint = max(1, len(ordered) // 2)
    early_pool, late_pool = ordered[:midpoint], ordered[midpoint:]
    early = _even(early_pool, a.early_count)
    late = _even(late_pool, a.late_count)
    if len(late) < a.late_count or len(early) < a.early_count:
        raise SystemExit("insufficient authentic Ascent anchors")
    rows = []
    for stratum, selected in (("late_ascent", late), ("early_ascent", early)):
        for row in selected:
            item = copy.deepcopy(row)
            item["diagnostic_stratum"] = stratum
            item["trajectory_parent_id"] = f"reference:{row['reference_index']}"
            rows.append(item)
    metadata = copy.deepcopy(source.metadata)
    metadata.update({
        "artifact_role": "ascent_reverse_curriculum_reset_bank",
        "certified_tube": False, "safe_claim_allowed": False,
        "source_bank_sha256": file_sha256(a.flight_bank),
        "reference_sha256": file_sha256(a.reference),
        "joint_state_contract": "exact same-index reference qpos/qvel/root pose/velocity",
    })
    SnapshotBank(rows, metadata).save(a.output_bank)
    save_json(a.output_report, {
        "status": "PASS", "artifact_role": "ascent_reverse_curriculum_reset_audit",
        "bank": str(Path(a.output_bank).resolve()), "bank_sha256": file_sha256(a.output_bank),
        "late_ascent": {"states": len(late), "indices": [r["reference_index"] for r in late]},
        "early_ascent": {"states": len(early), "indices": [r["reference_index"] for r in early]},
        "alignment_rejections": rejected, "t0_apex_entry": 0,
        "reference_after_joint_limit_violation_used": False,
    })
    print(json.dumps({"late": len(late), "early": len(early)}, indent=2))


if __name__ == "__main__":
    main()
