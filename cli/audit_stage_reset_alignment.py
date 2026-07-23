"""Audit Ascent/Apex reset alignment before any stage-controller training."""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import pandas as pd

from cli.prepare_stage_controller_pilots import aligned_reference_anchors
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    frame = pd.read_csv(args.reference)
    bank = SnapshotBank.load(args.bank)
    stages = {}
    for stage in ("ascent", "apex"):
        source = [
            row for row in bank.records
            if row.get("flight_subinterval") == stage
        ]
        accepted, rejected = aligned_reference_anchors(
            source, frame, model, stage
        )
        stages[stage] = {
            "source_records": len(source),
            "authentic_reference_anchors": len(accepted),
            "reference_indices": [int(row["reference_index"]) for row in accepted],
            "rejected": rejected,
            "qpos_qvel_root_time_aligned": len(accepted) > 0,
            "phase_at_t0": "flight" if accepted else None,
            "next_stage_at_t0": False if accepted else None,
            "short_reset_shock_probe_steps": 5,
        }
    payload = {
        "status": "PASS" if all(
            row["authentic_reference_anchors"] > 0 for row in stages.values()
        ) else "FAIL",
        "artifact_role": "stage_reset_authenticity_audit",
        "bank": str(Path(args.bank).resolve()),
        "bank_sha256": file_sha256(args.bank),
        "xml_sha256": file_sha256(cfg.xml_path),
        "reference_sha256": file_sha256(args.reference),
        "stages": stages,
        "training_authorized": {
            "ascent": stages["ascent"]["authentic_reference_anchors"] >= 6,
            "apex": stages["apex"]["authentic_reference_anchors"] >= 6,
        },
    }
    save_json(args.output, payload)


if __name__ == "__main__":
    main()
