"""Build immutable reverse curriculum banks for late-Ascent discovery PPO."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ascent-entry-bank", required=True)
    p.add_argument("--dynamic-apex-bank", required=True)
    p.add_argument("--acquisition-report", required=True)
    p.add_argument("--output-root", required=True)
    a = p.parse_args()
    entries = SnapshotBank.load(a.ascent_entry_bank)
    apex = SnapshotBank.load(a.dynamic_apex_bank)
    report = json.loads(Path(a.acquisition_report).read_text())
    success = set(report["successful_parent_ids"])
    near = [
        copy.deepcopy(row) for row in apex.records
        if row.get("trajectory_parent_id") in success
        and row.get("apex_snapshot_stratum") == "pre_event"
    ]
    successful_entries = [
        copy.deepcopy(row) for row in entries.records
        if row.get("trajectory_parent_id") in success
    ]
    if len({row["trajectory_parent_id"] for row in successful_entries}) < 2:
        raise SystemExit("Late-Ascent curriculum requires two successful parents")
    all_entries = [copy.deepcopy(row) for row in entries.records]
    blocks = {
        1: near,
        2: near + successful_entries,
        3: near + successful_entries + all_entries[::2],
        4: near + all_entries,
    }
    root = Path(a.output_root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "PASS", "artifact_role": "late_ascent_reverse_curriculum",
                "blocks": {}, "successful_parent_ids": sorted(success)}
    for block, rows in blocks.items():
        if not rows:
            rows = successful_entries
        for row in rows:
            row["training_only"] = False
            row["bootstrap_eligible"] = True
            # The environment has a frozen reset-source telemetry enum.
            # Late-Ascent is still the current Flight curriculum source; the
            # finer discovery identity remains in artifact_role/block fields.
            row["reset_source"] = "flight_curriculum"
            row["curriculum_block"] = block
        path = root / f"block_{block}_reset_bank.pkl"
        SnapshotBank(rows, {
            "artifact_role": "late_ascent_discovery_reset_bank",
            "certified_tube": False, "safe_claim_allowed": False,
            "curriculum_block": block,
            "ascent_entry_bank_sha256": file_sha256(a.ascent_entry_bank),
            "dynamic_apex_bank_sha256": file_sha256(a.dynamic_apex_bank),
        }).save(path)
        manifest["blocks"][str(block)] = {
            "path": str(path.resolve()), "sha256": file_sha256(path),
            "records": len(rows),
            "independent_parents": len({
                row.get("trajectory_parent_id") for row in rows
            }),
        }
    save_json(root / "report.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
