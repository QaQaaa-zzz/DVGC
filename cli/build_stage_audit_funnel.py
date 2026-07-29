"""Build the independent-audit input after a prospective branch screen."""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


def survivor_ids(reports: list[dict]) -> set[str]:
    ids: set[str] = set()
    for report in reports:
        for label in report["labels"]:
            if label["label"] == "positive" and int(label["s"]) == int(label["n"]):
                ids.add(str(label["candidate_id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-manifest", required=True)
    parser.add_argument("--screen-report", action="append", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    root = Path(args.output_root)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    manifest = json.loads(Path(args.assignment_manifest).read_text())
    reports = [json.loads(Path(path).read_text()) for path in args.screen_report]
    seed_bases = [report.get("seed_base") for report in reports]
    if None in seed_bases or len(seed_bases) != len(set(seed_bases)):
        raise SystemExit("screen reports require distinct explicit seed namespaces")
    survivors = survivor_ids(reports)
    assignments = {row["candidate_id"]: row for row in manifest["assignments"]}
    grouped: dict[str, list[dict]] = defaultdict(list)
    paths = {row["controller_id"]: row for row in manifest["outputs"]}
    for group in manifest["outputs"]:
        bank = SnapshotBank.load(group["candidate_bank"])
        for row in bank.records:
            if row["id"] not in survivors:
                continue
            controller = assignments[row["id"]]["selected_controller_id"]
            if controller != group["controller_id"]:
                raise SystemExit("candidate/controller assignment mismatch")
            item = copy.deepcopy(row)
            item.update({"artifact_role": "independent_audit_candidate",
                         "safe_claim_allowed": False,
                         "screen_passed": True,
                         "requires_independent_audit": True})
            grouped[controller].append(item)
    root.mkdir(parents=True)
    outputs = []
    for index, controller in enumerate(sorted(grouped)):
        output = root / f"audit_group_{index}.pkl"
        SnapshotBank(grouped[controller], {
            "artifact_role": "independent_audit_candidate_bank",
            "safe_claim_allowed": False,
            "requires_independent_audit": True,
            "controller_id": controller,
            "controller_path": paths[controller]["controller_path"],
            "assignment_manifest_sha256": file_sha256(args.assignment_manifest),
            "screen_report_sha256s": [file_sha256(path) for path in args.screen_report],
        }).save(output)
        outputs.append({"controller_id": controller, "controller_path": paths[controller]["controller_path"],
                        "bank": str(output), "bank_sha256": file_sha256(output),
                        "states": len(grouped[controller])})
    save_json(root / "manifest.json", {
        "status": "PASS", "artifact_role": "stage_independent_audit_funnel",
        "safe_claim_allowed": False, "survivors": len(survivors),
        "screen_rule": "positive and all 8 prospective branches succeed",
        "screen_seed_bases": seed_bases, "outputs": outputs,
    })
    print(json.dumps({"status": "PASS", "survivors": len(survivors), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
