"""Build the parent-balanced candidate-guided descent reset distribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_local import build_candidate_bootstrap_bank
from dvgc.runtime import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    if Path(args.output_bank).exists() or Path(args.output_report).exists():
        raise SystemExit("Refusing to overwrite descent bootstrap outputs")
    cfg = load_config(args.config)
    training, report = build_candidate_bootstrap_bank(SnapshotBank.load(args.bank), args.bank, cfg)
    training.save(args.output_bank)
    report.update({
        "status": "PASS",
        "output_bank_sha256": file_sha256(args.output_bank),
        "natural_reset_probability": float(cfg.natural_prob_flight),
        "actor_inputs_exclude": ["bootstrap_group", "descent_layer", "reset_parent_id", "reset_weight"],
    })
    save_json(args.output_report, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
