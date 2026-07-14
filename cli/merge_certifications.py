"""Merge globally indexed certification-bank parts into one policy-bound Tube."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.certification_merge import merge_certification_parts


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--parts",nargs="+",required=True); p.add_argument("--output-bank",required=True); a=p.parse_args()
    output=Path(a.output_bank); report_path=output.with_suffix(".cert.json")
    if output.exists() or report_path.exists(): raise SystemExit("Merged certification output already exists")
    paths=[Path(value) for value in a.parts]
    banks=[SnapshotBank.load(path) for path in paths]
    reports=[json.loads(path.with_suffix(".cert.json").read_text(encoding="utf-8")) for path in paths]
    try: merged,report=merge_certification_parts(banks,reports)
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    report["merged_parts"]=[str(path.resolve()) for path in paths]
    merged.save(output); report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"phase":report["phase"],"tube_version":report["tube_version"],"states":report["states"],"terminal_summary":report["terminal_summary"],"summary":report["summary"]},indent=2))


if __name__=="__main__": main()
