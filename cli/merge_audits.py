"""Merge globally indexed independent-audit parts into one formal report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.audit import merge_audit_reports


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts",nargs="+",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args(); output=Path(args.output)
    if output.exists(): raise SystemExit(f"Merged audit output already exists: {output}")
    paths=[Path(value) for value in args.parts]
    reports=[json.loads(path.read_text(encoding="utf-8")) for path in paths]
    try: merged=merge_audit_reports(reports)
    except ValueError as exc: raise SystemExit(str(exc)) from exc
    merged["merged_parts"]=[str(path.resolve()) for path in paths]
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(merged,indent=2),encoding="utf-8")
    print(json.dumps({key:value for key,value in merged.items() if key!="rows"},indent=2))


if __name__=="__main__": main()
