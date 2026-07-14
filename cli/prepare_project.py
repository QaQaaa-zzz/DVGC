"""Inspect the original XML and analyze the reference trajectory."""
from __future__ import annotations

import argparse
from pathlib import Path

from dvgc.model import save_model_report
from dvgc.reference import ReferenceTrajectory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", default="assets/orange_bike_2kg_horizontal.xml")
    parser.add_argument("--reference", default="data/reference_jump.csv")
    parser.add_argument("--docs", default="docs")
    args = parser.parse_args()

    docs = Path(args.docs)
    docs.mkdir(parents=True, exist_ok=True)
    save_model_report(args.xml, docs / "model_report.json")
    ReferenceTrajectory.load(args.reference).save_analysis(docs)
    print("Inspected the original XML without modifying it or creating a runtime copy.")
    print(f"Reports written to: {docs}")


if __name__ == "__main__":
    main()
