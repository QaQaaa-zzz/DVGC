"""Offline weak-prior analysis for the retained kinematic guideline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .config import file_sha256
from .constants import EXPECTED_REFERENCE_SHA256


RANGE_FIELDS = (
    "roll_angle",
    "pitch_angle",
    "pos_x",
    "pos_z",
    "vel_x",
    "vel_z",
    "hip_position",
    "knee_position",
)


def _extent(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    values = [float(row[field]) for row in rows]
    return {"min": min(values), "max": max(values)}


def analyze_reference(path: Path) -> dict[str, Any]:
    source = Path(path)
    identity = file_sha256(source)
    if identity != EXPECTED_REFERENCE_SHA256:
        raise ValueError("reference trajectory SHA-256 does not match the retained input")
    with source.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("reference trajectory is empty")
    missing = set(("time", *RANGE_FIELDS)) - set(rows[0])
    if missing:
        raise ValueError(f"reference trajectory is missing fields: {sorted(missing)}")
    return {
        "reference_sha256": identity,
        "row_count": len(rows),
        "time": _extent(rows, "time"),
        "ranges": {field: _extent(rows, field) for field in RANGE_FIELDS},
        "training_runtime_dependency": False,
        "claim": "kinematic_guideline_weak_prior_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = analyze_reference(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
