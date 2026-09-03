#!/usr/bin/env python3
"""Sanitize a prepared v3b workflow to the strict iteration-workflow schema.

This is an engineering-only repair for v3b workflow files produced before the
strict top-level schema/state-dir bug was caught. It does not alter any
scientific artifact, role assignment, probe panel, seed, label, or outcome.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from jit_dvgc.workflow.iteration_loop import load_workflow_config


ALLOWED_TOP_LEVEL = {
    "schema",
    "workflow_name",
    "state_dir",
    "variables",
    "environment",
    "stages",
}
REMOVED_V3B_METADATA = {
    "calibration_repair_plan",
    "calibration_repair_plan_sha256",
    "source_workflow",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("workflow must be a JSON object")
    return value


def sanitize_workflow(payload: Mapping[str, Any], *, state_dir: Path) -> dict[str, Any]:
    if payload.get("schema") != "jit_iteration_workflow_v1":
        raise ValueError("unsupported workflow schema")
    unknown = set(payload).difference(ALLOWED_TOP_LEVEL | REMOVED_V3B_METADATA)
    if unknown:
        raise ValueError(f"unexpected workflow fields beyond known v3b bug: {sorted(unknown)}")
    missing = ALLOWED_TOP_LEVEL.difference(payload)
    if missing:
        raise ValueError(f"workflow missing required top-level fields: {sorted(missing)}")
    result = {key: payload[key] for key in ALLOWED_TOP_LEVEL}
    result["state_dir"] = str(state_dir)
    result["workflow_name"] = str(payload["workflow_name"])
    return result


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    args = parser.parse_args()

    source = _read(args.input)
    state_dir = args.state_dir
    if state_dir is None:
        state_dir = args.output.with_suffix("").with_name(args.output.stem + "_state")
    sanitized = sanitize_workflow(source, state_dir=state_dir)
    _write_atomic(args.output, sanitized)
    config = load_workflow_config(args.output)
    result = {
        "schema": "jit_v3b_workflow_schema_repair_v1",
        "status": "valid",
        "input": str(args.input),
        "output": str(args.output),
        "workflow_name": config.workflow_name,
        "state_dir": config.state_dir,
        "removed_fields": sorted(set(source).intersection(REMOVED_V3B_METADATA)),
        "scientific_artifacts_modified": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
