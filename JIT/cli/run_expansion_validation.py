#!/usr/bin/env python3
"""Execute locked Iteration-0 continuation/expansion validation protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import jax

from jit_dvgc.expansion_validation_runtime import execute_expansion_validation
from jit_dvgc.expansion_validation_runtime_preflight import (
    audit_expansion_validation_runtime_preflight,
)
import jit_dvgc.fresh_shared_continuation_validation as fresh_shared
from jit_dvgc.fresh_validation_identity_audit import (
    audit_fresh_candidate_vs_consumed_validation,
)


FRESH_SHARED_CONFIG_SCHEMA = fresh_shared.CONFIG_SCHEMA


def _schema(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expansion validation config must be a JSON object")
    return str(payload.get("schema", ""))


def _write_machine_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _execute_fresh_with_identity_guard(
    config: Path,
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run fresh validation with consumed-bank identity audit before labeling.

    The scientific executor remains the single production implementation.  This
    CLI guard temporarily wraps its label entry point so the completed candidate
    catalog must pass the identity-only audit before any continuation outcome is
    generated.  Old validation labels/outcomes/predictions are never read.
    """

    original_label_candidates = fresh_shared._label_candidates
    audit_result: dict[str, Any] = {}

    def guarded_label_candidates(*args: Any, **kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output"])
        result = audit_fresh_candidate_vs_consumed_validation(config, output)
        audit_result.clear()
        audit_result.update(result)
        return original_label_candidates(*args, **kwargs)

    fresh_shared._label_candidates = guarded_label_candidates
    try:
        report = fresh_shared.execute_fresh_shared_validation(config, resume=resume)
    finally:
        fresh_shared._label_candidates = original_label_candidates

    # A completed report returned from an already-completed resume does not call
    # the label entry point again.  Re-audit identity so the CLI still reports a
    # closed independence record without inspecting any old outcomes.
    if not audit_result:
        output = Path(
            json.loads(config.read_text(encoding="utf-8"))["output_dir"]
        )
        audit_result.update(
            audit_fresh_candidate_vs_consumed_validation(config, output)
        )
    return report, audit_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run the zero-interaction artifact/protocol preflight and exit",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume only under the exact already-written runtime protocol",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help=(
            "write the machine-readable preflight/final report directly to this "
            "file; unlike stdout this is never contaminated by Warp/JAX logs"
        ),
    )
    args = parser.parse_args()

    fresh = _schema(args.config) == FRESH_SHARED_CONFIG_SCHEMA
    preflight = (
        fresh_shared.audit_fresh_shared_validation_preflight(args.config)
        if fresh
        else audit_expansion_validation_runtime_preflight(args.config)
    )
    if args.audit_only:
        _write_machine_json(args.json_output, preflight)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        return 0
    expected_status = (
        "fresh_validation_preflight_ready" if fresh else "runtime_preflight_ready"
    )
    if preflight.get("status") != expected_status:
        raise RuntimeError("expansion validation runtime preflight did not close")

    if jax.default_backend() != "gpu":
        raise RuntimeError("expansion validation runtime requires the visible JAX GPU")

    if fresh:
        report, identity_audit = _execute_fresh_with_identity_guard(
            args.config, resume=args.resume
        )
        report = {
            **report,
            "consumed_validation_candidate_identity_audit": {
                "status": identity_audit["status"],
                "exact_state_overlap_count": identity_audit[
                    "exact_state_overlap_count"
                ],
                "near_duplicate_observation_overlap_count": identity_audit[
                    "near_duplicate_observation_overlap_count"
                ],
                "consumed_validation_outcomes_read": identity_audit[
                    "consumed_validation_outcomes_read"
                ],
                "consumed_validation_predictions_read": identity_audit[
                    "consumed_validation_predictions_read"
                ],
            },
        }
    else:
        report = execute_expansion_validation(args.config, resume=args.resume)

    _write_machine_json(args.json_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
