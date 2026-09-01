#!/usr/bin/env python3
"""Execute locked Iteration-0 continuation/expansion validation protocols."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import jax
import numpy as np

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


def _audit_fresh_anchor_independence_declared(
    protocol: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply exactly the predeclared source-anchor independence contract.

    Fresh anchors must be group/state/observation-disjoint from TRAIN and from
    the already-consumed validation identity catalog.  Within the new fresh
    bank, the locked protocol rejects repeated *physical states* only; it does
    not declare near-equal actor observations from distinct fresh parents as an
    exclusion criterion.  Such within-bank observation proximity is therefore
    recorded diagnostically, never used for seed replacement or outcome-aware
    selection.
    """

    _up_manifest, up_raw = fresh_shared.load_frozen_upstream_checkpoint_train_evidence(
        Path(str(protocol["upstream_train_evidence"]))
    )
    _down_manifest, down_raw = fresh_shared.load_frozen_iteration_train_evidence(
        Path(str(protocol["downstream_train_evidence"]))
    )
    train_by_phase = {
        "upstream": [dict(row) for row in up_raw if row.get("phase") == "upstream"],
        "downstream": [dict(row) for row in down_raw if row.get("phase") == "downstream"],
    }
    old_rows = fresh_shared._rows_from_catalog(
        Path(str(protocol["consumed_validation_identity_catalog"]))
    )
    old_states = {str(row.get("state_sha256", "")) for row in old_rows}
    old_obs = np.asarray(
        [row["actor_observation"] for row in old_rows], dtype=np.float32
    )
    atol = float(protocol["near_duplicate_audit"]["actor_observation_atol"])

    phase_stats: dict[str, Any] = {}
    for phase in ("upstream", "downstream"):
        anchors = [
            dict(row) for row in source_report["entries"] if row["phase"] == phase
        ]
        train = train_by_phase[phase]
        train_groups = {str(row["parent_group_id"]) for row in train}
        train_states = {str(row["state_sha256"]) for row in train}
        train_obs = np.asarray(
            [row["actor_observation"] for row in train], dtype=np.float32
        )
        seen_obs: list[np.ndarray] = []
        seen_states: set[str] = set()
        near_pair_count = 0

        for row in anchors:
            group = str(row["parent_group_id"])
            state = str(row["state_sha256"])
            obs = np.asarray(row["actor_observation"], dtype=np.float32)

            if group in train_groups:
                raise ValueError("fresh validation source parent overlaps TRAIN group")
            if state in train_states or fresh_shared._near_any(
                obs, train_obs, atol=atol
            ):
                raise ValueError(
                    "fresh validation source anchor overlaps TRAIN state/observation"
                )
            if state in old_states or fresh_shared._near_any(obs, old_obs, atol=atol):
                raise ValueError(
                    "fresh validation source anchor overlaps consumed validation identity"
                )
            if state in seen_states:
                raise ValueError("fresh validation repeats source anchor state")

            if seen_obs:
                references = np.asarray(seen_obs, dtype=np.float32)
                near_pair_count += int(
                    np.sum(np.all(np.abs(references - obs) <= atol, axis=1))
                )

            seen_states.add(state)
            seen_obs.append(obs)

        phase_stats[phase] = {
            "parent_count": len(anchors),
            "train_parent_overlap_count": 0,
            "train_exact_or_near_overlap_count": 0,
            "consumed_validation_exact_or_near_overlap_count": 0,
            "fresh_anchor_exact_duplicate_count": 0,
            "fresh_anchor_near_duplicate_observation_pair_count": near_pair_count,
            "fresh_anchor_near_duplicate_observation_is_exclusion_rule": False,
        }

    return {
        "status": "independent",
        "actor_observation_atol": atol,
        "consumed_validation_outcomes_read": False,
        "fresh_anchor_rule": "reject_exact_duplicate_state_only_within_fresh_bank",
        "phase_stats": phase_stats,
    }


def _execute_fresh_with_identity_guard(
    config: Path,
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run fresh validation with the locked independence guards before labels.

    The scientific executor remains the single production implementation. This
    CLI guard restores the source-anchor audit to the exact predeclared contract
    and wraps the label entry point so the completed candidate catalog must pass
    the consumed-bank identity-only audit before any continuation outcome is
    generated. Old validation labels/outcomes/predictions are never read.
    """

    original_anchor_audit = fresh_shared._audit_fresh_anchor_independence
    original_label_candidates = fresh_shared._label_candidates
    audit_result: dict[str, Any] = {}

    def guarded_label_candidates(*args: Any, **kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output"])
        result = audit_fresh_candidate_vs_consumed_validation(config, output)
        audit_result.clear()
        audit_result.update(result)
        return original_label_candidates(*args, **kwargs)

    fresh_shared._audit_fresh_anchor_independence = (
        _audit_fresh_anchor_independence_declared
    )
    fresh_shared._label_candidates = guarded_label_candidates
    try:
        report = fresh_shared.execute_fresh_shared_validation(config, resume=resume)
    finally:
        fresh_shared._audit_fresh_anchor_independence = original_anchor_audit
        fresh_shared._label_candidates = original_label_candidates

    # A completed report returned from an already-completed resume does not call
    # the label entry point again. Re-audit identity so the CLI still reports a
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
