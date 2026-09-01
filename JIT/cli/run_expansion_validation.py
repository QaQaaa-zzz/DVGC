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


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _audit_fresh_anchor_independence_declared(
    protocol: Mapping[str, Any],
    *,
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply exactly the predeclared source-anchor independence contract.

    Fresh anchors must be group/state/observation-disjoint from TRAIN and from
    the already-consumed validation identity catalog. Within the new fresh
    bank, the locked protocol rejects repeated *physical states* only; it does
    not declare near-equal actor observations from distinct fresh parents as an
    exclusion criterion. Such within-bank observation proximity is therefore
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


def _apply_consumed_identity_exclusions(
    catalog: dict[str, Any],
    audit: Mapping[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    """Apply the locked consumed-bank overlap exclusions without replacement."""

    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise ValueError("fresh candidate catalog entries missing")
    if len(entries) != int(catalog.get("candidate_count", -1)):
        raise ValueError("fresh candidate catalog count drift before identity filtering")

    exact = {str(value) for value in audit.get("exact_overlap_candidate_ids", [])}
    near = {str(value) for value in audit.get("near_overlap_candidate_ids", [])}
    if exact & near:
        raise ValueError("fresh identity exclusion classes overlap")
    excluded = exact | near
    candidate_ids = {str(row.get("candidate_id", "")) for row in entries}
    if not excluded.issubset(candidate_ids):
        raise ValueError("fresh identity exclusion references unknown candidate")

    kept = [
        dict(row)
        for row in entries
        if str(row.get("candidate_id", "")) not in excluded
    ]
    exclusion_counts = dict(catalog.get("exclusion_counts", {}))
    exclusion_counts["consumed_validation_exact_state"] = (
        int(exclusion_counts.get("consumed_validation_exact_state", 0)) + len(exact)
    )
    exclusion_counts["consumed_validation_near_duplicate_observation"] = (
        int(exclusion_counts.get("consumed_validation_near_duplicate_observation", 0))
        + len(near)
    )

    filtered = {
        **catalog,
        "prefilter_candidate_count": len(entries),
        "candidate_count": len(kept),
        "exclusion_counts": dict(sorted(exclusion_counts.items())),
        "consumed_validation_identity_filter_applied": True,
        "no_replacement_after_consumed_validation_exclusion": True,
        "entries": kept,
    }
    catalog.clear()
    catalog.update(filtered)
    _write_machine_json(output / "candidate_catalog_independent.json", filtered)

    closed = {
        **dict(audit),
        "status": "independent_after_exclusion" if excluded else "independent",
        "accepted_candidate_count_after_exclusion": len(kept),
        "excluded_exact_state_count": len(exact),
        "excluded_near_duplicate_observation_count": len(near),
        "exact_state_overlap_count": 0,
        "near_duplicate_observation_overlap_count": 0,
        "no_replacement_after_exclusion": True,
    }
    _write_machine_json(output / "consumed_validation_identity_audit.json", closed)
    return closed


def _runtime_protocol_with_engineering_resume(
    original_runtime_protocol: Any,
    config_path: Path,
    config: Mapping[str, Any],
    scientific: Mapping[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    """Preserve a pre-label runtime protocol across a guard-only repair commit.

    A cross-HEAD resume is allowed only when every runtime field other than the
    repository head and its derived protocol SHA is identical, and no fresh
    continuation label/calibration artifact exists yet. The original runtime
    protocol remains authoritative for the already-completed acquisition.
    """

    proposed = original_runtime_protocol(config_path, config, scientific)
    if not resume:
        return proposed

    output = Path(str(config["output_dir"]))
    runtime_path = output / "runtime_protocol.json"
    if not runtime_path.exists():
        return proposed
    existing = _read_json_object(runtime_path)
    if existing == proposed:
        return existing

    def comparable(value: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(value)
        result.pop("repository_head", None)
        result.pop("protocol_sha256", None)
        return result

    if comparable(existing) != comparable(proposed):
        return proposed

    label_artifacts = [
        output / "labels.json",
        output / "label_progress.json",
        output / "label_failure.json",
        output / "upstream" / "calibration.json",
        output / "downstream" / "calibration.json",
        output / "summary.json",
    ]
    existing_label_artifacts = [str(path) for path in label_artifacts if path.exists()]
    if existing_label_artifacts:
        raise ValueError(
            "engineering cross-HEAD resume is forbidden after validation labeling/calibration"
        )

    existing_base = dict(existing)
    existing_sha = str(existing_base.pop("protocol_sha256", ""))
    if fresh_shared.canonical_sha256(existing_base) != existing_sha:
        raise ValueError("existing fresh validation runtime protocol SHA drift")

    repair = {
        "schema": "jit_fresh_validation_engineering_resume_repair_v1",
        "status": "authorized_prelabel_guard_repair",
        "original_repository_head": existing.get("repository_head"),
        "repair_repository_head": proposed.get("repository_head"),
        "original_runtime_protocol_sha256": existing.get("protocol_sha256"),
        "proposed_runtime_protocol_sha256": proposed.get("protocol_sha256"),
        "scientific_protocol_sha256": existing.get("scientific_protocol_sha256"),
        "scientific_protocol_unchanged": True,
        "attempt_schedule_unchanged": True,
        "policy_identity_unchanged": True,
        "candidate_acquisition_reused": (output / "candidate_catalog.json").exists(),
        "validation_labels_existed_before_repair": False,
        "repair_scope": (
            "apply already-predeclared consumed-validation candidate exclusions "
            "before labeling; no seed/panel/model/threshold change"
        ),
    }
    _write_machine_json(output / "engineering_resume_repair.json", repair)
    return existing


def _execute_fresh_with_identity_guard(
    config: Path,
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run fresh validation with the locked independence guards before labels."""

    original_anchor_audit = fresh_shared._audit_fresh_anchor_independence
    original_label_candidates = fresh_shared._label_candidates
    original_runtime_protocol = fresh_shared._runtime_protocol
    audit_result: dict[str, Any] = {}

    def guarded_runtime_protocol(
        config_path: Path,
        config_payload: Mapping[str, Any],
        scientific: Mapping[str, Any],
    ) -> dict[str, Any]:
        return _runtime_protocol_with_engineering_resume(
            original_runtime_protocol,
            config_path,
            config_payload,
            scientific,
            resume=resume,
        )

    def guarded_label_candidates(*args: Any, **kwargs: Any) -> dict[str, Any]:
        output = Path(kwargs["output"])
        catalog = kwargs.get("catalog")
        if not isinstance(catalog, dict):
            raise ValueError("fresh label guard requires mutable candidate catalog")
        raw_audit = audit_fresh_candidate_vs_consumed_validation(config, output)
        closed_audit = _apply_consumed_identity_exclusions(
            catalog, raw_audit, output=output
        )
        audit_result.clear()
        audit_result.update(closed_audit)
        return original_label_candidates(*args, **kwargs)

    fresh_shared._audit_fresh_anchor_independence = (
        _audit_fresh_anchor_independence_declared
    )
    fresh_shared._runtime_protocol = guarded_runtime_protocol
    fresh_shared._label_candidates = guarded_label_candidates
    try:
        report = fresh_shared.execute_fresh_shared_validation(config, resume=resume)
    finally:
        fresh_shared._audit_fresh_anchor_independence = original_anchor_audit
        fresh_shared._runtime_protocol = original_runtime_protocol
        fresh_shared._label_candidates = original_label_candidates

    if not audit_result:
        output = Path(
            json.loads(config.read_text(encoding="utf-8"))["output_dir"]
        )
        audit_path = output / "consumed_validation_identity_audit.json"
        if audit_path.exists():
            stored = _read_json_object(audit_path)
            if stored.get("status") in {"independent", "independent_after_exclusion"}:
                audit_result.update(stored)
        if not audit_result:
            raise ValueError("completed fresh validation is missing closed identity audit")
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
        help="resume only under the exact scientific protocol and sanctioned prelabel repair",
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
                "excluded_exact_state_count": identity_audit.get(
                    "excluded_exact_state_count", 0
                ),
                "excluded_near_duplicate_observation_count": identity_audit.get(
                    "excluded_near_duplicate_observation_count", 0
                ),
                "exact_state_overlap_count": identity_audit[
                    "exact_state_overlap_count"
                ],
                "near_duplicate_observation_overlap_count": identity_audit[
                    "near_duplicate_observation_overlap_count"
                ],
                "accepted_candidate_count_after_exclusion": identity_audit.get(
                    "accepted_candidate_count_after_exclusion"
                ),
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
