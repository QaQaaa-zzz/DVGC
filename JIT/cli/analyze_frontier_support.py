#!/usr/bin/env python3
"""Summarize completed frontier labels by the acquisition probe that produced them.

This is a read-only post-failure diagnostic.  It does not train, relabel, mutate
role membership, or authorize Tube construction.  Candidate metadata comes from
the completed acquisition catalog and binary outcomes come from the completed
continuation labels; the two artifacts are joined only by exact candidate_id.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _direction_name(perturbation: Mapping[str, Any]) -> str:
    if "action_name" in perturbation:
        return f"{perturbation['action_name']}:{int(perturbation['sign']):+d}"
    names = list(perturbation.get("action_names", ()))
    signs = list(perturbation.get("signs", ()))
    if len(names) != len(signs) or not names:
        raise ValueError("frontier perturbation direction metadata is incomplete")
    return "+".join(f"{name}:{int(sign):+d}" for name, sign in zip(names, signs, strict=True))


def _cell_summary(rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, int]:
    positives = sum(int(label["label"]) for _candidate, label in rows)
    return {
        "candidate_count": len(rows),
        "positive_count": positives,
        "negative_count": len(rows) - positives,
        "parent_group_count": len({str(candidate["parent_group_id"]) for candidate, _label in rows}),
    }


def analyze(role_root: Path) -> dict[str, Any]:
    role_root = Path(role_root)
    catalog_path = role_root / "acquisition" / "catalog.json"
    acquisition_protocol_path = role_root / "acquisition" / "protocol.json"
    labels_path = role_root / "labels" / "labels.json"
    labels_summary_path = role_root / "labels" / "summary.json"

    catalog = _read(catalog_path)
    acquisition_protocol = _read(acquisition_protocol_path)
    labels = _read(labels_path)
    labels_summary = _read(labels_summary_path)
    if not isinstance(catalog, dict) or catalog.get("status") != "completed":
        raise ValueError("completed acquisition catalog required")
    if not isinstance(acquisition_protocol, dict) or acquisition_protocol.get("status") != "predeclared":
        raise ValueError("predeclared acquisition protocol required")
    if not isinstance(labels, list) or not labels:
        raise ValueError("completed labels.json required")
    if not isinstance(labels_summary, dict) or labels_summary.get("status") != "completed":
        raise ValueError("completed label summary required")

    candidates = catalog.get("entries")
    if not isinstance(candidates, list) or len(candidates) != int(catalog.get("candidate_count", -1)):
        raise ValueError("acquisition candidate count drift")
    if len(labels) != int(labels_summary.get("candidate_count", -1)):
        raise ValueError("label count drift")

    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    by_label = {str(row["candidate_id"]): row for row in labels}
    if len(by_candidate) != len(candidates) or len(by_label) != len(labels):
        raise ValueError("candidate_id must be unique in acquisition and labels")
    if set(by_candidate) != set(by_label):
        missing_labels = sorted(set(by_candidate).difference(by_label))
        missing_candidates = sorted(set(by_label).difference(by_candidate))
        raise ValueError(
            "acquisition/label candidate_id mismatch: "
            f"missing_labels={missing_labels[:5]} missing_candidates={missing_candidates[:5]}"
        )

    joined = [(by_candidate[cid], by_label[cid]) for cid in sorted(by_candidate)]
    strengths = tuple(float(x) for x in acquisition_protocol.get("strengths", ()))
    durations = tuple(int(x) for x in acquisition_protocol.get("durations", ()))
    if not strengths or not durations:
        raise ValueError("acquisition protocol strength/duration grid missing")
    max_strength = max(strengths)
    max_duration = max(durations)

    by_phase: dict[str, Any] = {}
    for phase in ("upstream", "downstream"):
        phase_rows = [(candidate, label) for candidate, label in joined if candidate.get("phase") == phase]
        if not phase_rows:
            by_phase[phase] = {"candidate_count": 0}
            continue

        by_strength_duration: dict[tuple[float, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        by_direction: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        by_parent: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        outcome_counts: Counter[str] = Counter()

        for candidate, label in phase_rows:
            perturbation = candidate.get("perturbation")
            if not isinstance(perturbation, dict):
                raise ValueError("candidate perturbation metadata missing")
            strength = float(perturbation["strength"])
            duration = int(perturbation["duration"])
            direction = _direction_name(perturbation)
            by_strength_duration[(strength, duration)].append((candidate, label))
            by_direction[direction].append((candidate, label))
            by_parent[str(candidate["parent_group_id"])].append((candidate, label))
            outcome_counts[str(label.get("outcome_class", "unknown"))] += 1

        max_cell = by_strength_duration.get((max_strength, max_duration), [])
        phase_summary = _cell_summary(phase_rows)
        phase_summary.update(
            {
                "outcome_counts": dict(sorted(outcome_counts.items())),
                "max_strength": max_strength,
                "max_duration": max_duration,
                "max_strength_max_duration": _cell_summary(max_cell) if max_cell else {"candidate_count": 0},
                "all_positive_at_max_strength_max_duration": bool(max_cell) and all(int(label["label"]) == 1 for _candidate, label in max_cell),
                "by_strength_duration": [
                    {
                        "strength": strength,
                        "duration": duration,
                        **_cell_summary(rows),
                    }
                    for (strength, duration), rows in sorted(by_strength_duration.items())
                ],
                "by_direction": {
                    key: _cell_summary(rows) for key, rows in sorted(by_direction.items())
                },
                "by_parent_group": {
                    key: _cell_summary(rows) for key, rows in sorted(by_parent.items())
                },
            }
        )
        if phase_summary["negative_count"] == 0:
            if phase_summary["all_positive_at_max_strength_max_duration"]:
                interpretation = (
                    "no observed continuation failure even in the strongest/longest cell of this predeclared panel; "
                    "this panel does not bracket the policy-conditioned failure boundary"
                )
            else:
                interpretation = (
                    "no observed continuation failure in this phase; inspect directional/cell support before defining a new probe panel"
                )
        elif phase_summary["positive_count"] == 0:
            interpretation = "no observed continuation success; this panel lies outside useful two-class support"
        else:
            interpretation = "both continuation classes observed; inspect per-cell and per-parent coverage"
        phase_summary["interpretation"] = interpretation
        by_phase[phase] = phase_summary

    result = {
        "schema": "jit_frontier_support_diagnostics_v1",
        "status": "completed",
        "artifact_role": "post_failure_read_only_probe_support_diagnostic",
        "role_root": str(role_root),
        "iteration": int(catalog.get("iteration", -1)),
        "policy_actor_sha256": str(catalog.get("policy_actor_sha256", "")),
        "policy_payload_sha256": str(catalog.get("policy_payload_sha256", "")),
        "acquisition_protocol_sha256": str(catalog.get("protocol_sha256", "")),
        "label_protocol_sha256": str(labels_summary.get("protocol_sha256", "")),
        "candidate_count": len(joined),
        "strengths": list(strengths),
        "durations": list(durations),
        "by_phase": by_phase,
        "training_transitions": 0,
        "environment_interactions": 0,
        "new_labels_generated": 0,
        "role_membership_changed": False,
        "tube_construction_authorized": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "diagnostic_only": True,
            "may_inform_a_new_predeclared_protocol_after_this_failed_round": True,
            "may_not_retroactively_change_this_round": True,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.role_root)
    if args.output is not None:
        if args.output.exists():
            existing = _read(args.output)
            if existing != result:
                raise FileExistsError(f"diagnostic output already exists with different content: {args.output}")
        else:
            _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
