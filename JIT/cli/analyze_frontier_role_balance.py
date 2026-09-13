#!/usr/bin/env python3
"""Read-only support diagnostic for completed iterative frontier roles.

This tool is intentionally post-failure and non-authorizing. It can inspect
legacy single-panel roles and the phase-specific v3 protocol. It joins exact
candidate ids to completed continuation labels, summarizes class support by
phase/panel/parent, and compares role parent-score coverage without changing
role membership, thresholds, fields, Tube support, or policy training.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
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
    return "+".join(
        f"{name}:{int(sign):+d}"
        for name, sign in zip(names, signs, strict=True)
    )


def _cell_summary(rows: list[tuple[Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, int]:
    positives = sum(int(label["label"]) for _candidate, label in rows)
    return {
        "candidate_count": len(rows),
        "positive_count": positives,
        "negative_count": len(rows) - positives,
        "parent_group_count": len(
            {str(candidate["parent_group_id"]) for candidate, _label in rows}
        ),
    }


def _phase_panel(protocol: Mapping[str, Any], phase: str) -> dict[str, Any]:
    phase_protocols = protocol.get("phase_protocols")
    if isinstance(phase_protocols, Mapping):
        declared = phase_protocols.get(phase)
        if not isinstance(declared, Mapping) or not isinstance(declared.get("panel"), Mapping):
            raise ValueError(f"phase-specific acquisition panel missing for {phase}")
        panel = dict(declared["panel"])
    else:
        panel = {
            "action_names": list(protocol.get("selected_action_names", ())),
            "signs": list(protocol.get("selected_signs", ())),
            "strengths": list(protocol.get("strengths", ())),
            "durations": list(protocol.get("durations", ())),
            "active_action_dimensions": int(protocol.get("active_action_dimensions", 1)),
        }
    strengths = [float(value) for value in panel.get("strengths", ())]
    durations = [int(value) for value in panel.get("durations", ())]
    if not strengths or not durations:
        raise ValueError(f"acquisition strength/duration grid missing for {phase}")
    return {
        "action_names": [str(value) for value in panel.get("action_names", ())],
        "signs": [int(value) for value in panel.get("signs", ())],
        "strengths": strengths,
        "durations": durations,
        "active_action_dimensions": int(panel.get("active_action_dimensions", 1)),
    }


def analyze_role(role_root: Path) -> dict[str, Any]:
    role_root = Path(role_root)
    acquisition = role_root / "acquisition"
    labels_dir = role_root / "labels"
    catalog = _read(acquisition / "catalog.json")
    protocol = _read(acquisition / "protocol.json")
    labels = _read(labels_dir / "labels.json")
    label_summary = _read(labels_dir / "summary.json")

    if not isinstance(catalog, dict) or catalog.get("status") != "completed":
        raise ValueError("completed acquisition catalog required")
    if not isinstance(protocol, dict) or protocol.get("status") != "predeclared":
        raise ValueError("predeclared acquisition protocol required")
    if not isinstance(labels, list) or not labels:
        raise ValueError("completed labels.json required")
    if not isinstance(label_summary, dict) or label_summary.get("status") != "completed":
        raise ValueError("completed label summary required")

    candidates = catalog.get("entries")
    if not isinstance(candidates, list) or len(candidates) != int(catalog.get("candidate_count", -1)):
        raise ValueError("acquisition candidate count drift")
    if len(labels) != int(label_summary.get("candidate_count", -1)):
        raise ValueError("label count drift")

    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    by_label = {str(row["candidate_id"]): row for row in labels}
    if len(by_candidate) != len(candidates) or len(by_label) != len(labels):
        raise ValueError("candidate_id must be unique")
    if set(by_candidate) != set(by_label):
        raise ValueError("acquisition/label candidate_id mismatch")

    joined = [(by_candidate[cid], by_label[cid]) for cid in sorted(by_candidate)]
    role = str(protocol.get("logical_role", role_root.name))
    by_phase: dict[str, Any] = {}
    parent_sets: dict[str, list[str]] = {}

    for phase in ("upstream", "downstream"):
        rows = [
            (candidate, label)
            for candidate, label in joined
            if str(candidate.get("phase")) == phase
        ]
        if not rows:
            by_phase[phase] = {"candidate_count": 0}
            parent_sets[phase] = []
            continue

        panel = _phase_panel(protocol, phase)
        max_strength = max(panel["strengths"])
        max_duration = max(panel["durations"])
        by_cell: dict[tuple[float, int], list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        by_direction: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        by_parent: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        outcomes: Counter[str] = Counter()

        for candidate, label in rows:
            perturbation = candidate.get("perturbation")
            if not isinstance(perturbation, Mapping):
                raise ValueError("candidate perturbation metadata missing")
            by_cell[(float(perturbation["strength"]), int(perturbation["duration"]))].append((candidate, label))
            by_direction[_direction_name(perturbation)].append((candidate, label))
            by_parent[str(candidate["parent_group_id"])].append((candidate, label))
            outcomes[str(label.get("outcome_class", "unknown"))] += 1

        parent_support = {}
        anchor_scores: list[float] = []
        for parent, parent_rows in sorted(by_parent.items()):
            scores = {float(candidate["anchor_value_score"]) for candidate, _label in parent_rows}
            globals_ = {int(candidate["anchor_global_index"]) for candidate, _label in parent_rows}
            states = {str(candidate["parent_state_sha256"]) for candidate, _label in parent_rows}
            if len(scores) != 1 or len(globals_) != 1 or len(states) != 1:
                raise ValueError("parent anchor identity drift within role acquisition")
            score = next(iter(scores))
            anchor_scores.append(score)
            parent_support[parent] = {
                **_cell_summary(parent_rows),
                "anchor_value_score": score,
                "anchor_global_index": next(iter(globals_)),
                "parent_state_sha256": next(iter(states)),
            }

        max_cell = by_cell.get((max_strength, max_duration), [])
        summary = _cell_summary(rows)
        attempted = int(dict(catalog.get("phase_attempted_candidate_counts", {})).get(phase, 0))
        accepted = int(dict(catalog.get("phase_candidate_counts", {})).get(phase, len(rows)))
        summary.update(
            {
                "panel": panel,
                "attempted_candidate_count": attempted,
                "acquisition_acceptance_rate": float(accepted / attempted) if attempted else None,
                "acquisition_exclusion_counts": dict(
                    sorted(
                        dict(catalog.get("phase_exclusion_counts", {})).get(phase, {}).items()
                    )
                ),
                "outcome_counts": dict(sorted(outcomes.items())),
                "max_strength_max_duration": _cell_summary(max_cell) if max_cell else {"candidate_count": 0},
                "all_positive_at_max_strength_max_duration": bool(max_cell)
                and all(int(label["label"]) == 1 for _candidate, label in max_cell),
                "by_strength_duration": [
                    {
                        "strength": strength,
                        "duration": duration,
                        **_cell_summary(cell_rows),
                    }
                    for (strength, duration), cell_rows in sorted(by_cell.items())
                ],
                "by_direction": {
                    key: _cell_summary(direction_rows)
                    for key, direction_rows in sorted(by_direction.items())
                },
                "by_parent_group": parent_support,
                "anchor_value_score_summary": {
                    "count": len(anchor_scores),
                    "min": min(anchor_scores),
                    "max": max(anchor_scores),
                    "mean": mean(anchor_scores),
                    "sorted": sorted(anchor_scores),
                },
            }
        )
        by_phase[phase] = summary
        parent_sets[phase] = sorted(by_parent)

    return {
        "role": role,
        "role_root": str(role_root),
        "iteration": int(catalog.get("iteration", -1)),
        "policy_actor_sha256": str(catalog.get("policy_actor_sha256", "")),
        "policy_payload_sha256": str(catalog.get("policy_payload_sha256", "")),
        "acquisition_protocol_sha256": str(catalog.get("protocol_sha256", "")),
        "label_protocol_sha256": str(label_summary.get("protocol_sha256", "")),
        "candidate_count": len(joined),
        "by_phase": by_phase,
        "parent_sets": parent_sets,
    }


def analyze(named_roots: list[tuple[str, Path]]) -> dict[str, Any]:
    roles = {name: analyze_role(root) for name, root in named_roots}
    if len(roles) != len(named_roots):
        raise ValueError("role diagnostic names must be unique")

    identities = {
        (value["iteration"], value["policy_actor_sha256"], value["policy_payload_sha256"])
        for value in roles.values()
    }
    if len(identities) != 1:
        raise ValueError("role diagnostics do not share one frozen policy identity")

    comparisons: dict[str, Any] = {}
    role_names = list(roles)
    for phase in ("upstream", "downstream"):
        overlap = {}
        for i, left in enumerate(role_names):
            for right in role_names[i + 1 :]:
                intersection = sorted(
                    set(roles[left]["parent_sets"][phase]).intersection(
                        roles[right]["parent_sets"][phase]
                    )
                )
                overlap[f"{left}__{right}"] = {
                    "parent_group_overlap_count": len(intersection),
                    "overlap": intersection,
                }
        comparisons[phase] = {
            "role_anchor_value_score_summaries": {
                name: roles[name]["by_phase"][phase].get("anchor_value_score_summary")
                for name in role_names
            },
            "role_class_support": {
                name: {
                    key: roles[name]["by_phase"][phase].get(key)
                    for key in (
                        "candidate_count",
                        "positive_count",
                        "negative_count",
                        "parent_group_count",
                    )
                }
                for name in role_names
            },
            "parent_group_overlap": overlap,
        }

    iteration, actor, payload = next(iter(identities))
    return {
        "schema": "jit_frontier_role_balance_diagnostic_v1",
        "status": "completed",
        "artifact_role": "post_failure_read_only_role_balance_diagnostic",
        "iteration": iteration,
        "policy_actor_sha256": actor,
        "policy_payload_sha256": payload,
        "roles": roles,
        "comparisons": comparisons,
        "training_transitions": 0,
        "environment_interactions": 0,
        "new_labels_generated": 0,
        "role_membership_changed": False,
        "threshold_selected": False,
        "continuation_field_fit_authorized": False,
        "tube_construction_authorized": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
        "claim_boundary": {
            "diagnostic_only": True,
            "may_inform_new_predeclared_calibration_protocol": True,
            "may_not_reassign_or_relabel_exposed_parent_groups": True,
            "certified_safe_set_claim": False,
            "jce_jel_claim": False,
        },
    }


def _parse_named_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--role-root must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name or not path.strip():
        raise argparse.ArgumentTypeError("--role-root must be NAME=PATH")
    return name, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role-root",
        action="append",
        type=_parse_named_root,
        required=True,
        help="repeat NAME=PATH for completed-label frontier roots",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.role_root)
    if args.output is not None:
        if args.output.exists():
            existing = _read(args.output)
            if existing != result:
                raise FileExistsError(
                    f"diagnostic output already exists with different content: {args.output}"
                )
        else:
            _write(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
