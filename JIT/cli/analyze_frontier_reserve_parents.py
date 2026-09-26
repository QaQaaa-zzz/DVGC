#!/usr/bin/env python3
"""Read-only reserve-parent audit for a completed iterative frontier plan.

This diagnostic inspects the newest expansion shell of the source Tube and
reports parent-group-unique anchors that were *not* consumed by the existing
TRAIN/CALIBRATION/ACCEPTANCE plan.  It reads no continuation outcomes and makes
no mutations.  The purpose is to decide whether a post-calibration-failure
method revision can use fresh parent groups for a new calibration bank, or must
stop for a new real-dynamics parent-generation decision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from jit_dvgc.iterative_frontier_protocol import (
    _frontier_pool,
    _load_plan,
    _selected_policy,
    _source_tube,
)


def checkpoint_domain(parent_group_id: str) -> str:
    text = str(parent_group_id)
    return text.split("__", 1)[0]


def ordered_parent_unique(pool):
    ordered = sorted(
        pool,
        key=lambda item: (
            float(item[2]["value_score"]),
            str(item[2]["parent_group_id"]),
            str(item[2]["state_sha256"]),
        ),
    )
    result = []
    seen_groups: set[str] = set()
    seen_states: set[str] = set()
    for global_index, local_index, row in ordered:
        group = str(row["parent_group_id"])
        state = str(row["state_sha256"])
        if group in seen_groups or state in seen_states:
            continue
        seen_groups.add(group)
        seen_states.add(state)
        result.append(
            {
                "global_index": int(global_index),
                "entry_index": int(local_index),
                "parent_group_id": group,
                "checkpoint_domain": checkpoint_domain(group),
                "state_sha256": state,
                "value_score": float(row["value_score"]),
                "sampling_weight": float(row["sampling_weight"]),
            }
        )
    return result


def analyze(*, source_plan: Path) -> dict[str, Any]:
    plan = _load_plan(Path(source_plan))
    selected, record, _ = _selected_policy(Path(str(plan["selected_policy"])))
    artifact = _source_tube(selected, record, Path(str(plan["source_tube"])))
    if artifact.manifest["manifest_sha256"] != plan["source_tube_manifest_sha256"]:
        raise ValueError("frontier reserve source Tube identity drift")

    pools = _frontier_pool(artifact)
    anchors = [dict(row) for row in plan.get("anchors", [])]
    used_by_phase: dict[str, set[str]] = {"upstream": set(), "downstream": set()}
    role_counts_by_phase: dict[str, Counter[str]] = {
        "upstream": Counter(),
        "downstream": Counter(),
    }
    used_domains_by_phase: dict[str, Counter[str]] = {
        "upstream": Counter(),
        "downstream": Counter(),
    }
    for row in anchors:
        phase = str(row["phase"])
        group = str(row["parent_group_id"])
        used_by_phase[phase].add(group)
        role_counts_by_phase[phase][str(row["role"])] += 1
        used_domains_by_phase[phase][checkpoint_domain(group)] += 1

    phases: dict[str, Any] = {}
    for phase in ("upstream", "downstream"):
        all_unique = ordered_parent_unique(pools[phase])
        reserve = [row for row in all_unique if row["parent_group_id"] not in used_by_phase[phase]]
        reserve_domains = Counter(row["checkpoint_domain"] for row in reserve)
        all_domains = Counter(row["checkpoint_domain"] for row in all_unique)
        reserve_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in reserve:
            reserve_by_domain[row["checkpoint_domain"]].append(row)
        phases[phase] = {
            "newest_shell_parent_unique_count": len(all_unique),
            "used_parent_group_count": len(used_by_phase[phase]),
            "reserve_parent_group_count": len(reserve),
            "role_parent_group_counts": dict(sorted(role_counts_by_phase[phase].items())),
            "all_parent_groups_by_checkpoint_domain": dict(sorted(all_domains.items())),
            "used_parent_groups_by_checkpoint_domain": dict(
                sorted(used_domains_by_phase[phase].items())
            ),
            "reserve_parent_groups_by_checkpoint_domain": dict(sorted(reserve_domains.items())),
            "reserve_checkpoint_domain_count": len(reserve_domains),
            "reserve_parent_groups": reserve,
            "reserve_parent_groups_grouped": {
                domain: rows for domain, rows in sorted(reserve_by_domain.items())
            },
        }

    upstream_domains = set(phases["upstream"]["reserve_parent_groups_by_checkpoint_domain"])
    result = {
        "schema": "jit_iterative_frontier_reserve_parent_diagnostic_v1",
        "status": "completed_read_only",
        "source_plan": str(source_plan),
        "plan_sha256": str(plan["plan_sha256"]),
        "source_tube": str(plan["source_tube"]),
        "source_tube_manifest_sha256": str(plan["source_tube_manifest_sha256"]),
        "policy_actor_sha256": str(plan["policy_actor_sha256"]),
        "policy_payload_sha256": str(plan["policy_payload_sha256"]),
        "phases": phases,
        "decision_support": {
            "fresh_upstream_calibration_three_parents_available": (
                phases["upstream"]["reserve_parent_group_count"] >= 3
            ),
            "fresh_upstream_calibration_three_checkpoint_domains_available": (
                len(upstream_domains) >= 3
            ),
            "fresh_downstream_calibration_parent_available": (
                phases["downstream"]["reserve_parent_group_count"] >= 1
            ),
            "if_false_requires_new_parent_generation_decision": True,
        },
        "continuation_outcomes_read": False,
        "model_refit_performed": False,
        "environment_interactions": 0,
        "mutations_performed": 0,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(source_plan=args.source_plan), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
