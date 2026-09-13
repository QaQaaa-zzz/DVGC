#!/usr/bin/env python3
"""Read-only capacity audit for a new post-frontier parent-generation protocol.

The generic k>=1 frontier intentionally uses only the newest Tube shell.  When
that shell's predeclared TRAIN/CALIBRATION/ACCEPTANCE parent universe is fully
consumed, this diagnostic inspects the *entire* source Tube for root groups and
states that were not used by the current frontier roles.

These unused Tube entries are NOT declared fresh calibration parents.  They are
only possible roots for a future explicitly predeclared real-dynamics generator
whose descendants would receive new parent identities before any continuation
outcomes are inspected.  No continuation labels, calibration outcomes, TEST, or
final-evaluation data are read here.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from jit_dvgc.iterative_frontier_protocol import (
    _load_plan,
    _phase_indexed_rows,
    _selected_policy,
    _source_tube,
)

PHASES = ("upstream", "downstream")
LAYERS = ("retained_core", "newest_shell")


def checkpoint_domain(parent_group_id: str) -> str:
    text = str(parent_group_id)
    return text.split("__", 1)[0]


def source_family(row: Mapping[str, Any]) -> str:
    for field in ("source_name", "source_kind", "source_role", "role"):
        value = str(row.get(field, ""))
        if value:
            return value
    return "unspecified"


def _representatives(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 24,
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row.get("value_score", 1.0)),
            str(row.get("parent_group_id", "")),
            str(row.get("state_sha256", "")),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_groups: set[str] = set()
    seen_states: set[str] = set()
    for row in ordered:
        group = str(row["parent_group_id"])
        state = str(row["state_sha256"])
        if group in seen_groups or state in seen_states:
            continue
        seen_groups.add(group)
        seen_states.add(state)
        selected.append(
            {
                "global_index": int(row["global_index"]),
                "entry_index": int(row["entry_index"]),
                "layer": str(row["layer"]),
                "parent_group_id": group,
                "checkpoint_domain": checkpoint_domain(group),
                "state_sha256": state,
                "value_score": float(row.get("value_score", 1.0)),
                "source_family": source_family(row),
            }
        )
        if len(selected) >= int(maximum):
            break
    return selected


def summarize_phase(
    rows: Sequence[Mapping[str, Any]],
    *,
    used_groups: set[str],
) -> dict[str, Any]:
    all_rows = [dict(row) for row in rows]
    unused_rows = [
        row for row in all_rows if str(row["parent_group_id"]) not in used_groups
    ]
    all_groups = {str(row["parent_group_id"]) for row in all_rows}
    unused_groups = {str(row["parent_group_id"]) for row in unused_rows}
    unused_states = {str(row["state_sha256"]) for row in unused_rows}

    unused_groups_by_layer: dict[str, set[str]] = {layer: set() for layer in LAYERS}
    unused_states_by_layer: dict[str, set[str]] = {layer: set() for layer in LAYERS}
    domain_groups: dict[str, set[str]] = defaultdict(set)
    source_groups: dict[str, set[str]] = defaultdict(set)
    for row in unused_rows:
        layer = str(row["layer"])
        group = str(row["parent_group_id"])
        state = str(row["state_sha256"])
        unused_groups_by_layer.setdefault(layer, set()).add(group)
        unused_states_by_layer.setdefault(layer, set()).add(state)
        domain_groups[checkpoint_domain(group)].add(group)
        source_groups[source_family(row)].add(group)

    return {
        "entry_count": len(all_rows),
        "parent_group_count": len(all_groups),
        "current_frontier_used_parent_group_count": len(all_groups.intersection(used_groups)),
        "unused_for_current_frontier_entry_count": len(unused_rows),
        "unused_for_current_frontier_parent_group_count": len(unused_groups),
        "unused_for_current_frontier_physical_state_count": len(unused_states),
        "unused_parent_groups_by_layer": {
            layer: len(unused_groups_by_layer.get(layer, set())) for layer in LAYERS
        },
        "unused_physical_states_by_layer": {
            layer: len(unused_states_by_layer.get(layer, set())) for layer in LAYERS
        },
        "unused_parent_groups_by_checkpoint_domain": {
            key: len(value) for key, value in sorted(domain_groups.items())
        },
        "unused_parent_groups_by_source_family": {
            key: len(value) for key, value in sorted(source_groups.items())
        },
        "representative_unused_roots": _representatives(unused_rows),
    }


def analyze(*, source_plan: Path) -> dict[str, Any]:
    plan = _load_plan(Path(source_plan))
    selected, record, _ = _selected_policy(Path(str(plan["selected_policy"])))
    artifact = _source_tube(selected, record, Path(str(plan["source_tube"])))
    if artifact.manifest["manifest_sha256"] != plan["source_tube_manifest_sha256"]:
        raise ValueError("parent-generation root audit source Tube identity drift")

    core_count = int(artifact.manifest.get("core_retained_count", -1))
    if core_count <= 0 or core_count >= len(artifact.entries):
        raise ValueError("parent-generation root audit requires retained core plus newest shell")

    used_by_phase: dict[str, set[str]] = {phase: set() for phase in PHASES}
    for anchor in plan.get("anchors", []):
        phase = str(anchor["phase"])
        if phase not in PHASES:
            raise ValueError("frontier plan anchor phase drift")
        used_by_phase[phase].add(str(anchor["parent_group_id"]))

    rows_by_phase: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    for global_index, local_index, phase, source in _phase_indexed_rows(artifact):
        row = dict(source)
        group = str(row.get("parent_group_id", ""))
        state = str(row.get("state_sha256", ""))
        if not group or not state:
            raise ValueError("Tube entry missing parent/state identity")
        row.update(
            {
                "global_index": int(global_index),
                "entry_index": int(local_index),
                "layer": "retained_core" if global_index < core_count else "newest_shell",
            }
        )
        rows_by_phase[phase].append(row)

    phases = {
        phase: summarize_phase(rows_by_phase[phase], used_groups=used_by_phase[phase])
        for phase in PHASES
    }
    result = {
        "schema": "jit_parent_generation_root_capacity_diagnostic_v1",
        "status": "completed_read_only",
        "source_plan": str(source_plan),
        "plan_sha256": str(plan["plan_sha256"]),
        "source_tube": str(plan["source_tube"]),
        "source_tube_manifest_sha256": str(plan["source_tube_manifest_sha256"]),
        "source_tube_entry_count": len(artifact.entries),
        "source_tube_core_retained_count": core_count,
        "policy_actor_sha256": str(plan["policy_actor_sha256"]),
        "policy_payload_sha256": str(plan["policy_payload_sha256"]),
        "phases": phases,
        "decision_support": {
            "unused_upstream_tube_root_groups_exist": (
                phases["upstream"]["unused_for_current_frontier_parent_group_count"] > 0
            ),
            "unused_downstream_tube_root_groups_exist": (
                phases["downstream"]["unused_for_current_frontier_parent_group_count"] > 0
            ),
            "unused_newest_shell_groups_should_match_reserve_audit_zero": bool(
                phases["upstream"]["unused_parent_groups_by_layer"]["newest_shell"] == 0
                and phases["downstream"]["unused_parent_groups_by_layer"]["newest_shell"] == 0
            ),
            "unused_tube_entries_are_generation_roots_not_fresh_calibration_parents": True,
            "new_real_dynamics_descendants_required_before_new_train_or_calibration_roles": True,
            "if_phase_has_no_unused_tube_roots_use_natural_root_parent_generation": True,
        },
        "continuation_outcomes_read": False,
        "calibration_outcomes_read": False,
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
