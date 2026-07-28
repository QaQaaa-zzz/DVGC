"""Pure contracts for nominal backward-recovery Tube construction.

The structures here deliberately describe provisional nominal support.  They
do not confer certified-Tube or JEL status.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


ARTIFACT_ROLE = "nominal_provisional_tube"
PHASE_ORDER = ("landing", "descent", "apex", "ascent", "takeoff", "approach")


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    return hashlib.sha256(raw).hexdigest()


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


@dataclass(frozen=True)
class BackwardTubeNode:
    node_id: str
    phase: str
    layer: int
    region: str
    candidate_id: str
    source_state_hash: str
    physical_state: Mapping[str, Any]
    actor_observation: Sequence[float]
    parent_node_id: str | None
    parent_tube: str
    controller_type: str
    controller_artifact_sha256: str
    entry_tick: int
    downstream_entry_state: Mapping[str, Any]
    final_recovery: bool
    p0: bool
    p1: bool
    branch_results: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    nearest_neighbor_radius: float = 0.0
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)
    artifact_role: str = ARTIFACT_ROLE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.phase not in PHASE_ORDER or self.region not in {"early", "middle", "late", "root"}:
            raise ValueError("invalid phase or region")
        if self.layer < 0 or self.entry_tick < 0 or self.nearest_neighbor_radius < 0:
            raise ValueError("negative layer/tick/radius")
        if self.artifact_role != ARTIFACT_ROLE:
            raise ValueError("nominal nodes cannot claim a certified artifact role")
        if self.p1 and not self.p0:
            raise ValueError("P1 requires P0")
        if self.phase != "landing" and not self.parent_node_id:
            raise ValueError("non-root nodes require a parent")
        if self.p0 and not self.final_recovery:
            raise ValueError("P0 requires downstream Final-Recovery")
        if not self.source_state_hash or not self.controller_artifact_sha256:
            raise ValueError("missing immutable state/controller identity")


def p0_decision(repeats: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require two exact, legal, downstream-Final repeats."""
    if len(repeats) != 2:
        return {"pass": False, "reasons": ["requires_exactly_two_repeats"]}
    reasons = []
    for index, row in enumerate(repeats):
        if not row.get("active_prefix_exact", False): reasons.append(f"repeat_{index}_not_exact")
        if not row.get("downstream_entry", False): reasons.append(f"repeat_{index}_missed_downstream")
        if not row.get("final_recovery", False): reasons.append(f"repeat_{index}_missed_final")
        if row.get("illegal_contact") or row.get("penetration") or row.get("phase_violation"):
            reasons.append(f"repeat_{index}_illegal_physics")
        if row.get("timeout") or row.get("nonfinite"):
            reasons.append(f"repeat_{index}_unhealthy")
    event_signatures = [canonical_hash({k: r.get(k) for k in (
        "termination_tick", "downstream_entry_tick", "end_code", "final_recovery"
    )}) for r in repeats]
    if len(set(event_signatures)) != 1:
        reasons.append("repeat_event_mismatch")
    return {"pass": not reasons, "reasons": reasons, "repeat_event_sha256": event_signatures}


def p1_decision(p0: Mapping[str, Any], branches: Sequence[Mapping[str, Any]],
                nominal_failure_type: str | None) -> dict[str, Any]:
    """Require 3/4 fixed micro-branches with full downstream Final-Recovery."""
    reasons = []
    if not p0.get("pass", False): reasons.append("p0_not_passed")
    if len(branches) != 4: reasons.append("requires_exactly_four_branches")
    successes = 0
    new_failures = []
    for row in branches:
        success = bool(row.get("downstream_entry") and row.get("final_recovery")
                       and not row.get("illegal_contact") and not row.get("penetration")
                       and not row.get("phase_violation") and not row.get("timeout")
                       and not row.get("nonfinite"))
        successes += int(success)
        failure = row.get("failure_type")
        if failure and failure not in {nominal_failure_type, "horizon", "final_recovery"}:
            new_failures.append(str(failure))
    if successes < 3: reasons.append("fewer_than_three_full_chain_successes")
    if new_failures: reasons.append("new_failure_type")
    return {"pass": not reasons, "successes": successes, "branches": len(branches),
            "new_failure_types": sorted(set(new_failures)), "reasons": reasons}


def validate_parent_lineage(nodes: Sequence[BackwardTubeNode], root_ids: set[str]) -> dict[str, Any]:
    by_id = {node.node_id: node for node in nodes}
    duplicate = len(by_id) != len(nodes)
    failures: dict[str, str] = {}
    for node in nodes:
        seen, current = set(), node
        while current.node_id not in root_ids:
            if current.node_id in seen:
                failures[node.node_id] = "cycle"; break
            seen.add(current.node_id)
            if current.parent_node_id in root_ids:
                break
            if not current.parent_node_id or current.parent_node_id not in by_id:
                failures[node.node_id] = "missing_parent"; break
            parent = by_id[current.parent_node_id]
            if parent.layer >= current.layer:
                failures[node.node_id] = "nondecreasing_parent_layer"; break
            current = parent
    return {"valid": not duplicate and not failures, "duplicate_ids": duplicate,
            "complete": len(nodes) - len(failures), "total": len(nodes), "failures": failures}


def deduplicate_nodes(nodes: Sequence[BackwardTubeNode], feature_by_id: Mapping[str, Sequence[float]],
                      scale: Sequence[float], minimum_distance: float) -> tuple[list[BackwardTubeNode], list[dict]]:
    scale_array = np.maximum(np.asarray(scale, np.float64), 1e-8)
    priority = sorted(nodes, key=lambda n: (-int(n.p1), -int(n.p0), -sum(bool(x.get("final_recovery")) for x in n.branch_results), n.node_id))
    kept, rejected = [], []
    for node in priority:
        feature = np.asarray(feature_by_id[node.node_id], np.float64)
        distances = [float(np.linalg.norm((feature - np.asarray(feature_by_id[x.node_id])) / scale_array)) for x in kept]
        if distances and min(distances) < float(minimum_distance):
            rejected.append({"node_id": node.node_id, "reason": "normalized_duplicate", "distance": min(distances)})
        else:
            kept.append(node)
    return sorted(kept, key=lambda n: n.node_id), rejected


def balanced_rsi_weights(nodes: Sequence[BackwardTubeNode], frontier_multiplier: float = 1.5) -> dict[str, float]:
    eligible = [node for node in nodes if node.p1]
    if not eligible:
        raise ValueError("RSI requires P1 nodes")
    buckets: dict[tuple[str, int, str], list[BackwardTubeNode]] = {}
    for node in eligible:
        buckets.setdefault((node.candidate_id, node.layer, node.region), []).append(node)
    bucket_mass = 1.0 / len(buckets)
    raw = {}
    maximum_layer = max(node.layer for node in eligible)
    for key, group in buckets.items():
        for node in group:
            raw[node.node_id] = bucket_mass / len(group) * (frontier_multiplier if node.layer == maximum_layer else 1.0)
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def tube_gate(nodes: Sequence[BackwardTubeNode], *, freeze: bool = False) -> dict[str, Any]:
    p1 = [node for node in nodes if node.p1 and node.phase != "landing"]
    candidates = Counter(node.candidate_id for node in p1)
    required_nodes, required_layers = (32, 4) if freeze else (16, 3)
    checks = {
        "minimum_nodes": len(p1) >= required_nodes,
        "minimum_candidates": len(candidates) >= 6,
        "minimum_layers": len({node.layer for node in p1}) >= required_layers,
        "region_coverage": {node.region for node in p1} >= {"early", "middle", "late"},
        "candidate_cap": not p1 or max(candidates.values()) / len(p1) <= .35,
        "pointwise_precision": all(node.final_recovery for node in p1),
        "lineage_present": all(node.parent_node_id for node in p1),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "P1_nodes": len(p1), "candidate_counts": dict(candidates),
            "layers": sorted({node.layer for node in p1}), "regions": sorted({node.region for node in p1})}
