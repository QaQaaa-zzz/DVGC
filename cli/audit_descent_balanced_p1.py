"""Freeze and audit the initial balanced Descent P1 launch subset."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cli.runtime_gate import source_fingerprint
from dvgc.backward_tube import BackwardTubeNode, balanced_p1_launch_gate, balanced_p1_launch_subset
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.runtime import save_json


DEFAULT_REPORT = Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json")
DEFAULT_INDEX = Path("runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json")
DEFAULT_RUN = Path("runs/descent_diverse_p1_predecessor_recovery_v1")
P1_CAP = 5


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _records(index: dict) -> dict[str, dict]:
    result = {}
    banks = {}
    for row in index["rows"]:
        path = Path(row["source_artifact"])
        bank = banks.setdefault(path, SnapshotBank.load(path))
        record = bank.records[int(row["source_index"])]
        result[row["physical_state_sha256"]] = {"proposal": row, "record": record}
    return result


def _scale(features: np.ndarray, floors: np.ndarray) -> np.ndarray:
    q25, q75 = np.percentile(features, [25, 75], axis=0)
    return np.maximum((q75 - q25) / 1.349, floors)


def _distance(a, b, scale):
    return float(np.linalg.norm((np.asarray(a, np.float64) - np.asarray(b, np.float64)) / scale))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--proposal-index", default=str(DEFAULT_INDEX))
    args = parser.parse_args()
    root, report_path, index_path = Path(args.run), Path(args.report), Path(args.proposal_index)
    if root.exists():
        raise SystemExit(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    report = json.loads(report_path.read_text())
    index = json.loads(index_path.read_text())
    nodes = [BackwardTubeNode(**row) for row in report["nodes"] if row["p1"]]
    source = _records(index)
    if any(node.source_state_hash not in source for node in nodes):
        raise SystemExit("P1 source state missing from immutable proposal index")
    feature_by_id = {node.node_id: np.asarray(source[node.source_state_hash]["record"]["physical_feature"], np.float64)
                     for node in nodes}
    cfg = load_config("configs/default.json")
    scale = _scale(np.asarray(list(feature_by_id.values())), np.asarray(cfg.descent_entry_scale_floors))
    subset, excluded = balanced_p1_launch_subset(nodes, feature_by_id, scale, per_candidate_cap=P1_CAP)
    dominant = Counter(node.candidate_id for node in nodes).most_common(1)[0][0]
    by_candidate = defaultdict(list)
    audit_rows = []
    for node in nodes:
        proposal = source[node.source_state_hash]["proposal"]
        nearest = min((_distance(feature_by_id[node.node_id], feature_by_id[other.node_id], scale), other.node_id)
                      for other in nodes if other.node_id != node.node_id)
        same = [(_distance(feature_by_id[node.node_id], feature_by_id[other.node_id], scale), other.node_id)
                for other in nodes if other.node_id != node.node_id and other.candidate_id == node.candidate_id]
        remaining = [other for other in nodes if other.node_id != node.node_id]
        coverage_before = ({x.layer for x in nodes}, {x.region for x in nodes}, {x.parent_node_id for x in nodes})
        coverage_after = ({x.layer for x in remaining}, {x.region for x in remaining}, {x.parent_node_id for x in remaining})
        row = {
            "node_id": node.node_id, "candidate_id": node.candidate_id, "backward_layer": node.layer,
            "phase_bin": node.region, "source_tick": proposal["tick"], "state_hash": node.source_state_hash,
            "parent_node": node.parent_node_id, "parent_tube": node.parent_tube,
            "controller_type": node.controller_type, "controller_artifact_sha256": node.controller_artifact_sha256,
            "downstream_entry_tick": node.entry_tick, "final_recovery": node.final_recovery,
            "nearest_neighbor_distance": nearest[0], "nearest_neighbor_id": nearest[1],
            "same_candidate_nearest_distance": None if not same else min(same)[0],
            "same_candidate_nearest_id": None if not same else min(same)[1],
            "same_parent": sum(other.parent_node_id == node.parent_node_id for other in nodes) > 1,
            "same_controller_lineage": sum(other.controller_artifact_sha256 == node.controller_artifact_sha256 for other in nodes) > 1,
            "possible_redundancy": nearest[0] < float(cfg.descent_local_normalized_dedup_distance),
            "deletion_breaks_layer_phase_or_parent_coverage": coverage_after != coverage_before,
            "selected_for_balanced_launch": node in subset,
        }
        by_candidate[node.candidate_id].append(row)
        audit_rows.append(row)
    subset_gate = balanced_p1_launch_gate(subset)
    initial_needed = max(0, 16 - len(subset))
    dominant_count = sum(node.candidate_id == dominant for node in subset)
    cap_needed = max(0, int(np.ceil(dominant_count / .35)) - len(subset))
    needed = max(initial_needed, cap_needed)
    prereg = {
        "protocol": "descent_diverse_p1_predecessor_recovery_v1",
        "selection_frozen_before_new_predecessor_results": True,
        "selection_rule": [
            "retain all candidates with <=5 P1 nodes",
            "for an over-cap candidate select every (layer,region,parent) cluster medoid first",
            "fill remaining slots by normalized-feature farthest-point coverage",
            "break all ties by node_id",
        ],
        "feature_scale": scale.tolist(), "feature_scale_rule": "max(robust IQR/1.349, configured physical scale floor)",
        "per_candidate_node_cap": P1_CAP,
        "source_order": ["successful_trajectory_harvest", "student_predecessor_recapture", "forward_action_branch", "bounded_CEM"],
        "source_a": {"extract_active_prefix_only": True, "default_every_valid_tick": True,
                     "priority_entry_relative_ticks": [-1, -2, -3, -4, -5, -6],
                     "p0_repeats": 2, "p1_branches": 4, "p1_required_successes": 3},
        "source_b_offsets": [-1, -2, -3, -4],
        "source_c_action_range_fractions": [.05, .10], "source_c_modified_ticks": [1, 2, 3],
        "cem_tier1": {"maximum_proposals": 8, "horizon": 8, "samples": 64, "iterations": 5},
        "cem_tier2": {"maximum_proposals": 3, "horizon_options": [12, 16], "samples": 128, "iterations": 6},
        "heldout_used": False, "delay": False, "PPO_authorization": False,
    }
    save_json(root / "preregistration.json", prereg)
    asset_paths = {
        "xml": Path(cfg.xml_path), "landing_policy": Path("runs/decoupled_bootstrap_seed0_20260720/frozen/pi_l_frozen/params.pkl"),
        "canonical_C_L": Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl"),
        "descent_policy": Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy/params.pkl"),
        "descent_normalizer_bundle": Path("runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy/params.pkl"),
        "p0_p1_manifest_and_graph": report_path, "proposal_index": index_path,
        "original_student_trajectory_states_actions": Path("runs/unified_descent_cem_teacher_bootstrap_and_local_ppo_probe_v1/dataset_v3/teacher_dataset.pkl"),
        "v4_snapshot_schema_source": Path("dvgc/snapshot_timing.py"), "runtime_gate": Path("docs/RUNTIME_GATE.json"),
        "preregistration": root / "preregistration.json",
    }
    assets = {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in asset_paths.items()}
    assets["action_mapping"] = {"version": ACTION_MAPPING_VERSION, "sha256": _hash_text(ACTION_MAPPING_VERSION)}
    assets["runtime_source"] = {"sha256": source_fingerprint(Path.cwd())}
    save_json(root / "frozen_asset_manifest.json", {"status": "FROZEN", "assets": assets})
    save_json(root / "existing_p1_distribution_audit.json", {
        "status": "PASS", "P1_nodes": len(nodes), "candidate_counts": dict(sorted(Counter(x.candidate_id for x in nodes).items())),
        "dominant_candidate": dominant, "dominant_nodes": by_candidate[dominant], "rows": audit_rows,
        "feature_scale": scale.tolist(), "pointwise_replay_precision": sum(x.final_recovery for x in nodes) / len(nodes),
    })
    save_json(root / "balanced_p1_launch_subset_v1.json", {
        "status": "PREREGISTERED", "full_P1_nodes": len(nodes), "balanced_subset_nodes": len(subset),
        "node_ids": [node.node_id for node in subset], "excluded": excluded,
        "candidate_counts": dict(sorted(Counter(node.candidate_id for node in subset).items())),
        "layers": sorted({node.layer for node in subset}), "regions": sorted({node.region for node in subset}),
        "maximum_candidate_share": max(Counter(node.candidate_id for node in subset).values()) / len(subset),
        "new_non_dominant_P1_required": needed, "current_gate": subset_gate,
    })
    print(json.dumps({"full_P1": len(nodes), "balanced_subset": len(subset), "excluded": len(excluded),
                      "new_non_dominant_P1_required": needed, "dominant_candidate": dominant,
                      "current_max_share": max(Counter(node.candidate_id for node in subset).values()) / len(subset)}, indent=2))


if __name__ == "__main__":
    main()
