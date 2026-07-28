"""Freeze the passed diverse-P1 launch support and its immutable lineage."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from cli.build_backward_tube_proposal_index import state_hash
from cli.freeze_descent_predecessor_assets import verify_frozen_assets
from dvgc.backward_tube import BackwardTubeNode, balanced_p1_launch_gate, validate_parent_lineage
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.runtime import save_json


PRIOR = Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json")
C_L = Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl")


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", required=True)
    args = parser.parse_args(); root = Path(args.run)
    valid, failed = verify_frozen_assets(root)
    if not valid: raise SystemExit(f"frozen asset identity failure: {failed}")
    result = json.loads((root / "source_a_certification_results.json").read_text())
    selection = json.loads((root / "balanced_p1_launch_subset_v1_final.json").read_text())
    if result["balanced_gate"]["status"] != "PASS" or selection["status"] != "PASS":
        raise SystemExit("balanced P1 launch gate has not passed")
    old = [BackwardTubeNode(**row) for row in json.loads(PRIOR.read_text())["nodes"]]
    new = [BackwardTubeNode(**row) for row in result["nodes"]]
    all_nodes = old + new; p1 = [node for node in all_nodes if node.p1]
    by_id = {node.node_id: node for node in p1}
    selected = [by_id[node_id] for node_id in selection["node_ids"]]
    if balanced_p1_launch_gate(selected)["status"] != "PASS": raise SystemExit("frozen subset gate mismatch")
    safe_ids = {row["id"] for row in SnapshotBank.load(C_L).records if row["final"]["label"] == "safe"}
    lineage = validate_parent_lineage(all_nodes, safe_ids)
    if not lineage["valid"]: raise SystemExit(f"parent lineage failure: {lineage}")
    harvested = pickle.loads((root / "trajectory_harvested_snapshots.pkl").read_bytes())
    source_checks = []
    for node in new:
        index = int(node.physical_state["source_index"]); physical = harvested[index]["snapshot_v4"]["physical_state_t"]
        actual = state_hash(physical["qpos"], physical["qvel"], physical["ctrl_previous"], physical["qacc_warmstart"])
        source_checks.append({"node_id": node.node_id, "expected": node.source_state_hash,
                              "actual": actual, "exact": actual == node.source_state_hash})
    if not all(row["exact"] for row in source_checks): raise SystemExit("harvested source identity failure")
    save_json(root / "full_p1_bank_v1.json", {"status": "FROZEN", "artifact_role": "nominal_provisional_tube",
        "formal_tube_or_jel": False, "nodes": [node.to_dict() for node in p1],
        "count": len(p1), "source_identity": source_checks})
    save_json(root / "balanced_p1_launch_subset_v1_frozen.json", {"status": "FROZEN",
        "artifact_role": "rsi_launch_support", "formal_tube_or_jel": False,
        "nodes": [node.to_dict() for node in selected], "gate": balanced_p1_launch_gate(selected)})
    save_json(root / "tube_graph_v1.json", {"status": "FROZEN", "artifact_role": "nominal_provisional_tube_graph",
        "formal_tube_or_jel": False, "edges": [{"node_id": node.node_id, "parent_node_id": node.parent_node_id,
        "parent_tube": node.parent_tube, "layer": node.layer} for node in all_nodes], "lineage": lineage})
    save_json(root / "proposal_controller_lineage_v1.json", {"status": "FROZEN", "rows": [{
        "node_id": node.node_id, "candidate_id": node.candidate_id, "source_state_hash": node.source_state_hash,
        "controller_type": node.controller_type, "controller_artifact_sha256": node.controller_artifact_sha256,
        "parent_node_id": node.parent_node_id, "provenance_hashes": node.provenance_hashes,
    } for node in all_nodes]})
    artifacts = {name: file_sha256(root / name) for name in (
        "full_p1_bank_v1.json", "balanced_p1_launch_subset_v1_frozen.json",
        "tube_graph_v1.json", "proposal_controller_lineage_v1.json")}
    save_json(root / "balanced_p1_launch_freeze_manifest.json", {"status": "PASS",
        "classification": "DESCENT_BALANCED_P1_LAUNCH_GATE_PASS", "artifacts": artifacts,
        "heldout_used": False, "delay": False, "PPO_authorization": True})
    print(json.dumps({"status": "PASS", "full_P1": len(p1), "balanced_P1": len(selected),
                      "lineage": lineage["valid"], "artifacts": artifacts}, indent=2))


if __name__ == "__main__": main()
