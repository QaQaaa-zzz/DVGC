"""Bounded local expansion around ranked Descent P0 boundary anchors."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.pilot_descent_local_entry_v1 import _perturb_record
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _load_record
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json
from dvgc.trajectory_mining import canonical_state_byte_hash


SOURCE = Path("runs/descent_reachability_kernel_v2/ranked_unseen_parent_pilot_v2")
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
DEFAULT_RUN = Path("runs/descent_reachability_kernel_v2/ranked_boundary_neighborhood_v1")
SEED = 3_820_000_000


def local_deltas() -> np.ndarray:
    values = (-0.01, -0.005, 0.0, 0.005, 0.01)
    return np.asarray([(vx, vz) for vx in values for vz in values if vx != 0 or vz != 0], np.float32)


def p0_anchor_ids(certification: dict) -> list[str]:
    return sorted(row["node_id"] for row in certification["rows"] if row["P0"]["pass"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    args = parser.parse_args()
    source, root = Path(args.source), Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    certification = json.loads((source / "certification.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    anchors = p0_anchor_ids(certification)
    if not anchors:
        raise SystemExit("no P0 boundary anchor available")
    selected_by_id = {row["proposal_id"]: row for row in manifest["rows"]}
    if set(anchors) - set(selected_by_id):
        raise SystemExit("P0 anchor missing from frozen source manifest")

    frozen = {"C_L": file_sha256(C_L), "pi_D": file_sha256(PI_D / "params.pkl"), "pi_L": file_sha256(PI_L / "params.pkl")}
    expected = {"C_L": EXPECTED["C_L"], "pi_D": EXPECTED["pi_D"], "pi_L": EXPECTED["pi_L"]}
    if frozen != expected:
        raise SystemExit(f"frozen scientific asset mismatch: {frozen}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")

    deltas = local_deltas()
    root.mkdir(parents=True)
    inputs = {
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "source_certification_sha256": file_sha256(source / "certification.json"),
        "C_L": frozen["C_L"], "pi_D": frozen["pi_D"], "pi_L": frozen["pi_L"],
        "xml": EXPECTED["xml"], "seed": SEED,
    }
    save_json(root / "manifest.json", {
        "status": "FROZEN_BEFORE_OUTCOMES", "inputs": inputs, "anchor_ids": anchors,
        "deltas_vx_vz": deltas.tolist(),
        "selection": "all ranked P0 anchors; fixed 5x5 vx/vz grid excluding zero",
    })
    save_json(root / "cost_estimate.json", {
        "estimated_seconds": 1200 * len(anchors), "anchors": len(anchors),
        "states": len(deltas) * len(anchors), "rollouts_per_state": "2 exact + 4 P1 micro",
        "PPO_steps": 0, "independent_audit": False,
    })

    cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
        "use_bank_resets": False, "expert_chain_termination": False,
        "domain_randomization": False, "obs_noise_enable": False,
    })
    artifact = pickle.loads((EXPERT / "adapter.pkl").read_bytes())
    descent_policy, _, _ = load_bundle(PI_D, verify_files=True)
    landing_policy, _, _ = load_bundle(PI_L, verify_files=True)
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
    adapter = compact_observation_command_adapter(
        jnp.asarray(artifact["prototypes"]), jnp.asarray(artifact["targets"]),
        jnp.asarray(artifact["normalizer_mean"]), jnp.asarray(artifact["normalizer_std"]),
        float(artifact["radius"]), float(artifact["core_radius"]),
    )
    candidates, nodes = [], []
    for anchor_index, identifier in enumerate(anchors):
        source_row = selected_by_id[identifier]
        anchor = _load_record(source_row)
        for delta_index, delta in enumerate(deltas):
            seed = SEED + anchor_index * 1000 + delta_index * 10
            record = _perturb_record(env, anchor, delta, seed)
            node_id = hashlib.sha256(f"ranked-boundary:{identifier}:{delta_index}:{SEED}".encode()).hexdigest()[:32]
            record.update({
                "id": node_id, "origin_anchor_id": identifier, "candidate_kind": "descent_ranked_boundary_local",
                "construction_delta_vx_vz": delta.tolist(), "descent_region": source_row["region"],
                "artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
                "tube_metrics_eligible": False,
            })
            candidates.append(record)
            nodes.append({
                "node_id": node_id, "candidate_id": source_row["candidate_id"],
                "layer": source_row["shell_layer"], "region": source_row["region"],
                "source_state_hash": canonical_state_byte_hash(record), "physical_state": record,
                "parent_node_id": source_row["nearest_downstream_node_id"],
            })
    result = certify_policy(
        env, descent_policy, landing_policy, nodes, SEED + 100_000,
        record_loader=lambda node: node["physical_state"], descent_action_adapter=adapter,
        policy_identity_hash=artifact["policy_identity_hash"],
    )
    save_json(root / "certification.json", result)
    p0 = {row["node_id"] for row in result["rows"] if row["P0"]["pass"]}
    p1 = {row["node_id"] for row in result["rows"] if row["P1"]["pass"]}
    by_id = {row["id"]: row for row in candidates}
    p0_records, p1_records = [], []
    for identifier in p0:
        item = copy.deepcopy(by_id[identifier])
        item.update({"construction_P0": True, "construction_P1": identifier in p1,
                     "policy_identity_hash": artifact["policy_identity_hash"]})
        p0_records.append(item)
        if identifier in p1:
            p1_records.append(copy.deepcopy(item))
    metadata = {"artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
                "inputs": inputs, "policy_identity_hash": artifact["policy_identity_hash"]}
    p0_path = root / "p0_boundary_proposal_support.pkl"
    p1_path = root / "p1_candidate_proposal_support.pkl"
    SnapshotBank(p0_records, metadata | {"construction_gate": "P0"}).save(p0_path)
    SnapshotBank(p1_records, metadata | {"construction_gate": "P1"}).save(p1_path)
    status = "PASS" if len(p1_records) >= 2 else "FAIL"
    report = {
        "status": status, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "anchors": len(anchors), "states": len(candidates), "P0": len(p0), "P1": len(p1),
        "P1_parents": len({row["origin_anchor_id"] for row in p1_records}),
        "p0_bank": str(p0_path), "p0_bank_sha256": file_sha256(p0_path),
        "p1_bank": str(p1_path), "p1_bank_sha256": file_sha256(p1_path),
        "formal_tube_or_matcher": False, "PPO_authorization": False,
        "next": "fresh_independent_audit_before_tube_extension" if status == "PASS" else "ranked_boundary_not_robust",
    }
    save_json(root / "DESCENT_RANKED_BOUNDARY_NEIGHBORHOOD_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
