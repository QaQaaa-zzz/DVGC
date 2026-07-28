"""Certify a frozen, parent-disjoint reachability-ranked Descent pilot."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import subprocess
from pathlib import Path

import jax.numpy as jnp

from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L, _load_record
from cli.run_backward_descent_rsi_pilot import certify_policy
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


RANKING = Path("runs/descent_reachability_kernel_v2/construction_parent_cv_feature_selected/ranked_proposals.json")
MODEL = Path("runs/descent_reachability_kernel_v2/construction_parent_cv_feature_selected/model.npz")
INDEX = Path("runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json")
TUBE = Path("runs/descent_natural_bridge_candidates_v1/independent_audit_round2_round3_2x32/descent_tube_v4.pkl")
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
DEFAULT_RUN = Path("runs/descent_reachability_kernel_v2/ranked_unseen_parent_pilot_v2")


def validate_selection(ranking: dict, index_rows: list[dict], model_sha256: str) -> list[dict]:
    if ranking.get("status") != "PASS":
        raise ValueError("ranking did not pass its construction-only model gate")
    if ranking.get("artifact_role") != "proposal_only_reachability_ranking":
        raise ValueError("invalid ranking artifact role")
    if ranking.get("formal_tube_or_matcher") is not False or ranking.get("independent_audit_labels_used") is not False:
        raise ValueError("ranking crossed the proposal-only evidence boundary")
    if ranking.get("model_sha256") != model_sha256:
        raise ValueError("ranking/model hash mismatch")
    selected = ranking.get("selected", [])
    if len(selected) != 12:
        raise ValueError(f"expected 12 selected proposals, got {len(selected)}")
    if len({row["candidate_id"] for row in selected}) != len(selected):
        raise ValueError("selected proposals are not parent-disjoint")
    if set(row["region"] for row in selected) != {"early", "middle", "late"}:
        raise ValueError("selected proposals do not cover every Descent region")
    by_id = {row["proposal_id"]: row for row in index_rows}
    checked = []
    for row in selected:
        source = by_id.get(row["proposal_id"])
        if source is None:
            raise ValueError(f"proposal missing from immutable index: {row['proposal_id']}")
        for key in ("candidate_id", "region", "physical_state_sha256", "source_artifact", "source_index"):
            if row[key] != source[key]:
                raise ValueError(f"immutable index mismatch for {row['proposal_id']}:{key}")
        # The ranking intentionally stores only the acquisition fields.  The
        # immutable index remains authoritative for graph lineage fields.
        checked.append(source | {"reachability_score": row["reachability_score"]})
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--ranking", default=str(RANKING))
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--index", default=str(INDEX))
    parser.add_argument("--tube", default=str(TUBE))
    args = parser.parse_args()
    root = Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    ranking_path, model_path, index_path, tube_path = map(Path, (args.ranking, args.model, args.index, args.tube))
    ranking = json.loads(ranking_path.read_text())
    index_rows = json.loads(index_path.read_text())["rows"]
    selected = validate_selection(ranking, index_rows, file_sha256(model_path))

    frozen = {"C_L": file_sha256(C_L), "pi_D": file_sha256(PI_D / "params.pkl"), "pi_L": file_sha256(PI_L / "params.pkl")}
    expected = {"C_L": EXPECTED["C_L"], "pi_D": EXPECTED["pi_D"], "pi_L": EXPECTED["pi_L"]}
    if frozen != expected:
        raise SystemExit(f"frozen scientific asset mismatch: {frozen}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")

    root.mkdir(parents=True)
    save_json(root / "manifest.json", {
        "status": "FROZEN_BEFORE_OUTCOMES", "selection": "parent-held-out physical-16D kernel ranking",
        "ranking_path": str(ranking_path), "ranking_sha256": file_sha256(ranking_path),
        "model_path": str(model_path), "model_sha256": file_sha256(model_path),
        "index_sha256": file_sha256(index_path), "tube_sha256": file_sha256(tube_path),
        "rows": selected,
    })
    save_json(root / "cost_estimate.json", {
        "estimated_seconds": 900, "states": 12, "rollouts_per_state": "2 exact + 4 P1 micro",
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
    nodes = [{
        "node_id": row["proposal_id"], "candidate_id": row["candidate_id"],
        "layer": row["shell_layer"], "region": row["region"],
        "source_state_hash": row["physical_state_sha256"], "physical_state": row,
        "parent_node_id": row.get("nearest_downstream_node_id"),
    } for row in selected]
    certification = certify_policy(
        env, descent_policy, landing_policy, nodes, 3_700_000_000,
        record_loader=lambda node: _load_record(node["physical_state"]),
        descent_action_adapter=adapter, policy_identity_hash=artifact["policy_identity_hash"],
    )
    save_json(root / "certification.json", certification)
    p0 = {row["node_id"] for row in certification["rows"] if row["P0"]["pass"]}
    p1 = {row["node_id"] for row in certification["rows"] if row["P1"]["pass"]}
    accepted = []
    for row in selected:
        if row["proposal_id"] not in p1:
            continue
        record = copy.deepcopy(_load_record(row))
        record.update({
            "id": row["proposal_id"], "artifact_role": "proposal_support_bank",
            "safe_claim_allowed": False, "bootstrap_eligible": True,
            "reachability_score": row["reachability_score"],
            "policy_identity_hash": artifact["policy_identity_hash"],
        })
        accepted.append(record)
    output_bank = root / "ranked_proposal_support.pkl"
    SnapshotBank(accepted, {
        "artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
        "policy_identity_hash": artifact["policy_identity_hash"],
        "ranking_sha256": file_sha256(ranking_path),
    }).save(output_bank)
    p1_parents = len({row["candidate_id"] for row in selected if row["proposal_id"] in p1})
    status = "PASS" if len(p1) >= 2 and p1_parents >= 2 else "FAIL"
    report = {
        "status": status, "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "states": len(selected), "P0": len(p0), "P1": len(p1), "P1_parents": p1_parents,
        "regions": {region: {
            "selected": sum(row["region"] == region for row in selected),
            "P0": sum(row["region"] == region and row["proposal_id"] in p0 for row in selected),
            "P1": sum(row["region"] == region and row["proposal_id"] in p1 for row in selected),
        } for region in ("early", "middle", "late")},
        "proposal_bank": str(output_bank), "proposal_bank_sha256": file_sha256(output_bank),
        "formal_tube_or_matcher": False, "PPO_authorization": False,
        "next": "fresh_independent_audit_before_tube_extension" if status == "PASS" else "ranked_parent_support_insufficient",
    }
    save_json(root / "DESCENT_RANKED_REACHABILITY_PILOT_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
