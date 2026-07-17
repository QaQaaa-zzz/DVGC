"""Build the persistent exact seed registry for the descent-Tube route."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.config import file_sha256
from dvgc.runtime import save_json
from dvgc.seed_registry import (
    branch_seed_grid,
    exact_intersection_proof,
    make_claim,
    report_branch_seeds,
    save_registry,
)


RUN = Path("runs/stage_experts/descent_tube_seed0_20260716T2330")
BLOCK1_REPORT = Path("runs/stage_experts/descent_local_nonfinite_repair_seed0_20260716T1825/blocks/block_1_25600/current_policy_certified_sharded.cert.json")
ENTRY = Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl")
LANDING = Path("runs/landing/refinement_seed0/policy/params.pkl")
XML = Path("assets/orange_bike_4kg_horizontal.xml")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_claim(name, category, path, *, status="valid"):
    payload = load(path)
    return make_claim(
        name, category, report_branch_seeds(payload), status=status,
        base_seed=payload.get("seed"), seed_namespace=payload.get("seed_namespace"),
        artifact=str(path), artifact_sha256=file_sha256(path),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    run = args.run
    round1_audit = run/"round_1/pointwise_audit_seed9310000/merged.json"
    round2_construction = run/"round_2/construction/current_policy.cert.json"
    invalid_audit = run/"round_2/pointwise_audit_seed9330000/merged.json"
    current_root = run/"round_2/pointwise_audit_seed200000000"
    current_candidate = run/"round_2/frozen/D_all_unique.pkl"
    current_policy = run/"round_2/train/policy/params.pkl"

    required = [BLOCK1_REPORT, round1_audit, round2_construction, invalid_audit,
                current_candidate, current_policy, ENTRY, LANDING, XML]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing seed-registry inputs: {missing}")

    claims = [
        make_claim("descent_candidate_generation", "candidate_generation", [7_400_000],
                   status="valid", seed_semantics="fixed generation root seed"),
        make_claim("descent_ppo_continuation", "ppo", [0], status="valid",
                   seed_semantics="shared root seed for exact optimizer continuation"),
        artifact_claim("block1_construction", "construction_certification", BLOCK1_REPORT),
        artifact_claim("round1_pointwise_audit", "pointwise_audit", round1_audit),
        artifact_claim("block2_construction", "construction_certification", round2_construction),
        artifact_claim("round2_pointwise_seed9330000", "pointwise_audit", invalid_audit,
                       status="invalid_diagnostic_excluded"),
    ]
    planned = make_claim(
        "round2_pointwise_seed200000000", "pointwise_audit",
        branch_seed_grid(200_000_000, 98, 32), status="active",
        base_seed=200_000_000, state_count=98, branches_per_state=32,
        branch_variation_indices=list(range(32)),
    )
    proof = exact_intersection_proof(planned, claims)
    if proof["status"] != "PASS":
        raise SystemExit(f"Current pointwise audit seed conflict: {proof['intersection_preview']}")
    claims.append(planned)
    for name, category in (
        ("acquisition_certification", "acquisition_certification"),
        ("continuous_matcher_audit", "matcher_audit"),
        ("final_independent_audit", "final_independent_audit"),
    ):
        claims.append(make_claim(name, category, [], status="not_allocated"))

    registry_path = run/"seed_registry.json"
    registry = save_registry(
        registry_path, claims, status="ACTIVE",
        note="Historical intersections are retained as evidence; every future claim must pass an exact set check before launch.",
    )
    current_root.mkdir(parents=True, exist_ok=True)
    proof_path = current_root/"seed_intersection_proof.json"
    save_json(proof_path, {
        **proof,
        "registry": str(registry_path),
        "registry_sha256": file_sha256(registry_path),
        "invalid_seed9330000_excluded": True,
    })
    manifest_path = current_root/"pointwise_audit_manifest.json"
    save_json(manifest_path, {
        "status": "ACTIVE",
        "seed": 200_000_000,
        "seed_namespace": "descent_pointwise_round_2:descent_entry",
        "global_indices": [0, 98],
        "states": 98,
        "branches_per_state": 32,
        "branch_variation_indices": list(range(32)),
        "seed_set_sha256": planned["seed_set_sha256"],
        "seed_intersection_proof": str(proof_path),
        "seed_intersection_proof_sha256": file_sha256(proof_path),
        "policy_hash": file_sha256(current_policy),
        "candidate_bank_sha256": file_sha256(current_candidate),
        "xml_sha256": file_sha256(XML),
        "landing_entry_set_sha256": file_sha256(ENTRY),
        "landing_policy_hash": file_sha256(LANDING),
        "exact_membership_only": True,
        "continuous_matcher_active": False,
        "invalid_seed9330000_excluded": True,
    })
    print(json.dumps({
        "status": "PASS", "registry": str(registry_path),
        "claims": len(registry["claims"]),
        "historical_intersections": len(registry["historical_intersections"]),
        "current_intersection_count": proof["intersection_count"],
        "manifest": str(manifest_path),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
