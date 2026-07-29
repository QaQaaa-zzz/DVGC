"""Eight-branch frozen-policy screen for network-ranked Descent candidates."""
from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.audit_descent_compact_adapter_v1 import _perturb_batch
from cli.run_backward_descent_nominal_pilot import C_L, EXPECTED, PI_D, PI_L
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import compact_observation_command_adapter, make_descent_landing_rollout
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS, branch_seed
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


PILOTS = (
    Path("runs/descent_reachability_network_v3/final_semantics_ranked_pilot_4x4_20260729/DESCENT_REACHABILITY_RANKED_PILOT_V1_REPORT.json"),
    Path("runs/descent_reachability_network_v3/final_semantics_ranked_pilot_remaining_20260729/DESCENT_REACHABILITY_RANKED_PILOT_V1_REPORT.json"),
)
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
BASE_TUBE = Path("runs/descent_natural_bridge_candidates_v1/independent_audit_round2_round3_2x32/descent_tube_v4.pkl")
DEFAULT_RUN = Path("runs/descent_reachability_network_v3/frozen_policy_screen_3x8_20260729")
BRANCHES = 8
SEED = 3_810_000_000
NAMESPACE = "descent-network-ranked-frozen-policy-screen-v1"


def _atomic_bank(path: Path, records: list[dict], metadata: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    SnapshotBank(records, metadata).save(temporary)
    os.replace(temporary, path)


def collect_final_safe_candidates(pilots: list[Path]) -> list[dict]:
    records, identities = [], set()
    cache: dict[str, list] = {}
    for path in pilots:
        if "independent_audit" in str(path).lower():
            raise ValueError("independent audit labels are forbidden")
        report = json.loads(path.read_text())
        if report.get("artifact_role") != "reachability_ranked_construction_certification_pilot":
            raise ValueError(f"ineligible pilot role: {path}")
        for row in report["rows"]:
            if not row["final_safety_P1"]["pass"]:
                continue
            proposal = row["proposal"]
            artifact = proposal["source_artifact"]
            if artifact not in cache:
                cache[artifact] = pickle.loads(Path(artifact).read_bytes())
            source = cache[artifact][int(proposal["source_index"])]
            identity = proposal["physical_state_hash"]
            if source["physical_state_hash"] != identity or identity in identities:
                raise ValueError(f"candidate identity mismatch/duplicate: {identity}")
            identities.add(identity)
            record = copy.deepcopy(source["snapshot_v4"])
            record.update({
                "id": identity, "state_byte_hash": identity,
                "candidate_id": proposal["candidate_id"],
                "descent_region": proposal["region"], "descent_layer": proposal["shell_layer"],
                "source_artifact": artifact, "source_index": proposal["source_index"],
                "proposal_score": proposal["predicted_p_safe"],
                "artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
            })
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--pilot", action="append", default=[])
    args = parser.parse_args()
    root = Path(args.run)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    pilots = [Path(path) for path in args.pilot] or list(PILOTS)
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    if file_sha256(C_L) != EXPECTED["C_L"] or file_sha256(PI_D / "params.pkl") != EXPECTED["pi_D"] or file_sha256(PI_L / "params.pkl") != EXPECTED["pi_L"]:
        raise SystemExit("frozen policy/C_L mismatch")
    artifact = pickle.loads((EXPERT / "adapter.pkl").read_bytes())
    base_tube = SnapshotBank.load(BASE_TUBE)
    if artifact["policy_identity_hash"] != base_tube.metadata["policy_identity_hash"]:
        raise SystemExit("frozen Tube policy identity mismatch")
    candidates = collect_final_safe_candidates(pilots)
    base_ids = {row["id"] for row in base_tube.records}
    if any(row["id"] in base_ids for row in candidates):
        raise SystemExit("candidate already exists in base Tube")
    root.mkdir(parents=True)
    inputs = {
        "pilots": [{"path": str(path), "sha256": file_sha256(path)} for path in pilots],
        "base_tube": {"path": str(BASE_TUBE), "sha256": file_sha256(BASE_TUBE)},
        "adapter": {"path": str(EXPERT / "adapter.pkl"), "sha256": file_sha256(EXPERT / "adapter.pkl")},
        "policy_identity_hash": artifact["policy_identity_hash"],
        "seed": SEED, "namespace": NAMESPACE,
    }
    save_json(root / "manifest.json", {"status": "FROZEN_BEFORE_OUTCOMES", "inputs": inputs,
        "states": len(candidates), "branches": BRANCHES, "dynamics_variants": DYNAMICS_VARIANTS,
        "gate": "at least 7/8 legal Final-Recovery; Chain reported separately"})
    save_json(root / "cost_estimate.json", {"estimated_seconds": 300, "states": len(candidates),
        "branches_per_state": BRANCHES, "rollouts": len(candidates) * BRANCHES, "PPO_steps": 0})

    dparams, _, _ = load_bundle(PI_D, verify_files=True)
    lparams, _, _ = load_bundle(PI_L, verify_files=True)
    variants = []
    for spec in DYNAMICS_VARIANTS:
        cfg = load_config("configs/backward_descent_rsi_pilot_v1.json", {
            "use_bank_resets": False, "expert_chain_termination": False,
            "domain_randomization": False, "obs_noise_enable": False,
            **{key: value for key, value in spec.items() if key != "id"},
        })
        env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
        adapter = compact_observation_command_adapter(
            jnp.asarray(artifact["prototypes"]), jnp.asarray(artifact["targets"]),
            jnp.asarray(artifact["normalizer_mean"]), jnp.asarray(artifact["normalizer_std"]),
            float(artifact["radius"]), float(artifact["core_radius"]),
        )
        variants.append((spec["id"], env, make_descent_landing_rollout(
            env, dparams, lparams, horizon=200, residual_ticks=8, descent_action_adapter=adapter)))

    rows, survivors = [], []
    for state_index, record in enumerate(candidates):
        evidence = []
        for variant_index, (variant, env, rollout) in enumerate(variants):
            indices = list(range(variant_index, BRANCHES, len(variants)))
            seeds, deltas = [], []
            for branch in indices:
                seed = branch_seed(SEED, state_index, branch)
                seeds.append(seed)
                deltas.append(np.random.default_rng(seed).uniform(-0.02, 0.02, 2).astype(np.float32))
            batch = _perturb_batch(env, record, seeds, deltas)
            raw = jax.device_get(rollout(batch, jnp.zeros((len(seeds), 2, 4), jnp.float32),
                                         jax.random.PRNGKey(branch_seed(SEED, state_index, variant_index))))
            for local, branch in enumerate(indices):
                code = int(np.asarray(raw["end_code"])[local])
                final = bool(np.asarray(raw["final_recovery"])[local])
                chain = bool(np.asarray(raw["downstream_entry"])[local])
                evidence.append({
                    "branch_index": branch, "branch_seed": seeds[local], "dynamics_variant": variant,
                    "final_recovery": final, "chain_success": chain,
                    "end_code": code, "end_reason": END_REASON.get(code, "unknown"),
                    "termination_tick": int(np.asarray(raw["termination_tick"])[local]),
                })
        evidence.sort(key=lambda row: row["branch_index"])
        finals = sum(row["final_recovery"] for row in evidence)
        chains = sum(row["chain_success"] and row["final_recovery"] for row in evidence)
        nonfinite = sum(row["end_reason"] == "nonfinite" for row in evidence)
        passed = finals >= 7 and nonfinite == 0
        rows.append({"id": record["id"], "candidate_id": record["candidate_id"],
                     "final": finals, "chain_final": chains, "branches": BRANCHES,
                     "pass": passed, "evidence": evidence})
        if passed:
            item = copy.deepcopy(record)
            item.update({"screen_final": finals, "screen_chain_final": chains,
                         "screen_branches": BRANCHES, "screen_namespace": NAMESPACE})
            survivors.append(item)
        save_json(root / "screen.partial.json", {"inputs": inputs, "rows": rows})
        print(f"[frozen-policy-screen] {state_index + 1}/{len(candidates)} Final={finals}/8 Chain+Final={chains}/8", flush=True)
    status = "PASS" if survivors else "FAIL"
    output = root / "frozen_policy_screen_survivors.pkl"
    _atomic_bank(output, survivors, {"artifact_role": "proposal_support_bank", "safe_claim_allowed": False,
        "policy_identity_hash": artifact["policy_identity_hash"], "screen_namespace": NAMESPACE,
        "source_manifest_sha256": file_sha256(root / "manifest.json")})
    report = {
        "status": status, "artifact_role": "frozen_policy_candidate_screen",
        "formal_tube_or_matcher": False, "states": len(candidates), "survivors": len(survivors),
        "rows": rows, "failure_reasons": dict(Counter(
            event["end_reason"] for row in rows for event in row["evidence"] if not event["final_recovery"])),
        "survivor_bank": str(output), "survivor_bank_sha256": file_sha256(output),
        "PPO_authorization": False,
        "next": "independent_32_branch_tube_extension_audit" if status == "PASS" else "frozen_policy_support_gap",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    save_json(root / "DESCENT_NETWORK_FROZEN_POLICY_SCREEN_V1_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps({key: report[key] for key in (
        "status", "states", "survivors", "failure_reasons", "survivor_bank", "PPO_authorization", "next")}, indent=2))


if __name__ == "__main__":
    main()
