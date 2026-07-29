"""Evaluate real Apex bridge snapshots under the frozen Descent Tube-v5 stack."""
from __future__ import annotations

import argparse
import copy
import json
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


SOURCE = Path("runs/apex_to_descent_local_pilot_v1/receding_feedback_v1/deterministic/stable_physical_descent.pkl")
TUBE = Path("runs/descent_reachability_network_v3/independent_tube_extension_3x32_20260729/descent_tube_v5.pkl")
EXPERT = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter")
DEFAULT_RUN = Path("runs/apex_to_descent_local_pilot_v1/tube_v5_final_safe_screen_4x8_20260729")
BRANCHES = 8
SEED = 4_210_000_000


def validate_source(records: list[dict]) -> None:
    if len(records) != 4 or len({row["trajectory_parent_id"] for row in records}) != 4:
        raise ValueError("Apex bridge screen requires four parent-disjoint records")
    for row in records:
        if row.get("candidate_kind") != "stable_physical_descent_proposal":
            raise ValueError("unexpected bridge candidate kind")
        if not row.get("apex_seen") or int(row.get("oracle_phase", -1)) != 2:
            raise ValueError("bridge record is not a real post-Apex Flight/Descent state")
        if row.get("formal_descent_support_entry") is not False:
            raise ValueError("this screen must not reuse old formal-entry positives")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=str(DEFAULT_RUN))
    parser.add_argument("--source", default=str(SOURCE))
    args = parser.parse_args()
    root, source_path = Path(args.run), Path(args.source)
    if root.exists():
        raise SystemExit(f"refusing overwrite {root}")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    source = SnapshotBank.load(source_path)
    validate_source(source.records)
    tube = SnapshotBank.load(TUBE)
    artifact = pickle.loads((EXPERT / "adapter.pkl").read_bytes())
    if tube.metadata.get("policy_identity_hash") != artifact["policy_identity_hash"]:
        raise SystemExit("Tube/adapter identity mismatch")
    if file_sha256(C_L) != EXPECTED["C_L"] or file_sha256(PI_D / "params.pkl") != EXPECTED["pi_D"] or file_sha256(PI_L / "params.pkl") != EXPECTED["pi_L"]:
        raise SystemExit("frozen scientific asset mismatch")
    root.mkdir(parents=True)
    inputs = {"source_sha256": file_sha256(source_path), "tube_sha256": file_sha256(TUBE),
              "adapter_sha256": file_sha256(EXPERT / "adapter.pkl"),
              "policy_identity_hash": artifact["policy_identity_hash"], "seed": SEED}
    save_json(root / "manifest.json", {"status": "FROZEN_BEFORE_OUTCOMES", "inputs": inputs,
        "states": 4, "branches": BRANCHES, "dynamics_variants": DYNAMICS_VARIANTS,
        "success": "at least 7/8 legal Final-Recovery; no exact C_D membership prerequisite"})
    save_json(root / "cost_estimate.json", {"estimated_seconds": 300, "rollouts": 4 * BRANCHES,
        "PPO_steps": 0, "new_search": False})
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
    rows, admitted = [], []
    for state_index, record in enumerate(source.records):
        evidence = []
        for variant_index, (variant, env, rollout) in enumerate(variants):
            indices = list(range(variant_index, BRANCHES, len(variants)))
            seeds, deltas = [], []
            for branch in indices:
                seed = branch_seed(SEED, state_index, branch)
                seeds.append(seed)
                deltas.append(np.random.default_rng(seed).uniform(-.02, .02, 2).astype(np.float32))
            batch = _perturb_batch(env, record, seeds, deltas)
            raw = jax.device_get(rollout(batch, jnp.zeros((len(seeds), 2, 4), jnp.float32),
                                         jax.random.PRNGKey(branch_seed(SEED, state_index, variant_index))))
            for local, branch in enumerate(indices):
                code = int(np.asarray(raw["end_code"])[local])
                final = bool(np.asarray(raw["final_recovery"])[local])
                chain = bool(np.asarray(raw["downstream_entry"])[local])
                evidence.append({"branch": branch, "seed": seeds[local], "variant": variant,
                    "final_recovery": final, "chain": chain, "chain_final": chain and final,
                    "end_code": code, "end_reason": END_REASON.get(code, "unknown"),
                    "termination_tick": int(np.asarray(raw["termination_tick"])[local])})
        evidence.sort(key=lambda row: row["branch"])
        finals = sum(row["final_recovery"] for row in evidence)
        chains = sum(row["chain_final"] for row in evidence)
        passed = finals >= 7 and not any(row["end_reason"] == "nonfinite" for row in evidence)
        rows.append({"id": record["id"], "trajectory_parent_id": record["trajectory_parent_id"],
                     "Final": finals, "Chain_Final": chains, "branches": BRANCHES,
                     "admissible": passed, "evidence": evidence})
        if passed:
            item = copy.deepcopy(record)
            item.update({"artifact_role": "expert_conditioned_provisional_envelope",
                         "safe_claim_allowed": False, "tube_metrics_eligible": False,
                         "downstream_policy_identity_hash": artifact["policy_identity_hash"],
                         "screen_final": finals, "screen_chain_final": chains,
                         "candidate_kind": "apex_to_descent_final_safe_entry_proposal"})
            admitted.append(item)
        save_json(root / "screen.partial.json", {"inputs": inputs, "rows": rows})
        print(f"[apex-descent-screen] {state_index + 1}/4 Final={finals}/8 Chain+Final={chains}/8", flush=True)
    output = root / "apex_to_descent_final_safe_entries.pkl"
    SnapshotBank(admitted, {"artifact_role": "expert_conditioned_provisional_envelope",
        "formal_tube_or_jel": False, "safe_claim_allowed": False,
        "source_sha256": file_sha256(source_path), "tube_sha256": file_sha256(TUBE),
        "downstream_policy_identity_hash": artifact["policy_identity_hash"]}).save(output)
    status = "PASS" if len(admitted) >= 2 else "FAIL"
    report = {"status": status, "artifact_role": "apex_to_descent_entry_screen",
        "states": len(rows), "admissible_states": len(admitted),
        "admissible_parents": len({row["trajectory_parent_id"] for row in rows if row["admissible"]}),
        "Final": sum(row["Final"] for row in rows), "Chain_Final": sum(row["Chain_Final"] for row in rows),
        "failure_reasons": dict(Counter(event["end_reason"] for row in rows for event in row["evidence"]
                                         if not event["final_recovery"])),
        "rows": rows, "entry_bank": str(output), "entry_bank_sha256": file_sha256(output),
        "PPO_authorization": False,
        "next": "apex_stage_label_acquisition_against_final_safe_entries" if status == "PASS" else "apex_bridge_downstream_policy_gap",
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    save_json(root / "APEX_TO_DESCENT_TUBE_V5_SCREEN_REPORT.json", report)
    save_json(root / "completed.json", {"status": status, "next": report["next"]})
    print(json.dumps({key: report[key] for key in (
        "status", "states", "admissible_states", "admissible_parents", "Final", "Chain_Final",
        "failure_reasons", "entry_bank", "PPO_authorization", "next")}, indent=2))


if __name__ == "__main__":
    main()
