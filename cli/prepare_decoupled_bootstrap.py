"""Freeze immutable pi_L/C_L/pi_F inputs for decoupled expert bootstrap."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256
from dvgc.experts import StageExpertRegistry, policy_bundle_hash
from dvgc.policy import copy_bundle, load_bundle
from dvgc.runtime import save_json

CANONICAL_C_L_ROLES = frozenset({
    "canonical_certified_landing_entry_set",
    "canonical_certified_landing_entry_set_extended",
})


def _owned_copy(source: Path, destination: Path) -> Path:
    if not destination.exists():
        copy_bundle(source, destination)
    if policy_bundle_hash(source) != policy_bundle_hash(destination):
        raise SystemExit(f"Owned policy copy differs from source: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landing-policy", required=True)
    parser.add_argument("--flight-policy", required=True)
    parser.add_argument("--landing-entry-set", required=True)
    parser.add_argument("--flight-bank", required=True)
    parser.add_argument("--runtime-gate", default="docs/RUNTIME_GATE.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    contract_path = root / "frozen_contract.json"
    registry_path = root / "expert_registry.json"
    landing_source = Path(args.landing_policy).resolve()
    flight_source = Path(args.flight_policy).resolve()
    entry_path = Path(args.landing_entry_set).resolve()
    bank_path = Path(args.flight_bank).resolve()
    gate = json.loads(Path(args.runtime_gate).read_text())
    if gate.get("status") != "PASS" or not gate.get("source_fingerprint"):
        raise SystemExit("Runtime gate is not PASS/current")

    _, landing_cfg, landing_manifest = load_bundle(landing_source, verify_files=True)
    _, flight_cfg, flight_manifest = load_bundle(flight_source, verify_files=True)
    if landing_manifest.get("stage") != "landing" or flight_manifest.get("stage") != "flight":
        raise SystemExit("Policy stages do not match pi_L/pi_F ownership")
    for key in ("xml_sha256", "action_mapping_version"):
        if landing_manifest.get(key) != flight_manifest.get(key):
            raise SystemExit(f"pi_L/pi_F {key} mismatch")
    if landing_cfg["actor_history_steps"] != flight_cfg["actor_history_steps"]:
        raise SystemExit("pi_L/pi_F PolicyState history mismatch")

    entry = SnapshotBank.load(entry_path)
    safe = entry.records_for_phase("landing", final_labels=["safe"], include_training_only=False)
    matcher = entry.metadata.get("entry_matcher") or {}
    if entry.metadata.get("entry_bank_role") not in CANONICAL_C_L_ROLES:
        raise SystemExit("C_L is not a canonical certified Landing-entry set")
    if not safe or float(matcher.get("radius", 0.0)) <= 0.0:
        raise SystemExit("C_L has no Final-safe entries or calibrated matcher")
    if entry.metadata.get("last_policy_version") != landing_manifest.get("policy_version"):
        raise SystemExit("C_L was not certified by the frozen pi_L version")

    landing_owned = _owned_copy(landing_source, root / "pi_l_frozen")
    flight_owned = _owned_copy(flight_source, root / "pi_f_init")
    registry = StageExpertRegistry.build(
        {"landing": landing_owned, "flight": flight_owned},
        {"flight": entry_path},
        runtime_source_fingerprint=gate["source_fingerprint"],
    )
    if registry_path.exists():
        existing = StageExpertRegistry.load(registry_path)
        if existing.registry_hash != registry.registry_hash:
            raise SystemExit("Existing decoupled registry has different inputs")
    else:
        registry.save(registry_path)

    contract = {
        "status": "PASS",
        "protocol": "decoupled_bootstrap_experts_then_shared_consolidation_v1",
        "artifact_role": "expert_bootstrap_frozen_contract",
        "landing": {
            "policy": str(landing_owned.resolve()),
            "policy_version": landing_manifest["policy_version"],
            "policy_hash": file_sha256(landing_owned / "params.pkl"),
            "bundle_hash": policy_bundle_hash(landing_owned),
        },
        "flight_initial": {
            "policy": str(flight_owned.resolve()),
            "policy_version": flight_manifest["policy_version"],
            "policy_hash": file_sha256(flight_owned / "params.pkl"),
            "bundle_hash": policy_bundle_hash(flight_owned),
            "training_objective": "Flight_to_fixed_canonical_C_L",
            "landing_retention_required": False,
        },
        "canonical_c_l": {
            "path": str(entry_path),
            "sha256": file_sha256(entry_path),
            "version": entry.metadata.get("last_tube_version"),
            "final_safe_count": len(safe),
            "matcher_version": matcher.get("version"),
            "matcher_radius": matcher.get("radius"),
            "entry_window_steps": matcher.get("entry_window_steps"),
            "immutable_during_flight_training": True,
        },
        "flight_candidate_bank": {"path": str(bank_path), "sha256": file_sha256(bank_path)},
        "registry": {"path": str(registry_path.resolve()), "hash": registry.registry_hash},
        "runtime": {
            "gate": str(Path(args.runtime_gate).resolve()),
            "source_fingerprint": gate["source_fingerprint"],
            "xml_sha256": gate.get("xml_sha256"),
        },
        "handoff": {
            "irreversible": True,
            "preserve_physics": True,
            "preserve_policy_state_and_history": True,
        },
        "expert_certification_role": "expert_conditioned_provisional_envelope",
        "formal_jel_role": "final_shared_policy_jel",
    }
    if contract_path.exists():
        existing = json.loads(contract_path.read_text())
        if existing != contract:
            raise SystemExit("Existing frozen contract differs from current inputs")
    else:
        save_json(contract_path, contract)
    print(json.dumps(contract, indent=2))


if __name__ == "__main__":
    main()
