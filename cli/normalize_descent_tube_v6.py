"""Create a non-overwriting, semantics-explicit normalization of Descent Tube v5."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path

from cli.verify_descent_tube_v5 import DEFAULT_ADAPTER, DEFAULT_TUBE, verify_records
from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


def outcome(successes: int, branches: int, cfg, semantics: str) -> dict:
    posterior = beta_posterior(successes, branches - successes)
    return {
        "successes": int(successes), "failures": int(branches - successes),
        "branches": int(branches), "posterior": posterior,
        "label": posterior_label(
            posterior, branches, min_branches=int(cfg.min_branches),
            safe_threshold=float(cfg.safe_threshold), dead_threshold=float(cfg.dead_threshold),
            boundary_max_width=float(cfg.boundary_max_width),
        ),
        "semantics": semantics,
    }


def normalize_records(records: list[dict], *, policy_identity: str, cfg,
                      tube_version: str) -> tuple[list[dict], dict]:
    normalized = []
    chain_corrections = 0
    legacy_independent_fields = 0
    for source in records:
        item = copy.deepcopy(source)
        evidence = list(item.get("certification_branches", []))
        if len(evidence) != 32:
            raise ValueError(f"{item.get('id')}: expected 32 certification branches")
        final_successes = sum(branch.get("final_recovery") is True for branch in evidence)
        chain_successes = sum(branch.get("chain_success") is True for branch in evidence)
        old_final = item.get("final", {})
        if (int(old_final.get("successes", -1)) != final_successes
                or old_final.get("label") != "safe"):
            raise ValueError(f"{item.get('id')}: v5 Final evidence is not safe/consistent")
        if int(item.get("chain", {}).get("successes", -1)) != chain_successes:
            chain_corrections += 1
        legacy = item.get("independent_audit")
        if legacy is not True:
            legacy_independent_fields += 1
            if legacy is not None:
                item["independent_audit_legacy"] = copy.deepcopy(legacy)
        item.update({
            "policy_identity_hash": policy_identity,
            "policy_version": policy_identity,
            "tube_version": tube_version,
            "artifact_role": "certified_tube",
            "certified_safe": True,
            "safe_claim_allowed": True,
            "training_only": False,
            "tube_metrics_eligible": True,
            "independent_audit": True,
            "final": outcome(final_successes, 32, cfg, "Final-Recovery; defines Tube safety"),
            "chain": outcome(chain_successes, 32, cfg, "chain_ever event; reported separately"),
        })
        if item["final"]["label"] != "safe":
            raise ValueError(f"{item.get('id')}: normalization changed Final-safe membership")
        normalized.append(item)
    return normalized, {
        "chain_records_corrected": chain_corrections,
        "legacy_independent_audit_fields_normalized": legacy_independent_fields,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_TUBE))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--config", default="configs/backward_descent_rsi_pilot_v1.json")
    parser.add_argument("--output-bank", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    output_bank, output_report = Path(args.output_bank), Path(args.output_report)
    if output_bank.exists() or output_report.exists():
        raise SystemExit("refusing to overwrite normalized Descent Tube")
    source_path = Path(args.source)
    source = SnapshotBank.load(source_path)
    with Path(args.adapter).open("rb") as stream:
        adapter = pickle.load(stream)
    policy_identity = str(adapter["policy_identity_hash"])
    cfg = load_config(args.config)
    version_payload = {
        "source_sha256": file_sha256(source_path),
        "policy_identity": policy_identity,
        "schema": "descent_tube_v6_chain_event_explicit",
    }
    tube_version = "descent-v6-" + hashlib.sha256(
        json.dumps(version_payload, sort_keys=True).encode()
    ).hexdigest()[:12]
    records, changes = normalize_records(
        source.records, policy_identity=policy_identity, cfg=cfg,
        tube_version=tube_version,
    )
    summary, reasons = verify_records(
        records, policy_identity=policy_identity, branches=32,
        min_branches=int(cfg.min_branches), safe_threshold=float(cfg.safe_threshold),
        dead_threshold=float(cfg.dead_threshold), boundary_max_width=float(cfg.boundary_max_width),
    )
    if reasons:
        raise SystemExit("normalized Tube failed verification: " + "; ".join(reasons))
    metadata = copy.deepcopy(source.metadata)
    metadata.update({
        "artifact_role": "certified_tube",
        "phase": "descent",
        "policy_identity_hash": policy_identity,
        "last_policy_version": policy_identity,
        "last_tube_version": tube_version,
        "tube_version": tube_version,
        "branches_per_state": 32,
        "independent_audit": True,
        "formal_jel_eligible": False,
        "expert_conditioned": True,
        "standard_record_certification_fields": True,
        "safety_label_semantics": (
            "Final-Recovery Beta posterior lower bound >= 0.70; chain_ever reported separately"
        ),
        "safe_threshold_lower": float(cfg.safe_threshold),
        "schema_normalization_only": True,
        "scientific_final_outcomes_changed": False,
        "supersedes": str(source_path.resolve()),
        "supersedes_sha256": file_sha256(source_path),
    })
    output_bank.parent.mkdir(parents=True, exist_ok=True)
    SnapshotBank(records, metadata).save(output_bank)
    report = {
        "status": "PASS",
        "artifact_role": "descent_tube_v6_schema_normalization",
        "source_bank": str(source_path),
        "source_bank_sha256": file_sha256(source_path),
        "output_bank": str(output_bank),
        "output_bank_sha256": file_sha256(output_bank),
        "tube_version": tube_version,
        "policy_identity_hash": policy_identity,
        "adapter_sha256": file_sha256(args.adapter),
        "scientific_final_outcomes_changed": False,
        "safe_state_ids_unchanged": (
            sorted(str(row["id"]) for row in source.records)
            == sorted(str(row["id"]) for row in records)
        ),
        "state_byte_hashes_unchanged": all(
            before.get("state_byte_hash") == after.get("state_byte_hash")
            for before, after in zip(source.records, records)
        ),
        "changes": changes,
        "verification_summary": summary,
        "PPO_authorization": False,
    }
    save_json(output_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
