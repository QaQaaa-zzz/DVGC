"""Verify the frozen Descent Tube-v5/v6 evidence and provenance contract."""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

from dvgc.bank import SnapshotBank, beta_posterior, posterior_label
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


DEFAULT_TUBE = Path(
    "runs/descent_reachability_network_v3/"
    "independent_tube_extension_3x32_20260729/descent_tube_v5.pkl"
)
DEFAULT_EXTENSION_REPORT = DEFAULT_TUBE.with_name("DESCENT_NETWORK_TUBE_EXTENSION_V1_REPORT.json")
DEFAULT_EXTENSION_MANIFEST = DEFAULT_TUBE.with_name("manifest.json")
DEFAULT_ADAPTER = Path("runs/descent_diverse_p1_predecessor_recovery_v5_p1_core_adapter/adapter.pkl")
DEFAULT_C_L = Path("runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl")
DEFAULT_XML = Path("assets/orange_bike_4kg_horizontal.xml")


def verify_records(records: list[dict], *, policy_identity: str, branches: int,
                   min_branches: int, safe_threshold: float, dead_threshold: float,
                   boundary_max_width: float) -> tuple[dict, list[str]]:
    reasons: list[str] = []
    seeds = []
    success_histogram = Counter()
    failure_reasons = Counter()
    namespaces = Counter()
    for row in records:
        record_id = str(row.get("id"))
        evidence = list(row.get("certification_branches", []))
        if len(evidence) != branches:
            reasons.append(f"{record_id}: branch count {len(evidence)} != {branches}")
            continue
        final_successes = sum(branch.get("final_recovery") is True for branch in evidence)
        chain_successes = sum(branch.get("chain_success") is True for branch in evidence)
        success_histogram[final_successes] += 1
        for branch in evidence:
            seed = branch.get("branch_seed", branch.get("seed"))
            seeds.append(seed)
            namespaces[str(branch.get("seed_namespace"))] += 1
            if branch.get("final_recovery") is not True:
                failure_reasons[str(branch.get("end_reason", "unknown"))] += 1
            if str(branch.get("end_reason")) in {"timeout", "nonfinite"}:
                reasons.append(f"{record_id}: forbidden {branch.get('end_reason')} branch")
        final = row.get("final", {})
        expected_final_label = posterior_label(
            beta_posterior(final_successes, branches - final_successes), branches,
            min_branches=min_branches, safe_threshold=safe_threshold,
            dead_threshold=dead_threshold, boundary_max_width=boundary_max_width,
        )
        if (int(final.get("successes", -1)) != final_successes
                or int(final.get("failures", -1)) != branches - final_successes
                or int(final.get("branches", -1)) != branches
                or final.get("label") != expected_final_label
                or expected_final_label != "safe"):
            reasons.append(f"{record_id}: Final posterior/evidence mismatch")
        chain = row.get("chain", {})
        expected_chain_label = posterior_label(
            beta_posterior(chain_successes, branches - chain_successes), branches,
            min_branches=min_branches, safe_threshold=safe_threshold,
            dead_threshold=dead_threshold, boundary_max_width=boundary_max_width,
        )
        if (int(chain.get("successes", -1)) != chain_successes
                or int(chain.get("failures", -1)) != branches - chain_successes
                or int(chain.get("branches", -1)) != branches
                or chain.get("label") != expected_chain_label):
            reasons.append(f"{record_id}: Chain posterior/evidence mismatch")
        if not (row.get("artifact_role") == "certified_tube"
                and row.get("certified_safe") is True
                and row.get("safe_claim_allowed") is True
                and row.get("training_only") is False
                and row.get("tube_metrics_eligible") is True
                and row.get("independent_audit") is True):
            reasons.append(f"{record_id}: certified Tube flags")
        row_policy = row.get("policy_identity_hash") or row.get("policy_version")
        if row_policy != policy_identity:
            reasons.append(f"{record_id}: policy identity")
    if None in seeds or len(seeds) != len(set(seeds)):
        reasons.append("branch seeds absent or duplicated across Tube")
    summary = {
        "states": len(records),
        "branches_per_state": branches,
        "total_branches": len(seeds),
        "unique_branch_seeds": len(set(seeds)),
        "exact_32_of_32_states": success_histogram.get(branches, 0),
        "posterior_safe_with_failures_states": sum(
            count for successes, count in success_histogram.items() if successes < branches
        ),
        "final_success_histogram": {str(key): value for key, value in sorted(success_histogram.items())},
        "physical_failure_reasons": dict(failure_reasons),
        "seed_namespaces": dict(namespaces),
    }
    return summary, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tube", default=str(DEFAULT_TUBE))
    parser.add_argument("--extension-report", default=str(DEFAULT_EXTENSION_REPORT))
    parser.add_argument("--extension-manifest", default=str(DEFAULT_EXTENSION_MANIFEST))
    parser.add_argument("--normalization-report", default=None)
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--canonical-entry", default=str(DEFAULT_C_L))
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--config", default="configs/backward_descent_rsi_pilot_v1.json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")

    tube_path = Path(args.tube)
    bank = SnapshotBank.load(tube_path)
    extension = json.loads(Path(args.extension_report).read_text())
    manifest = json.loads(Path(args.extension_manifest).read_text())
    normalization = (json.loads(Path(args.normalization_report).read_text())
                     if args.normalization_report else None)
    with Path(args.adapter).open("rb") as stream:
        adapter = pickle.load(stream)
    cfg = load_config(args.config)
    policy_identity = str(adapter["policy_identity_hash"])
    summary, row_reasons = verify_records(
        bank.records, policy_identity=policy_identity,
        branches=int(bank.metadata.get("branches_per_state", 0)),
        min_branches=int(cfg.min_branches), safe_threshold=float(cfg.safe_threshold),
        dead_threshold=float(cfg.dead_threshold), boundary_max_width=float(cfg.boundary_max_width),
    )

    base_report_path = Path(str(bank.metadata.get("certification_report", "")))
    inputs = manifest.get("inputs", {})
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    from cli.runtime_gate import source_fingerprint
    source_tube_hash = (str(bank.metadata.get("supersedes_sha256"))
                        if normalization is not None else file_sha256(tube_path))
    checks = {
        "tube_role": bank.metadata.get("artifact_role") == "certified_tube",
        "descent_phase": bank.metadata.get("phase") == "descent",
        "independent_audit": bank.metadata.get("independent_audit") is True,
        "not_formal_jel": bank.metadata.get("formal_jel_eligible") is False,
        "final_and_chain_separate": (
            "Final-Recovery" in str(bank.metadata.get("safety_label_semantics"))
            and "Chain" in str(bank.metadata.get("safety_label_semantics")).replace("chain", "Chain")
            and "separate" in str(bank.metadata.get("safety_label_semantics"))
        ),
        "policy_identity": bank.metadata.get("policy_identity_hash") == policy_identity,
        "adapter_identity": bank.metadata.get("adapter_sha256") == file_sha256(args.adapter),
        "base_report_exists": base_report_path.is_file(),
        "base_report_identity": (base_report_path.is_file()
                                 and bank.metadata.get("certification_report_sha256")
                                 == file_sha256(base_report_path)),
        "extension_report_pass": extension.get("status") == "PASS",
        "extension_report_tube_identity": extension.get("tube_sha256") == source_tube_hash,
        "extension_manifest_frozen": manifest.get("status") == "FROZEN_BEFORE_AUDIT",
        "manifest_policy_identity": inputs.get("policy_identity_hash") == policy_identity,
        "manifest_adapter_identity": inputs.get("adapter", {}).get("sha256") == file_sha256(args.adapter),
        "manifest_canonical_entry_identity": inputs.get("C_L") == file_sha256(args.canonical_entry),
        "manifest_xml_identity": inputs.get("xml") == file_sha256(args.xml),
        "runtime_gate_current": (gate.get("status") == "PASS"
                                 and gate.get("source_fingerprint") == source_fingerprint(Path.cwd())),
        "row_contract": not row_reasons,
    }
    if normalization is not None:
        checks.update({
            "normalization_report_pass": normalization.get("status") == "PASS",
            "normalization_output_identity": (
                normalization.get("output_bank_sha256") == file_sha256(tube_path)
            ),
            "normalization_source_identity": (
                normalization.get("source_bank_sha256") == source_tube_hash
                and bank.metadata.get("supersedes_sha256") == source_tube_hash
            ),
            "normalization_final_outcomes_unchanged": (
                normalization.get("scientific_final_outcomes_changed") is False
                and normalization.get("safe_state_ids_unchanged") is True
                and normalization.get("state_byte_hashes_unchanged") is True
            ),
        })
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "artifact_role": "descent_tube_independent_structure_verification",
        "checks": checks,
        "row_reasons": row_reasons,
        "summary": summary,
        "posterior_safe_definition": {
            "branches": int(bank.metadata.get("branches_per_state", 0)),
            "safe_threshold_lower": float(cfg.safe_threshold),
            "min_branches": int(cfg.min_branches),
            "boundary_max_width": float(cfg.boundary_max_width),
        },
        "tube_sha256": file_sha256(tube_path),
        "policy_identity_hash": policy_identity,
        "adapter_sha256": file_sha256(args.adapter),
        "canonical_entry_sha256": file_sha256(args.canonical_entry),
        "xml_sha256": file_sha256(args.xml),
    }
    save_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
