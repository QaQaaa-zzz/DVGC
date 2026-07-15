"""Record a validated expert gate recovered by an immutable entry extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dvgc.config import file_sha256
from dvgc.experts import StageExpertRegistry
from dvgc.runtime import save_json


def validate(registry, evaluation, bank, entry_set):
    flight = registry.specs["flight"]
    reasons = []
    if evaluation.get("candidate_bank_sha256") != file_sha256(bank): reasons.append("candidate bank mismatch")
    if evaluation.get("entry_set_sha256") != file_sha256(entry_set): reasons.append("entry set mismatch")
    if evaluation.get("registry_hash") != registry.registry_hash: reasons.append("registry mismatch")
    if flight.downstream_entry_set_sha256 != file_sha256(entry_set): reasons.append("expert entry provenance mismatch")
    if float(evaluation.get("chain_rate", 0.0)) <= 0: reasons.append("Chain is zero")
    if float(evaluation.get("composite_final_rate", 0.0)) <= 0.075: reasons.append("composite Final did not exceed baseline")
    if float(evaluation.get("timeout_rate", 1.0)) != 0: reasons.append("composite timeout")
    return reasons


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--registry", required=True); p.add_argument("--evaluation", required=True)
    p.add_argument("--bank", required=True); p.add_argument("--entry-set", required=True)
    p.add_argument("--output", required=True); a = p.parse_args(); out = Path(a.output)
    if out.exists(): raise SystemExit(f"Output exists: {out}")
    registry = StageExpertRegistry.load(a.registry); evaluation = json.loads(Path(a.evaluation).read_text())
    reasons = validate(registry, evaluation, a.bank, a.entry_set)
    payload = {
        "status": "PASS" if not reasons else "FAIL", "stage": "flight", "curriculum": "late_descent",
        "recovery": "independently_certified_entry_extension", "reasons": reasons,
        "policy": registry.specs["flight"].checkpoint_path, "policy_hash": registry.specs["flight"].policy_hash,
        "registry": str(Path(a.registry).resolve()), "registry_hash": registry.registry_hash,
        "entry_set": str(Path(a.entry_set).resolve()), "entry_set_sha256": file_sha256(a.entry_set),
        "candidate_bank_sha256": file_sha256(a.bank), "evaluation": str(Path(a.evaluation).resolve()),
        "chain_rate": evaluation.get("chain_rate"), "composite_final_rate": evaluation.get("composite_final_rate"),
        "chain_missed_final_rate": evaluation.get("chain_missed_final_rate"), "timeout_rate": evaluation.get("timeout_rate"),
    }
    save_json(out, payload); print(json.dumps(payload, indent=2))
    if reasons: raise SystemExit(2)


if __name__ == "__main__": main()
