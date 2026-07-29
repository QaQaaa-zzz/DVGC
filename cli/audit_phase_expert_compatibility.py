"""Audit frozen expert architecture/provenance before shared-policy consolidation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from dvgc.config import file_sha256
from dvgc.policy import load_bundle
from dvgc.runtime import save_json


def parameter_signature(params) -> list[dict]:
    return [{"shape": list(np.asarray(leaf).shape), "dtype": str(np.asarray(leaf).dtype)}
            for leaf in jax.tree.leaves(params)]


def compatible(rows: list[dict]) -> bool:
    fields = ("policy_network_version", "action_mapping_version", "xml_sha256",
              "actor_history_steps", "parameter_signature")
    return bool(rows) and all(all(row[field] == rows[0][field] for row in rows[1:])
                              for field in fields)


def parse_assignment(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("assignment must be stage=value")
    stage, target = value.split("=", 1)
    if not stage or not target:
        raise ValueError("assignment must have non-empty stage and value")
    return stage, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, help="stage=policy_bundle")
    parser.add_argument("--nonparametric-controller", action="append", default=[], help="stage=descriptor")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("refusing overwrite expert compatibility audit")
    rows = []
    for assignment in args.policy:
        stage, raw_path = parse_assignment(assignment)
        path = Path(raw_path)
        params, cfg, manifest = load_bundle(path, verify_files=True)
        rows.append({
            "stage": stage, "policy_path": str(path),
            "params_sha256": file_sha256(path / "params.pkl"),
            "manifest_sha256": file_sha256(path / "manifest.json"),
            "policy_version": manifest["policy_version"],
            "policy_network_version": manifest["policy_network_version"],
            "action_mapping_version": manifest["action_mapping_version"],
            "xml_sha256": manifest["xml_sha256"],
            "actor_history_steps": int(cfg["actor_history_steps"]),
            "parameter_signature": parameter_signature(params),
        })
    nonparametric = [dict(zip(("stage", "descriptor"), parse_assignment(value)))
                     for value in args.nonparametric_controller]
    passed = compatible(rows) and any(row["stage"] == "apex" for row in nonparametric)
    payload = {
        "status": "PASS" if passed else "FAIL",
        "artifact_role": "phase_expert_shared_actor_compatibility_audit",
        "parameterized_experts": rows,
        "nonparametric_controllers": nonparametric,
        "apex_teacher_requires_action_extraction": True,
        "architecture_compatible": compatible(rows),
        "same_xml": len({row["xml_sha256"] for row in rows}) == 1,
        "same_action_mapping": len({row["action_mapping_version"] for row in rows}) == 1,
        "shared_actor_distillation_authorized": passed,
        "PPO_authorization": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, payload)
    print(json.dumps({key: value for key, value in payload.items()
                      if key != "parameterized_experts"}, indent=2))
    if not passed:
        raise SystemExit("phase expert architecture/provenance compatibility failed")


if __name__ == "__main__":
    main()
