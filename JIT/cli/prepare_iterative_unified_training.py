#!/usr/bin/env python3
"""Generate the fixed repair02-derived training config for pi_(k+1)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.config import file_sha256
from jit_dvgc.soft_tube import load_soft_tube
from jit_dvgc.unified_formal import load_unified_formal_config


DEFAULT_TEMPLATE = Path("JIT/configs/pi_unified_iter1_tube1_core_replay75_natural10.json")


def _read(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soft-tube", type=Path, required=True)
    parser.add_argument("--tube-rsi-smoke-report", type=Path, required=True)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    args = parser.parse_args()

    if args.output_config.exists():
        raise FileExistsError(f"training config already exists: {args.output_config}")
    template = _read(args.template)
    # Prove that the source template still means the exact recipe we selected
    # empirically today.  We intentionally do not inherit its Tube/run claims.
    loaded_template = load_unified_formal_config(args.template)
    if template.get("tube_sampling") != {
        "schema": "jit_tube_rsi_core_replay_v1",
        "selection": "phase_then_source_then_entry",
        "core_probability": 0.75,
        "expansion_probability": 0.25,
        "core_within_source": "uniform",
        "expansion_within_source": "value_weighted",
        "source_core_definition": "first_core_retained_count_entries",
    }:
        raise ValueError("automatic iteration template lost the repair02 75/25 replay recipe")
    if loaded_template.reset_mixture.natural_reset_probability != 0.1:
        raise ValueError("automatic iteration template natural reset drift")

    tube = load_soft_tube(args.soft_tube)
    iteration = int(tube.manifest.get("iteration", -1))
    source_iteration = int(tube.manifest.get("source_iteration", -1))
    if iteration < 2 or source_iteration != iteration - 1:
        raise ValueError("automatic training expects Tube_(k+1), k>=1")
    if int(tube.manifest.get("core_retained_count", 0)) != int(
        tube.manifest.get("source_tube_entry_count", -1)
    ):
        raise ValueError("automatic training Tube core partition drift")
    if int(tube.manifest.get("expansion_count", 0)) <= 0:
        raise ValueError("automatic training Tube has no new expansion")

    smoke = _read(args.tube_rsi_smoke_report)
    if smoke.get("schema") != "jit_tube_rsi_smoke_v1" or smoke.get("status") != "completed":
        raise ValueError("Tube-RSI smoke is not completed")
    if smoke.get("tube_rsi_smoke") != "GO":
        raise ValueError("Tube-RSI smoke is not GO")
    if smoke.get("soft_tube_manifest_sha256") != tube.manifest["manifest_sha256"]:
        raise ValueError("Tube-RSI smoke Tube identity drift")
    if smoke.get("test_data_used") is not False or smoke.get("validation_data_used") is not False:
        raise ValueError("Tube-RSI smoke touched forbidden data")

    payload = dict(template)
    payload["inputs"] = {
        **dict(template["inputs"]),
        "soft_tube_path": str(args.soft_tube),
        "soft_tube_manifest_sha256": str(tube.manifest["manifest_sha256"]),
        "tube_rsi_smoke_report": str(args.tube_rsi_smoke_report),
        "tube_rsi_smoke_report_sha256": file_sha256(args.tube_rsi_smoke_report),
    }
    payload["run_declaration"] = {
        "run_id": str(args.run_id),
        "output_dir": f"JIT/runs/pi_unified/{args.run_id}",
        "purpose": "automatic_core_retaining_envelope_iteration_fixed_repair02_recipe",
        "status": "predeclared_not_started",
    }
    payload["claim_boundary"] = {
        "formal_method_stage_training": True,
        "iteration": iteration,
        "policy_name": f"pi_{iteration}",
        "source_policy_name": f"pi_{source_iteration}",
        "source_tube_iteration": iteration,
        "candidate_revision": "automatic_fixed_repair02_recipe",
        "fixed_training_recipe": "fresh_actor_critic_optimizer; Tube90/natural10; within_Tube old_core75/new_expansion25",
        "soft_tube_support_predeclared": True,
        "fresh_initialization_unchanged": True,
        "ppo_hyperparameters_unchanged": True,
        "reward_physics_action_semantics_unchanged": True,
        "reset_probabilities_unchanged": True,
        "tube_manifest_sha256": str(tube.manifest["manifest_sha256"]),
        "core_replay_contract": "phase_50_50_then_source_core75_expansion25_core_uniform_expansion_value_weighted",
        "core_preservation_claim": False,
        "new_expansion_retention_claim": False,
        "final_policy_claim": False,
        "certified_safe_tube_claim": False,
        "jce_jel_claim": False,
        "test_data_used": False,
        "validation_data_used": False,
    }
    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Validate the generated object through the canonical loader before returning.
    loaded = load_unified_formal_config(args.output_config)
    if loaded.soft_tube_manifest_sha256 != tube.manifest["manifest_sha256"]:
        raise ValueError("generated automatic training config failed identity roundtrip")
    result = {
        "schema": "jit_iterative_unified_training_config_preparation_v1",
        "status": "completed",
        "iteration": iteration,
        "policy_name": f"pi_{iteration}",
        "run_id": str(args.run_id),
        "config": str(args.output_config),
        "config_sha256": loaded.config_sha256,
        "soft_tube_manifest_sha256": tube.manifest["manifest_sha256"],
        "tube_rsi_smoke_report_sha256": file_sha256(args.tube_rsi_smoke_report),
        "training_transitions": 0,
        "test_data_used": False,
        "validation_data_used": False,
    }
    result_path = args.output_config.with_suffix(args.output_config.suffix + ".prepared.json")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
