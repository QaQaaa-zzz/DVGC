#!/usr/bin/env python3
"""Collect TRAIN-only real-dynamics frontier candidates under frozen pi_k."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax

from jit_dvgc.checkpoint import load_checkpoint
from jit_dvgc.config import file_sha256
from jit_dvgc.constants import ACTION_ORDER
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.tube import load_soft_tube
from jit_dvgc.acquisition import (
    DEFAULT_ANCHORS_PER_PHASE,
    DEFAULT_FRONTIER_SCORE_CEILING,
    DEFAULT_UNIFIED_BOUNDARY_DURATIONS,
    DEFAULT_UNIFIED_BOUNDARY_STRENGTHS,
    collect_unified_boundary_candidates,
    select_disjoint_tube_boundary_anchors,
    select_tube_boundary_anchors,
)
from jit_dvgc.repair_acceptance import (
    REPAIR_ACCEPTANCE_SCHEMA,
    consumed_gate_exclusions,
)
from jit_dvgc.training import (
    build_unified_formal_environment,
    checkpoint_identity,
    load_frozen_unified_manifest,
    load_unified_formal_config,
)


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _load_repair_predeclaration(path: Path) -> dict:
    payload = _read_json(path)
    if payload.get("schema") != REPAIR_ACCEPTANCE_SCHEMA:
        raise ValueError("unsupported repair acceptance acquisition schema")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("repair acceptance acquisition protocol missing")
    if protocol.get("status") != "predeclared_before_repair_training":
        raise ValueError("repair acceptance bank must be predeclared before repair training")
    if _canonical_sha256(protocol) != payload.get("expected_protocol_sha256"):
        raise ValueError("repair acceptance acquisition protocol SHA-256 drift")

    source_iteration = int(protocol.get("source_iteration", -1))
    candidate_iteration = int(protocol.get("candidate_iteration", -1))
    if source_iteration < 0 or candidate_iteration != source_iteration + 1:
        raise ValueError(
            "repair acceptance acquisition must declare k -> k+1 iteration progression"
        )

    acquisition = protocol.get("acquisition")
    if not isinstance(acquisition, dict):
        raise ValueError("repair acceptance acquisition settings missing")
    if int(acquisition.get("anchors_per_phase", 0)) <= 0:
        raise ValueError("repair acceptance anchors_per_phase invalid")
    minimum = int(acquisition.get("minimum_anchors_per_phase", 0))
    if minimum <= 0 or minimum > int(acquisition["anchors_per_phase"]):
        raise ValueError("repair acceptance minimum anchor count invalid")
    active_dimensions = int(acquisition.get("active_action_dimensions", 1))
    action_names = tuple(str(value) for value in acquisition.get("action_names", ()))
    if active_dimensions <= 0 or active_dimensions > len(action_names):
        raise ValueError("repair acceptance active_action_dimensions invalid")

    isolation = protocol.get("isolation")
    if isolation != {
        "exclude_consumed_boundary_states": True,
        "exclude_consumed_boundary_parent_groups_by_phase": True,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }:
        raise ValueError("repair acceptance isolation contract drift")
    consumed_gate_exclusions(protocol)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predeclaration",
        type=Path,
        help=(
            "predeclared repair-acceptance acquisition contract; when supplied, "
            "scientific acquisition knobs and all consumed-gate exclusions come from it"
        ),
    )
    parser.add_argument("--anchors-per-phase", type=int, default=DEFAULT_ANCHORS_PER_PHASE)
    parser.add_argument(
        "--frontier-score-ceiling",
        type=float,
        default=DEFAULT_FRONTIER_SCORE_CEILING,
        help=(
            "maximum bootstrap continuation score eligible for frontier probing; "
            "default 0.5 is the fixed binary decision boundary"
        ),
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=list(DEFAULT_UNIFIED_BOUNDARY_STRENGTHS),
    )
    parser.add_argument(
        "--durations",
        type=int,
        nargs="+",
        default=list(DEFAULT_UNIFIED_BOUNDARY_DURATIONS),
    )
    parser.add_argument(
        "--action-names",
        nargs="+",
        choices=list(ACTION_ORDER),
        default=list(ACTION_ORDER),
    )
    parser.add_argument(
        "--signs",
        type=int,
        nargs="+",
        choices=(-1, 1),
        default=[-1, 1],
    )
    parser.add_argument(
        "--active-action-dimensions",
        type=int,
        default=1,
        help=(
            "number of simultaneously perturbed action dimensions; 1 preserves the "
            "historical one-axis action-basis protocol"
        ),
    )
    parser.add_argument("--protocol-seed", type=int, default=9_510_001)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="select Tube frontier anchors without constructing MJX or spending interactions",
    )
    args = parser.parse_args()

    predeclared = None
    predeclared_sha = None
    minimum_anchors_per_phase = 1
    consumed_identity = None
    consumed_states: set[str] = set()
    consumed_groups: dict[str, set[str]] = {"upstream": set(), "downstream": set()}
    if args.predeclaration is not None:
        predeclared = _load_repair_predeclaration(args.predeclaration)
        predeclared_sha = file_sha256(args.predeclaration)
        protocol = predeclared["protocol"]
        if str(args.frozen_policy) != str(protocol["baseline_frozen_policy"]):
            raise ValueError("repair acceptance frozen-policy path drift")
        acq = protocol["acquisition"]
        args.anchors_per_phase = int(acq["anchors_per_phase"])
        minimum_anchors_per_phase = int(acq["minimum_anchors_per_phase"])
        args.frontier_score_ceiling = float(acq["frontier_score_ceiling"])
        args.strengths = [float(x) for x in acq["strengths"]]
        args.durations = [int(x) for x in acq["durations"]]
        args.action_names = [str(x) for x in acq["action_names"]]
        args.signs = [int(x) for x in acq["signs"]]
        args.active_action_dimensions = int(acq.get("active_action_dimensions", 1))
        args.protocol_seed = int(acq["protocol_seed"])
        consumed_states, consumed_groups, consumed_identity = consumed_gate_exclusions(
            protocol
        )

    if args.anchors_per_phase <= 0:
        parser.error("--anchors-per-phase must be positive")
    if not 0.0 <= args.frontier_score_ceiling <= 1.0:
        parser.error("--frontier-score-ceiling must lie in [0, 1]")
    if len(set(args.action_names)) != len(args.action_names):
        parser.error("--action-names must be unique")
    if len(set(args.signs)) != len(args.signs):
        parser.error("--signs must be unique")
    if not 1 <= args.active_action_dimensions <= len(args.action_names):
        parser.error("--active-action-dimensions must lie in [1, len(action_names)]")

    frozen = load_frozen_unified_manifest(args.frozen_policy)
    record = frozen["policy"]
    config = load_unified_formal_config(Path(record["formal_config"]))
    if config.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("frozen unified policy/formal config SHA-256 mismatch")
    artifact = load_soft_tube(Path(config.soft_tube_path))
    if artifact.manifest["manifest_sha256"] != config.soft_tube_manifest_sha256:
        raise ValueError("formal config/source Tube SHA-256 mismatch")

    if predeclared is not None:
        protocol = predeclared["protocol"]
        if int(record["iteration"]) != int(protocol["source_iteration"]):
            raise ValueError("repair acceptance baseline policy iteration drift")
        if record["name"] != protocol["baseline_policy_name"]:
            raise ValueError("repair acceptance baseline policy name drift")
        if record["actor_sha256"] != protocol["baseline_actor_sha256"]:
            raise ValueError("repair acceptance baseline actor drift")
        if record["payload_sha256"] != protocol["baseline_payload_sha256"]:
            raise ValueError("repair acceptance baseline payload drift")
        if artifact.manifest["manifest_sha256"] != protocol["source_tube_manifest_sha256"]:
            raise ValueError("repair acceptance source Tube drift")
        anchors, audit = select_disjoint_tube_boundary_anchors(
            artifact,
            max_per_phase=args.anchors_per_phase,
            minimum_per_phase=minimum_anchors_per_phase,
            frontier_score_ceiling=args.frontier_score_ceiling,
            excluded_state_sha256=tuple(sorted(consumed_states)),
            excluded_parent_groups={
                phase: tuple(sorted(consumed_groups[phase]))
                for phase in ("upstream", "downstream")
            },
        )
        audit = {
            **audit,
            "predeclaration": str(args.predeclaration),
            "predeclaration_file_sha256": predeclared_sha,
            "predeclared_protocol_sha256": predeclared["expected_protocol_sha256"],
            "consumed_gates": consumed_identity,
            "claim_boundary": {
                "fresh_nonfinal_acceptance_candidate_generation": True,
                "replacement_candidate_not_yet_trained": True,
                "not_model_training_data": True,
                "not_tube_construction_data": True,
                "jce_jel_claim": False,
            },
        }
    else:
        anchors, audit = select_tube_boundary_anchors(
            artifact,
            max_per_phase=args.anchors_per_phase,
            frontier_score_ceiling=args.frontier_score_ceiling,
        )

    if args.audit_only:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "anchor_audit.json", audit)
        if predeclared is not None:
            _write_json(output / "predeclaration.json", predeclared)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0

    if jax.default_backend() != "gpu":
        raise RuntimeError("unified boundary acquisition requires the visible JAX GPU")
    runtime_config, runtime_artifact, env = build_unified_formal_environment(
        Path(record["formal_config"])
    )
    if runtime_config.config_sha256 != config.config_sha256:
        raise ValueError("unified boundary runtime formal config drift")
    if runtime_artifact.manifest["manifest_sha256"] != artifact.manifest["manifest_sha256"]:
        raise ValueError("unified boundary runtime source Tube drift")
    if env._bundle.xml_sha256 != record["xml_sha256"]:
        raise ValueError("frozen unified policy/runtime XML mismatch")

    payload = load_checkpoint(
        Path(record["checkpoint"]),
        expected=checkpoint_identity(runtime_config, env),
    )
    if int(payload.training_transitions) != int(record["source_training_transitions"]):
        raise ValueError("unified boundary checkpoint transition drift")
    policy = make_checkpoint_policy(env, payload, deterministic=True)

    try:
        report = collect_unified_boundary_candidates(
            anchors,
            args.output_dir,
            env=env,
            policy=jax.jit(policy),
            policy_record=record,
            frozen_manifest_sha256=file_sha256(args.frozen_policy),
            protocol_seed=args.protocol_seed,
            frontier_score_ceiling=args.frontier_score_ceiling,
            strengths=tuple(args.strengths),
            durations=tuple(args.durations),
            action_names=tuple(args.action_names),
            signs=tuple(args.signs),
            active_action_dimensions=args.active_action_dimensions,
        )
        _write_json(Path(args.output_dir) / "anchor_audit.json", audit)
        if predeclared is not None:
            _write_json(Path(args.output_dir) / "predeclaration.json", predeclared)
        print(
            json.dumps(
                {
                    "anchor_audit": audit,
                    "collection": {
                        key: value for key, value in report.items() if key != "entries"
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except BaseException as exc:
        output = Path(args.output_dir)
        if output.exists():
            _write_json(
                output / "failure.json",
                {
                    "schema": "jit_unified_boundary_failure_v1",
                    "status": "engineering_error",
                    "frozen_policy": str(args.frozen_policy),
                    "frozen_policy_sha256": file_sha256(args.frozen_policy),
                    "source_tube_manifest_sha256": artifact.manifest["manifest_sha256"],
                    "policy_iteration": int(record["iteration"]),
                    "policy_actor_sha256": str(record["actor_sha256"]),
                    "policy_payload_sha256": str(record["payload_sha256"]),
                    "frontier_score_ceiling": float(args.frontier_score_ceiling),
                    "active_action_dimensions": int(args.active_action_dimensions),
                    "predeclaration": (
                        str(args.predeclaration) if args.predeclaration else None
                    ),
                    "predeclaration_file_sha256": predeclared_sha,
                    "training_transitions": 0,
                    "test_data_used": False,
                    "validation_data_used": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
