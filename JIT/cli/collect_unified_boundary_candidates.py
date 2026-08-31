#!/usr/bin/env python3
"""Collect TRAIN-only real-dynamics frontier candidates under frozen pi_k."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.checkpoint import load_checkpoint
from jit_dvgc.config import file_sha256
from jit_dvgc.constants import ACTION_ORDER
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.soft_tube import load_soft_tube
from jit_dvgc.unified_boundary import (
    DEFAULT_ANCHORS_PER_PHASE,
    DEFAULT_UNIFIED_BOUNDARY_DURATIONS,
    DEFAULT_UNIFIED_BOUNDARY_STRENGTHS,
    collect_unified_boundary_candidates,
    select_tube_boundary_anchors,
)
from jit_dvgc.unified_formal import (
    build_unified_formal_environment,
    load_unified_formal_config,
)
from jit_dvgc.unified_policy_freeze import load_frozen_unified_manifest
from jit_dvgc.unified_training import checkpoint_identity


def _write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchors-per-phase", type=int, default=DEFAULT_ANCHORS_PER_PHASE)
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
    parser.add_argument("--protocol-seed", type=int, default=9_510_001)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="select Tube frontier anchors without constructing MJX or spending interactions",
    )
    args = parser.parse_args()

    if args.anchors_per_phase <= 0:
        parser.error("--anchors-per-phase must be positive")
    if len(set(args.action_names)) != len(args.action_names):
        parser.error("--action-names must be unique")
    if len(set(args.signs)) != len(args.signs):
        parser.error("--signs must be unique")

    frozen = load_frozen_unified_manifest(args.frozen_policy)
    record = frozen["policy"]
    config = load_unified_formal_config(Path(record["formal_config"]))
    if config.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("frozen unified policy/formal config SHA-256 mismatch")
    artifact = load_soft_tube(Path(config.soft_tube_path))
    if artifact.manifest["manifest_sha256"] != config.soft_tube_manifest_sha256:
        raise ValueError("formal config/source Tube SHA-256 mismatch")

    anchors, audit = select_tube_boundary_anchors(
        artifact,
        max_per_phase=args.anchors_per_phase,
    )
    if args.audit_only:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        _write_json(output / "anchor_audit.json", audit)
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
        raise ValueError("frozen unified policy checkpoint transition drift")
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
            strengths=tuple(args.strengths),
            durations=tuple(args.durations),
            action_names=tuple(args.action_names),
            signs=tuple(args.signs),
        )
        _write_json(Path(args.output_dir) / "anchor_audit.json", audit)
        print(
            json.dumps(
                {
                    "anchor_audit": audit,
                    "collection": {key: value for key, value in report.items() if key != "entries"},
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
                    "training_transitions": 0,
                    "test_data_used": False,
                    "validation_data_used": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
