#!/usr/bin/env python3
"""Label real-dynamics envelope candidates under one frozen unified policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from jit_dvgc.checkpoint import load_checkpoint
from jit_dvgc.config import file_sha256
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.continuation.labels import (
    DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS,
    DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED,
    label_unified_continuations,
    validate_candidate_snapshot,
    validate_unified_boundary_catalog,
)
from jit_dvgc.snapshots.unified import load_unified_envelope_snapshot
from jit_dvgc.training.formal import build_unified_formal_environment
from jit_dvgc.training.freeze import load_frozen_unified_manifest
from jit_dvgc.training.unified import checkpoint_identity


def _read_json(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-ticks",
        type=int,
        default=DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS,
    )
    parser.add_argument(
        "--protocol-seed",
        type=int,
        default=DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED,
    )
    args = parser.parse_args()

    if args.max_ticks <= 0:
        parser.error("--max-ticks must be positive")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")

    frozen = load_frozen_unified_manifest(args.frozen_policy)
    record = frozen["policy"]
    if record["policy_role"] != "envelope_expansion_authority":
        raise ValueError("frozen unified policy is not an expansion authority")
    frozen_sha = file_sha256(args.frozen_policy)

    # Close the entire disk/provenance bank before spending the first rollout
    # interaction. A late corrupt snapshot must not invalidate a partially
    # consumed labeling budget.
    catalog = _read_json(args.catalog)
    rows = validate_unified_boundary_catalog(
        catalog,
        policy_record=record,
        frozen_manifest_sha256=frozen_sha,
    )
    acquisition_protocol = _read_json(Path(args.catalog).parent / "protocol.json")
    if acquisition_protocol.get("protocol_sha256") != catalog.get("protocol_sha256"):
        raise ValueError("unified acquisition protocol/catalog SHA mismatch")
    for row in rows:
        snapshot_path = (
            Path(args.catalog).parent / str(row["source_bank"]) / str(row["snapshot"])
        )
        snapshot = load_unified_envelope_snapshot(snapshot_path)
        validate_candidate_snapshot(snapshot, row, policy_record=record)
    print(f"snapshot_preflight=GO candidates={len(rows)}")

    if jax.default_backend() != "gpu":
        raise RuntimeError("unified continuation labeling requires the visible JAX GPU")
    config, _artifact, env = build_unified_formal_environment(
        Path(record["formal_config"])
    )
    if config.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("unified continuation formal config mismatch")
    if env._bundle.xml_sha256 != record["xml_sha256"]:
        raise ValueError("unified continuation runtime XML mismatch")
    if int(args.max_ticks) != int(config.ppo.episode_horizon):
        raise ValueError(
            "unified continuation labeling must use the frozen policy episode horizon"
        )

    payload = load_checkpoint(
        Path(record["checkpoint"]),
        expected=checkpoint_identity(config, env),
    )
    if int(payload.training_transitions) != int(record["source_training_transitions"]):
        raise ValueError("unified continuation checkpoint transition mismatch")
    if file_sha256(Path(record["checkpoint"]) / "payload.pkl") != record["payload_sha256"]:
        raise ValueError("unified continuation checkpoint payload SHA-256 mismatch")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))

    report = label_unified_continuations(
        args.catalog,
        args.output_dir,
        env=env,
        policy=policy,
        policy_record=record,
        frozen_manifest_sha256=frozen_sha,
        max_ticks=args.max_ticks,
        protocol_seed=args.protocol_seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
