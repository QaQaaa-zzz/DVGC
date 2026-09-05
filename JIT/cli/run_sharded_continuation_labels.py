#!/usr/bin/env python3
"""Run memory-bounded shards of one logical frozen continuation-label protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax

from jit_dvgc.checkpoint import load_checkpoint
from jit_dvgc.config import file_sha256
from jit_dvgc.continuation import lock_negative_acceptance_bank
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.training import (
    build_unified_formal_environment,
    checkpoint_identity,
    load_frozen_unified_manifest,
)
from jit_dvgc.unified_continuation_shards import (
    label_unified_continuation_shard,
    merge_unified_continuation_shards,
)


REPAIR_ACCEPTANCE_SCHEMA = "jit_repair_acceptance_boundary_acquisition_v1"
WARP_MEMORY_MAINTENANCE_STEP_INTERVAL = 64
WARP_MEMORY_TELEMETRY_STEP_INTERVAL = 4096


def _read_json(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _canonical_sha256(payload) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()


def _load_predeclaration(path: Path) -> dict:
    payload = _read_json(path)
    if payload.get("schema") != REPAIR_ACCEPTANCE_SCHEMA:
        raise ValueError("unsupported repair acceptance predeclaration schema")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("repair acceptance protocol missing")
    if protocol.get("status") != "predeclared_before_repair_training":
        raise ValueError("repair acceptance was not predeclared before replacement training")
    if _canonical_sha256(protocol) != payload.get("expected_protocol_sha256"):
        raise ValueError("repair acceptance predeclaration SHA-256 drift")
    labeling = protocol.get("labeling")
    bank_lock = protocol.get("bank_lock")
    if not isinstance(labeling, dict) or not isinstance(bank_lock, dict):
        raise ValueError("repair acceptance labeling/bank-lock contract missing")
    if labeling.get("policy_mode") != "deterministic":
        raise ValueError("repair acceptance labeling must be deterministic")
    if labeling.get("baseline_policy_only") is not True:
        raise ValueError("repair acceptance labeling must use baseline only")
    if bank_lock.get("candidate_training_must_not_start_before_lock") is not True:
        raise ValueError("candidate training must remain blocked until bank lock")
    return payload


def _validate_acquisition_binding(
    predeclared: dict, predeclared_path: Path, catalog_path: Path
) -> None:
    root = Path(catalog_path).parent
    copied = _read_json(root / "predeclaration.json")
    if copied != predeclared:
        raise ValueError("acquisition copied predeclaration semantic drift")
    audit = _read_json(root / "anchor_audit.json")
    if audit.get("predeclaration_file_sha256") != file_sha256(predeclared_path):
        raise ValueError("acquisition anchor-audit predeclaration SHA drift")
    if (
        audit.get("predeclared_protocol_sha256")
        != predeclared["expected_protocol_sha256"]
    ):
        raise ValueError("acquisition anchor-audit logical protocol drift")


def _build_memory_stable_step(env, *, device: str = "cuda:0"):
    try:
        import warp as wp
    except Exception as exc:
        raise RuntimeError("Warp runtime is required for GPU continuation labeling") from exc

    compiled_step = jax.jit(env.step)
    mempool_supported = bool(wp.is_mempool_supported(device))
    mempool_enabled = bool(wp.is_mempool_enabled(device)) if mempool_supported else False
    if mempool_enabled:
        wp.set_mempool_release_threshold(device, 0)

    print(
        "warp_memory_maintenance=enabled "
        f"device={device} step_interval={WARP_MEMORY_MAINTENANCE_STEP_INTERVAL} "
        f"mempool_supported={mempool_supported} mempool_enabled={mempool_enabled}"
    )
    step_count = 0

    def step_with_memory_maintenance(state, action):
        nonlocal step_count
        next_state = compiled_step(state, action)
        step_count += 1
        if step_count % WARP_MEMORY_MAINTENANCE_STEP_INTERVAL == 0:
            jax.block_until_ready(next_state)
            wp.synchronize_device(device)
        if step_count % WARP_MEMORY_TELEMETRY_STEP_INTERVAL == 0:
            fields = [f"steps={step_count}"]
            if mempool_enabled:
                fields.extend(
                    [
                        f"mempool_current={wp.get_mempool_used_mem_current(device)}",
                        f"mempool_high={wp.get_mempool_used_mem_high(device)}",
                    ]
                )
            fields.append(f"device_free={wp.get_device(device).free_memory}")
            print("warp_memory " + " ".join(fields))
        return next_state

    return step_with_memory_maintenance


def _run_shard(args) -> int:
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    predeclared = _load_predeclaration(args.predeclaration)
    _validate_acquisition_binding(predeclared, args.predeclaration, args.catalog)
    repair = predeclared["protocol"]
    if str(args.frozen_policy) != str(repair["baseline_frozen_policy"]):
        raise ValueError("sharded labeling baseline frozen-policy path drift")
    max_ticks = int(repair["labeling"]["max_ticks"])
    protocol_seed = int(repair["labeling"]["protocol_seed"])

    frozen = load_frozen_unified_manifest(args.frozen_policy)
    record = frozen["policy"]
    if record["name"] != repair["baseline_policy_name"]:
        raise ValueError("sharded labeling baseline policy-name drift")
    if record["actor_sha256"] != repair["baseline_actor_sha256"]:
        raise ValueError("sharded labeling baseline actor drift")
    if record["payload_sha256"] != repair["baseline_payload_sha256"]:
        raise ValueError("sharded labeling baseline payload drift")
    frozen_sha = file_sha256(args.frozen_policy)

    if jax.default_backend() != "gpu":
        raise RuntimeError("sharded continuation labeling requires the visible JAX GPU")
    config, _artifact, env = build_unified_formal_environment(
        Path(record["formal_config"])
    )
    if config.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("sharded continuation formal config mismatch")
    if env._bundle.xml_sha256 != record["xml_sha256"]:
        raise ValueError("sharded continuation runtime XML mismatch")
    if max_ticks != int(config.ppo.episode_horizon):
        raise ValueError("sharded continuation horizon drift")

    payload = load_checkpoint(
        Path(record["checkpoint"]),
        expected=checkpoint_identity(config, env),
    )
    if int(payload.training_transitions) != int(record["source_training_transitions"]):
        raise ValueError("sharded continuation checkpoint transition mismatch")
    if file_sha256(Path(record["checkpoint"]) / "payload.pkl") != record["payload_sha256"]:
        raise ValueError("sharded continuation checkpoint payload SHA-256 mismatch")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
    step_fn = _build_memory_stable_step(env)

    report = label_unified_continuation_shard(
        args.catalog,
        args.output_dir,
        env=env,
        policy=policy,
        policy_record=record,
        frozen_manifest_sha256=frozen_sha,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        max_ticks=max_ticks,
        protocol_seed=protocol_seed,
        compiled_step_fn=step_fn,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _merge(args) -> int:
    if args.output_dir.exists():
        raise FileExistsError(f"output already exists: {args.output_dir}")
    predeclared = _load_predeclaration(args.predeclaration)
    _validate_acquisition_binding(predeclared, args.predeclaration, args.catalog)
    report = merge_unified_continuation_shards(
        args.catalog, args.shard_dir, args.output_dir
    )
    repair = predeclared["protocol"]
    labeling = repair["labeling"]
    if report["policy_name"] != repair["baseline_policy_name"]:
        raise ValueError("merged labeling baseline policy-name drift")
    if report["policy_actor_sha256"] != repair["baseline_actor_sha256"]:
        raise ValueError("merged labeling baseline actor drift")
    if report["policy_payload_sha256"] != repair["baseline_payload_sha256"]:
        raise ValueError("merged labeling baseline payload drift")

    merged_protocol = _read_json(args.output_dir / "protocol.json")
    if int(merged_protocol["protocol_seed"]) != int(labeling["protocol_seed"]):
        raise ValueError("merged labeling protocol seed drift")
    if int(merged_protocol["max_ticks_per_candidate"]) != int(labeling["max_ticks"]):
        raise ValueError("merged labeling horizon drift")

    (args.output_dir / "acceptance_predeclaration.json").write_text(
        json.dumps(predeclared, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    identity = {
        "schema": "jit_repair_acceptance_label_binding_v1",
        "predeclaration": str(args.predeclaration),
        "predeclaration_file_sha256": file_sha256(args.predeclaration),
        "predeclared_protocol_sha256": predeclared["expected_protocol_sha256"],
        "baseline_policy_name": report["policy_name"],
        "baseline_actor_sha256": report["policy_actor_sha256"],
        "baseline_payload_sha256": report["policy_payload_sha256"],
        "max_ticks": int(labeling["max_ticks"]),
        "protocol_seed": int(labeling["protocol_seed"]),
        "execution": "independent_contiguous_shards_merged_by_global_candidate_index",
        "training_transitions": 0,
        "validation_data_used": False,
        "test_data_used": False,
        "final_evaluation_data_used": False,
    }
    (args.output_dir / "acceptance_predeclaration_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    bank_lock = repair["bank_lock"]
    bank = lock_negative_acceptance_bank(
        args.output_dir / "labels.json",
        args.catalog,
        args.output_dir / "acceptance_bank.json",
        target_tube_path=Path(bank_lock["target_tube"]),
        expected_target_tube_manifest_sha256=str(
            bank_lock["target_tube_manifest_sha256"]
        ),
        predeclaration_file_sha256=file_sha256(args.predeclaration),
        predeclared_protocol_sha256=str(predeclared["expected_protocol_sha256"]),
        minimum_negative_states_per_phase=int(
            bank_lock["minimum_negative_states_per_phase"]
        ),
        minimum_negative_parent_groups_per_phase=int(
            bank_lock["minimum_negative_parent_groups_per_phase"]
        ),
    )
    print(
        json.dumps(
            {
                "labeling": report,
                "acceptance_bank": {
                    key: value for key, value in bank.items() if key != "entries"
                },
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="label one execution shard")
    run.add_argument("--frozen-policy", type=Path, required=True)
    run.add_argument("--catalog", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--predeclaration", type=Path, required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--shard-count", type=int, required=True)

    merge = sub.add_parser("merge", help="strictly merge all shards and lock bank")
    merge.add_argument("--catalog", type=Path, required=True)
    merge.add_argument("--output-dir", type=Path, required=True)
    merge.add_argument("--predeclaration", type=Path, required=True)
    merge.add_argument("--shard-dir", type=Path, action="append", required=True)

    args = parser.parse_args()
    return _run_shard(args) if args.command == "run" else _merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
