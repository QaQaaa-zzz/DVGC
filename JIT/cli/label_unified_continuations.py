#!/usr/bin/env python3
"""Label real-dynamics envelope candidates under one frozen unified policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax

from jit_dvgc.checkpoint import load_checkpoint
from jit_dvgc.config import file_sha256
from jit_dvgc.ppo import make_checkpoint_policy
from jit_dvgc.continuation import (
    DEFAULT_UNIFIED_CONTINUATION_MAX_TICKS,
    DEFAULT_UNIFIED_CONTINUATION_PROTOCOL_SEED,
    label_unified_continuations,
    lock_negative_acceptance_bank,
    validate_candidate_snapshot,
    validate_unified_boundary_catalog,
)
from jit_dvgc.snapshots import load_unified_envelope_snapshot
from jit_dvgc.training import (
    build_unified_formal_environment,
    checkpoint_identity,
    load_frozen_unified_manifest,
)


REPAIR_ACCEPTANCE_SCHEMA = "jit_repair_acceptance_boundary_acquisition_v1"


def _read_json(path: Path):
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
        raise ValueError("unsupported repair acceptance predeclaration schema")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("repair acceptance predeclaration protocol missing")
    if protocol.get("status") != "predeclared_before_repair_training":
        raise ValueError("repair acceptance labeling was not predeclared before training")
    if _canonical_sha256(protocol) != payload.get("expected_protocol_sha256"):
        raise ValueError("repair acceptance predeclaration SHA-256 drift")

    labeling = protocol.get("labeling")
    if not isinstance(labeling, dict):
        raise ValueError("repair acceptance baseline labeling contract missing")
    if labeling.get("policy_mode") != "deterministic":
        raise ValueError("repair acceptance baseline labeling must be deterministic")
    if int(labeling.get("max_ticks", 0)) != 400:
        raise ValueError("repair acceptance baseline labeling horizon drift")
    if int(labeling.get("protocol_seed", -1)) < 0:
        raise ValueError("repair acceptance baseline labeling seed is invalid")
    if labeling.get("baseline_policy_only") is not True:
        raise ValueError("repair acceptance labeling must use baseline policy only")

    bank_lock = protocol.get("bank_lock")
    if not isinstance(bank_lock, dict):
        raise ValueError("repair acceptance bank-lock contract missing")
    if bank_lock.get("selection") != "all_baseline_continuation_negative_candidates":
        raise ValueError("repair acceptance bank selection drift")
    if bank_lock.get("exclude_target_tube_states") is not True:
        raise ValueError("repair acceptance bank must exclude target Tube states")
    if not str(bank_lock.get("target_tube", "")):
        raise ValueError("repair acceptance target Tube path missing")
    if len(str(bank_lock.get("target_tube_manifest_sha256", ""))) != 64:
        raise ValueError("repair acceptance target Tube manifest SHA-256 invalid")
    if bank_lock.get("physical_state_unique") is not True:
        raise ValueError("repair acceptance bank must require physical-state uniqueness")
    if bank_lock.get("require_both_phases") is not True:
        raise ValueError("repair acceptance bank must require both phases")
    if int(bank_lock.get("minimum_negative_states_per_phase", 0)) <= 0:
        raise ValueError("repair acceptance negative-state minimum invalid")
    if int(bank_lock.get("minimum_negative_parent_groups_per_phase", 0)) <= 0:
        raise ValueError("repair acceptance negative-parent minimum invalid")
    if bank_lock.get("candidate_training_must_not_start_before_lock") is not True:
        raise ValueError("repair candidate training must be blocked until bank lock")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-policy", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--predeclaration",
        type=Path,
        help=(
            "predeclared fresh repair-acceptance contract; when supplied, max-ticks "
            "and protocol seed are taken from the locked baseline-labeling section"
        ),
    )
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

    predeclared = None
    predeclared_sha = None
    if args.predeclaration is not None:
        predeclared = _load_repair_predeclaration(args.predeclaration)
        predeclared_sha = file_sha256(args.predeclaration)
        protocol = predeclared["protocol"]
        if str(args.frozen_policy) != str(protocol["baseline_frozen_policy"]):
            raise ValueError("repair acceptance labeling baseline frozen-policy path drift")
        args.max_ticks = int(protocol["labeling"]["max_ticks"])
        args.protocol_seed = int(protocol["labeling"]["protocol_seed"])
        acquisition_root = Path(args.catalog).parent
        copied = acquisition_root / "predeclaration.json"
        audit_path = acquisition_root / "anchor_audit.json"
        if file_sha256(copied) != predeclared_sha:
            raise ValueError("repair acceptance acquisition/predeclaration identity drift")
        audit = _read_json(audit_path)
        if audit.get("predeclaration_file_sha256") != predeclared_sha:
            raise ValueError("repair acceptance anchor audit/predeclaration drift")
        if audit.get("predeclared_protocol_sha256") != predeclared["expected_protocol_sha256"]:
            raise ValueError("repair acceptance anchor audit protocol drift")

    if args.max_ticks <= 0:
        parser.error("--max-ticks must be positive")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")

    frozen = load_frozen_unified_manifest(args.frozen_policy)
    record = frozen["policy"]
    if record["policy_role"] != "envelope_expansion_authority":
        raise ValueError("frozen unified policy is not an expansion authority")
    frozen_sha = file_sha256(args.frozen_policy)
    if predeclared is not None:
        protocol = predeclared["protocol"]
        if record["name"] != protocol["baseline_policy_name"]:
            raise ValueError("repair acceptance labeling baseline policy-name drift")
        if record["actor_sha256"] != protocol["baseline_actor_sha256"]:
            raise ValueError("repair acceptance labeling baseline actor drift")
        if record["payload_sha256"] != protocol["baseline_payload_sha256"]:
            raise ValueError("repair acceptance labeling baseline payload drift")

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

    acceptance_bank = None
    if predeclared is not None:
        output = Path(args.output_dir)
        (output / "acceptance_predeclaration.json").write_text(
            json.dumps(predeclared, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output / "acceptance_predeclaration_identity.json").write_text(
            json.dumps(
                {
                    "schema": "jit_repair_acceptance_label_binding_v1",
                    "predeclaration": str(args.predeclaration),
                    "predeclaration_file_sha256": predeclared_sha,
                    "predeclared_protocol_sha256": predeclared["expected_protocol_sha256"],
                    "baseline_policy_name": record["name"],
                    "baseline_actor_sha256": record["actor_sha256"],
                    "baseline_payload_sha256": record["payload_sha256"],
                    "max_ticks": int(args.max_ticks),
                    "protocol_seed": int(args.protocol_seed),
                    "training_transitions": 0,
                    "validation_data_used": False,
                    "test_data_used": False,
                    "final_evaluation_data_used": False,
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        bank_lock = predeclared["protocol"]["bank_lock"]
        acceptance_bank = lock_negative_acceptance_bank(
            output / "labels.json",
            args.catalog,
            output / "acceptance_bank.json",
            target_tube_path=Path(bank_lock["target_tube"]),
            expected_target_tube_manifest_sha256=str(
                bank_lock["target_tube_manifest_sha256"]
            ),
            predeclaration_file_sha256=str(predeclared_sha),
            predeclared_protocol_sha256=str(predeclared["expected_protocol_sha256"]),
            minimum_negative_states_per_phase=int(
                bank_lock["minimum_negative_states_per_phase"]
            ),
            minimum_negative_parent_groups_per_phase=int(
                bank_lock["minimum_negative_parent_groups_per_phase"]
            ),
        )

    payload_out = {"labeling": report}
    if acceptance_bank is not None:
        payload_out["acceptance_bank"] = {
            key: value for key, value in acceptance_bank.items() if key != "entries"
        }
    print(json.dumps(payload_out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
