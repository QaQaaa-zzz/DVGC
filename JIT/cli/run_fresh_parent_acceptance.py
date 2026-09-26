#!/usr/bin/env python3
"""Prepare, audit, or acquire a fresh physical-parent acceptance boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_new(path: Path, payload) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _prepare(args) -> int:
    from jit_dvgc.acquisition.fresh_parent import (
        prepare_fresh_parent_acceptance_predeclaration,
    )

    payload = prepare_fresh_parent_acceptance_predeclaration(
        baseline_frozen_policy=args.baseline_frozen_policy,
        target_tube=args.target_tube,
        consumed_gate_roots=args.consumed_gate,
        consumed_baseline_probe_roots=args.consumed_baseline_probe,
        acquisition_seed=args.acquisition_seed,
        labeling_seed=args.labeling_seed,
        parent_excitation_strength=args.parent_excitation_strength,
        parent_excitation_duration=args.parent_excitation_duration,
        upstream_apex_offset=args.upstream_apex_offset,
        downstream_apex_offset=args.downstream_apex_offset,
        parent_near_duplicate_atol=args.parent_near_duplicate_atol,
        minimum_parent_groups=args.minimum_parent_groups,
        boundary_strengths=args.boundary_strengths,
        boundary_durations=args.boundary_durations,
        boundary_active_action_dimensions=args.boundary_active_action_dimensions,
        minimum_negative_states_per_phase=args.minimum_negative_states_per_phase,
        minimum_negative_parent_groups_per_phase=(
            args.minimum_negative_parent_groups_per_phase
        ),
    )
    _write_new(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _audit(args) -> int:
    from jit_dvgc.acquisition.fresh_parent import audit_fresh_parent_predeclaration

    predeclared = _read_json(args.predeclaration)
    report = audit_fresh_parent_predeclaration(predeclared)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predeclaration.json").write_text(
        json.dumps(predeclared, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "design_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


def _acquire(args) -> int:
    import jax

    from jit_dvgc.acquisition.fresh_parent import (
        audit_fresh_parent_predeclaration,
        collect_fresh_parent_anchors,
        collect_snapshot_anchor_boundary_candidates,
    )
    from jit_dvgc.checkpoint import load_checkpoint
    from jit_dvgc.config import file_sha256
    from jit_dvgc.ppo import make_checkpoint_policy
    from jit_dvgc.repair_acceptance import canonical_sha256
    from jit_dvgc.training import (
        build_unified_formal_environment,
        checkpoint_identity,
        load_frozen_unified_manifest,
    )

    predeclared = _read_json(args.predeclaration)
    design = audit_fresh_parent_predeclaration(predeclared)
    protocol = predeclared["protocol"]
    if canonical_sha256(protocol) != predeclared["expected_protocol_sha256"]:
        raise ValueError("fresh parent predeclaration SHA drift")
    if str(args.frozen_policy) != str(protocol["baseline_frozen_policy"]):
        raise ValueError("fresh parent frozen-policy path drift")
    if jax.default_backend() != "gpu":
        raise RuntimeError("fresh parent acceptance acquisition requires the visible JAX GPU")

    frozen = load_frozen_unified_manifest(args.frozen_policy)
    record = frozen["policy"]
    if record["name"] != protocol["baseline_policy_name"]:
        raise ValueError("fresh parent baseline policy name drift")
    if record["actor_sha256"] != protocol["baseline_actor_sha256"]:
        raise ValueError("fresh parent baseline actor drift")
    if record["payload_sha256"] != protocol["baseline_payload_sha256"]:
        raise ValueError("fresh parent baseline payload drift")

    config, _artifact, env = build_unified_formal_environment(Path(record["formal_config"]))
    if config.config_sha256 != record["formal_config_sha256"]:
        raise ValueError("fresh parent formal config drift")
    if env._bundle.xml_sha256 != record["xml_sha256"]:
        raise ValueError("fresh parent XML drift")
    if int(config.ppo.episode_horizon) != int(protocol["labeling"]["max_ticks"]):
        raise ValueError("fresh parent horizon drift")
    payload = load_checkpoint(
        Path(record["checkpoint"]), expected=checkpoint_identity(config, env)
    )
    if file_sha256(Path(record["checkpoint"]) / "payload.pkl") != record["payload_sha256"]:
        raise ValueError("fresh parent checkpoint payload SHA drift")
    policy = jax.jit(make_checkpoint_policy(env, payload, deterministic=True))
    step_fn = jax.jit(env.step)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "predeclaration.json").write_text(
        json.dumps(predeclared, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "design_audit.json").write_text(
        json.dumps(design, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    try:
        anchors, parent_report = collect_fresh_parent_anchors(
            predeclared,
            output,
            env=env,
            policy=policy,
            policy_record=record,
            compiled_step_fn=step_fn,
        )
        anchor_audit = {
            "schema": "jit_fresh_parent_anchor_audit_v1",
            "status": "completed",
            "predeclaration": str(args.predeclaration),
            "predeclaration_file_sha256": file_sha256(args.predeclaration),
            "predeclared_protocol_sha256": predeclared["expected_protocol_sha256"],
            "parent_source": {
                key: value for key, value in parent_report.items() if key != "records"
            },
            "selected_anchor_count": len(anchors),
            "selected_phase_counts": {
                "upstream": sum(a.phase == "upstream" for a in anchors),
                "downstream": sum(a.phase == "downstream" for a in anchors),
            },
            "training_transitions": 0,
            "validation_data_used": False,
            "test_data_used": False,
            "final_evaluation_data_used": False,
        }
        (output / "anchor_audit.json").write_text(
            json.dumps(anchor_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        report = collect_snapshot_anchor_boundary_candidates(
            predeclared,
            anchors,
            output,
            env=env,
            policy=policy,
            policy_record=record,
            frozen_manifest_sha256=file_sha256(args.frozen_policy),
            compiled_step_fn=step_fn,
        )
        print(
            json.dumps(
                {
                    "parent_source": {
                        key: value
                        for key, value in parent_report.items()
                        if key != "records"
                    },
                    "candidate_acquisition": {
                        key: value for key, value in report.items() if key != "entries"
                    },
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except BaseException as exc:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "schema": "jit_fresh_parent_acceptance_failure_v1",
                    "status": "engineering_or_readiness_error",
                    "predeclaration": str(args.predeclaration),
                    "frozen_policy": str(args.frozen_policy),
                    "training_transitions": 0,
                    "validation_data_used": False,
                    "test_data_used": False,
                    "final_evaluation_data_used": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="predeclare fresh physical-parent acceptance acquisition")
    prepare.add_argument("--baseline-frozen-policy", type=Path, required=True)
    prepare.add_argument("--target-tube", type=Path, required=True)
    prepare.add_argument("--consumed-gate", type=Path, action="append", required=True)
    prepare.add_argument("--consumed-baseline-probe", type=Path, action="append", default=[])
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--acquisition-seed", type=int, required=True)
    prepare.add_argument("--labeling-seed", type=int, required=True)
    prepare.add_argument("--parent-excitation-strength", type=float, default=0.10)
    prepare.add_argument("--parent-excitation-duration", type=int, default=2)
    prepare.add_argument("--upstream-apex-offset", type=int, default=-10)
    prepare.add_argument("--downstream-apex-offset", type=int, default=10)
    prepare.add_argument("--parent-near-duplicate-atol", type=float, default=1.0e-5)
    prepare.add_argument("--minimum-parent-groups", type=int, default=4)
    prepare.add_argument("--boundary-strengths", type=float, nargs="+", default=[0.15, 0.30, 0.50])
    prepare.add_argument("--boundary-durations", type=int, nargs="+", default=[2, 4, 8])
    prepare.add_argument("--boundary-active-action-dimensions", type=int, default=2)
    prepare.add_argument("--minimum-negative-states-per-phase", type=int, default=10)
    prepare.add_argument("--minimum-negative-parent-groups-per-phase", type=int, default=3)

    audit = sub.add_parser("audit", help="zero-interaction audit of a fresh-parent declaration")
    audit.add_argument("--predeclaration", type=Path, required=True)
    audit.add_argument("--output-dir", type=Path, required=True)

    acquire = sub.add_parser("acquire", help="run parent trajectories and unlabeled boundary acquisition")
    acquire.add_argument("--frozen-policy", type=Path, required=True)
    acquire.add_argument("--predeclaration", type=Path, required=True)
    acquire.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "audit":
        return _audit(args)
    return _acquire(args)


if __name__ == "__main__":
    raise SystemExit(main())
