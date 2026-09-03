#!/usr/bin/env python3
"""Quick engineering check of one intermediate checkpoint from the pi0 full warm-start B run.

No training is performed.  One process evaluates exactly one checkpoint on the
same Tube0 core bank and locked 260-state boundary bank used by the B final
quickcheck.  Baseline pi0 outcomes are reused from the already-completed B final
gate, so this runner only rolls out the candidate checkpoint.  Running one
checkpoint per Python process avoids JAX/Warp compiled-executable and allocator
cache accumulation across checkpoints.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

import jit_dvgc.analysis.paired_policy_gate as gate_mod
from jit_dvgc.checkpoint import CheckpointIdentity, load_checkpoint
from jit_dvgc.config import file_sha256, load_config
from jit_dvgc.constants import ACTION_ORDER, ACTOR_FRAME_FIELDS, ACTOR_TASK_FIELDS
from jit_dvgc.handoff_bank import pytree_sha256
from jit_dvgc.unified_continuation_labels import fresh_unified_continuation_start
from jit_dvgc.unified_envelope_snapshot import load_unified_envelope_snapshot

import train_unified_from_pi0_full as full_warm


B_CONFIG = Path(
    "JIT/configs/pi_unified_iter1_pi0_full_warmstart_tube1_core_replay75_natural10.json"
)
B_RUN = Path(
    "JIT/runs/pi_unified/"
    "pi_1_pi0_full_warmstart_tube1_core_replay75_natural10_10009600_seed821101_20260903"
)
BASE_GATE_CONFIG = Path(
    "JIT/configs/pi0_to_pi1_warmstart_B_quickcheck_20260903.json"
)
BASE_GATE_RECORDS = Path(
    "JIT/runs/pi_unified_gate/"
    "pi0_to_pi1_warmstart_B_quickcheck_20260903/records.json"
)
OUTPUT_ROOT = Path(
    "JIT/runs/pi_unified_gate/"
    "pi0_to_pi1_warmstart_B_checkpoint_sweep_20260903"
)
TRANSITIONS = (1_024_000, 2_508_800, 5_017_600, 7_500_800)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _candidate_record(transition: int, config) -> dict:
    checkpoint = B_RUN / "checkpoints" / f"transition_{transition}"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"missing B checkpoint: {checkpoint}")

    up_config = load_config(Path(config.up_config_path))
    down_config = load_config(Path(config.down_config_path))
    if up_config.config_sha256 != config.up_config_sha256:
        raise ValueError("B upstream config hash drift")
    if down_config.config_sha256 != config.down_config_sha256:
        raise ValueError("B downstream config hash drift")
    xml_sha = str(up_config.model["xml_sha256"])
    if xml_sha != str(down_config.model["xml_sha256"]):
        raise ValueError("B phase XML mismatch")

    identity = CheckpointIdentity(
        config_sha256=config.config_sha256,
        xml_sha256=xml_sha,
        actor_frame_fields=ACTOR_FRAME_FIELDS,
        actor_task_fields=ACTOR_TASK_FIELDS,
        action_order=ACTION_ORDER,
    )
    payload = load_checkpoint(checkpoint, expected=identity)
    if int(payload.training_transitions) != int(transition):
        raise ValueError("B intermediate checkpoint transition drift")

    sidecar = _read_json(checkpoint / "identity.json")
    payload_sha = file_sha256(checkpoint / "payload.pkl")
    if str(sidecar.get("payload_sha256", "")) != payload_sha:
        raise ValueError("B intermediate checkpoint payload hash drift")

    run_id = str(config.raw["run_declaration"]["run_id"])
    return {
        "name": "pi_1",
        "iteration": 1,
        "policy_role": "engineering_checkpoint_candidate",
        "checkpoint": str(checkpoint),
        "formal_config": str(B_CONFIG),
        "formal_config_sha256": config.config_sha256,
        "xml_sha256": xml_sha,
        "source_training_run_id": run_id,
        "source_training_transitions": int(transition),
        "source_reset_mixture": config.reset_mixture.as_dict(),
        "payload_sha256": payload_sha,
        "normalizer_sha256": pytree_sha256(payload.observation_normalizer),
        "actor_sha256": pytree_sha256(payload.actor_params),
        "critic_sha256": pytree_sha256(payload.critic_params),
        "actor_frame_fields": list(ACTOR_FRAME_FIELDS),
        "actor_task_fields": list(ACTOR_TASK_FIELDS),
        "action_order": list(ACTION_ORDER),
    }


def _baseline_by_state() -> dict[tuple[str, str], dict]:
    payload = _read_json(BASE_GATE_RECORDS)
    rows = payload.get("records")
    if not isinstance(rows, list) or len(rows) != 482:
        raise ValueError("B final gate records must contain exactly 482 rows")
    result: dict[tuple[str, str], dict] = {}
    for source in rows:
        row = dict(source)
        key = (str(row.get("bank_role", "")), str(row.get("state_sha256", "")))
        if key in result or not key[0] or len(key[1]) != 64:
            raise ValueError("B final gate baseline record identity drift")
        for field in (
            "baseline_success",
            "baseline_outcome_class",
            "baseline_environment_interactions",
        ):
            if field not in row:
                raise ValueError(f"B final gate record missing {field}")
        result[key] = row
    return result


def _evaluate_one(
    protocol: dict,
    baseline_record: dict,
    candidate_record: dict,
    baseline_rows: dict[tuple[str, str], dict],
) -> dict:
    env, core_tube, target_tube = gate_mod._build_runtime(
        protocol, baseline_record, candidate_record
    )
    core_bank, boundary_bank, boundary_source = gate_mod._lock_bank(
        protocol, core_tube, target_tube, baseline_record
    )

    candidate_policy = gate_mod._checkpoint_policy(env, candidate_record)
    max_ticks = int(protocol["runtime"]["max_ticks"])
    base_seed = int(protocol["runtime"]["protocol_seed"])
    reset_tube = jax.jit(env.reset_tube_index)
    step_fn = jax.jit(env.step)
    records = []
    candidate_interactions = 0

    rows = [*core_bank, *boundary_bank]
    if len(rows) != 482:
        raise ValueError("checkpoint quickcheck bank must contain exactly 482 states")

    for index, row in enumerate(rows):
        key = (str(row["bank_role"]), str(row["state_sha256"]))
        baseline = baseline_rows.get(key)
        if baseline is None:
            raise ValueError(f"missing reused baseline row: {key}")

        if row["bank_role"] == "core":
            candidate_state = reset_tube(
                np.int32(row["phase_index"]), np.int32(row["entry_index"])
            )
            if gate_mod._sha256_state(candidate_state) != row["state_sha256"]:
                raise ValueError("checkpoint quickcheck core reset state drift")
        else:
            snapshot = load_unified_envelope_snapshot(Path(str(row["snapshot"])))
            candidate_state = fresh_unified_continuation_start(snapshot, env)

        state_seed = base_seed + index * 10_000
        candidate = gate_mod._rollout(
            env,
            candidate_policy,
            candidate_state,
            step_fn=step_fn,
            start_phase=int(row["phase_index"]),
            max_ticks=max_ticks,
            seed=state_seed,
        )
        candidate_interactions += int(candidate["environment_interactions"])
        records.append(
            {
                **row,
                "baseline_success": bool(baseline["baseline_success"]),
                "candidate_success": bool(candidate["success"]),
                "baseline_outcome_class": str(baseline["baseline_outcome_class"]),
                "candidate_outcome_class": candidate["outcome_class"],
                "baseline_environment_interactions": int(
                    baseline["baseline_environment_interactions"]
                ),
                "candidate_environment_interactions": candidate[
                    "environment_interactions"
                ],
            }
        )

    gates = gate_mod.summarize_paired_gate_records(
        records,
        minimum_candidate_success_parent_groups=int(
            protocol["boundary"]["minimum_candidate_success_parent_groups"]
        ),
        require_baseline_success_each_phase=bool(
            protocol["core"]["require_baseline_success_each_phase"]
        ),
    )
    return {
        "schema": "jit_pi0_full_warmstart_checkpoint_quickcheck_v2",
        "status": "completed",
        "checkpoint_transitions": int(candidate_record["source_training_transitions"]),
        "candidate_actor_sha256": candidate_record["actor_sha256"],
        "candidate_critic_sha256": candidate_record["critic_sha256"],
        "candidate_payload_sha256": candidate_record["payload_sha256"],
        "core_gate": gates["core"],
        "boundary_gate": gates["boundary"],
        "boundary_source": boundary_source,
        "candidate_environment_interactions": candidate_interactions,
        "baseline_reused_from": str(BASE_GATE_RECORDS),
        "engineering_quickcheck_only": True,
        "training_transitions": 0,
        "validation_data_used": False,
        "test_data_used": False,
    }


def _print_result(report: dict) -> None:
    core = report["core_gate"]
    boundary = report["boundary_gate"]
    print(
        f"{report['checkpoint_transitions']}: "
        f"core={core['candidate_success_count']}/222 "
        f"regressions={core['regression_count']} "
        f"boundary={boundary['candidate_success_count']}/260 "
        f"groups={boundary['candidate_success_parent_group_count']}"
    )


def _write_summary_if_complete() -> None:
    reports = []
    for transition in TRANSITIONS:
        path = OUTPUT_ROOT / f"transition_{transition}" / "report.json"
        if not path.is_file():
            return
        report = _read_json(path)
        if report.get("status") != "completed":
            return
        reports.append(report)

    summary = {
        "schema": "jit_pi0_full_warmstart_checkpoint_sweep_v2",
        "status": "completed",
        "source_run": str(B_RUN),
        "base_gate_config": str(BASE_GATE_CONFIG),
        "baseline_reused_from": str(BASE_GATE_RECORDS),
        "transitions": [int(x) for x in TRANSITIONS],
        "results": [
            {
                "checkpoint_transitions": r["checkpoint_transitions"],
                "core_success": r["core_gate"]["candidate_success_count"],
                "core_regressions": r["core_gate"]["regression_count"],
                "upstream_core_success": r["core_gate"]["phase_counts"]["upstream"]["candidate_success_count"],
                "downstream_core_success": r["core_gate"]["phase_counts"]["downstream"]["candidate_success_count"],
                "boundary_success": r["boundary_gate"]["candidate_success_count"],
                "boundary_parent_groups": r["boundary_gate"]["candidate_success_parent_group_count"],
                "upstream_boundary_success": r["boundary_gate"]["phase_counts"]["upstream"]["candidate_success_count"],
                "downstream_boundary_success": r["boundary_gate"]["phase_counts"]["downstream"]["candidate_success_count"],
            }
            for r in reports
        ],
    }
    _write_json(OUTPUT_ROOT / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transition",
        type=int,
        required=True,
        choices=TRANSITIONS,
        help="evaluate exactly one B intermediate checkpoint in this process",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing per-checkpoint report",
    )
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError("checkpoint quickcheck requires the visible JAX GPU backend")
    if not BASE_GATE_CONFIG.is_file():
        raise FileNotFoundError(
            f"B final quickcheck config missing: {BASE_GATE_CONFIG}; run B gate prepare first"
        )
    if not BASE_GATE_RECORDS.is_file():
        raise FileNotFoundError(
            f"B final gate records missing: {BASE_GATE_RECORDS}; run B gate first"
        )

    out = OUTPUT_ROOT / f"transition_{args.transition}" / "report.json"
    if out.is_file() and not args.force:
        report = _read_json(out)
        _print_result(report)
        print(f"existing report reused: {out}")
        _write_summary_if_complete()
        return 0

    base_gate = gate_mod.load_paired_policy_gate_config(BASE_GATE_CONFIG)
    protocol = dict(base_gate["protocol"])
    _, baseline_record = gate_mod._load_policy(protocol, "baseline")
    baseline_rows = _baseline_by_state()
    b_config = full_warm._load_warm_target_config(B_CONFIG)

    original_loader = gate_mod.load_unified_formal_config

    def compatible_loader(path: Path):
        path = Path(path)
        if path == B_CONFIG:
            return b_config
        return original_loader(path)

    gate_mod.load_unified_formal_config = compatible_loader
    try:
        candidate_record = _candidate_record(args.transition, b_config)
        report = _evaluate_one(
            protocol, baseline_record, candidate_record, baseline_rows
        )
        _write_json(out, report)
    finally:
        gate_mod.load_unified_formal_config = original_loader

    _print_result(report)
    _write_summary_if_complete()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
