#!/usr/bin/env python3
"""Fast engineering check of one intermediate checkpoint from pi0 full warm-start B.

No training is performed.  One process evaluates exactly one checkpoint on the
same Tube0 core bank and locked 260-state boundary bank used by the B final
quickcheck.  Baseline pi0 outcomes are reused from the already-completed B final
gate.  Candidate rollouts are evaluated in GPU batches with ``vmap`` and a
compiled ``while_loop`` rather than one state / one tick Python dispatches.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

# Leave headroom for MuJoCo-Warp allocations instead of letting JAX reserve most
# of the 24 GiB device before Warp requests its collision buffers.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
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
DEFAULT_BATCH_SIZE = 64


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


def _stack_states(states):
    """Stack equally-shaped Brax/JAX state pytrees along a new batch axis."""

    def stack_leaf(*values):
        first = values[0]
        if first is None:
            return None
        return jnp.stack([jnp.asarray(value) for value in values], axis=0)

    return jax.tree_util.tree_map(
        stack_leaf,
        *states,
        is_leaf=lambda value: value is None,
    )


def _make_batched_rollout(env, policy, max_ticks: int):
    """Compile a many-environment candidate rollout with no per-tick host sync."""

    max_ticks = int(max_ticks)

    def rollout_one(initial_state, seed, start_phase):
        del start_phase  # classification happens on host after rollout
        apex_seen = jnp.asarray(initial_state.info["up_events"].apex_seen, dtype=bool)
        phase_transitioned = jnp.asarray(
            initial_state.info["phase_transitioned"], dtype=bool
        )
        recovery_success = jnp.asarray(
            initial_state.info["down_events"].recovery_success, dtype=bool
        )
        expert_switching_seen = jnp.asarray(
            initial_state.info["expert_switching_used"], dtype=bool
        )

        carry = (
            jnp.asarray(0, dtype=jnp.int32),
            initial_state,
            apex_seen,
            phase_transitioned,
            recovery_success,
            expert_switching_seen,
        )

        def cond_fn(value):
            tick, state, *_ = value
            return jnp.logical_and(
                tick < max_ticks,
                jnp.logical_not(jnp.asarray(state.done, dtype=bool)),
            )

        def body_fn(value):
            (
                tick,
                state,
                apex_seen,
                phase_transitioned,
                recovery_success,
                expert_switching_seen,
            ) = value
            key = jax.random.fold_in(jax.random.PRNGKey(seed), tick)
            result = policy(state.obs, key)
            action = result[0] if isinstance(result, tuple) else result
            state = env.step(state, action)
            apex_seen = jnp.logical_or(
                apex_seen,
                jnp.asarray(state.info["up_events"].apex_seen, dtype=bool),
            )
            phase_transitioned = jnp.logical_or(
                phase_transitioned,
                jnp.asarray(state.info["phase_transitioned"], dtype=bool),
            )
            recovery_success = jnp.logical_or(
                recovery_success,
                jnp.asarray(state.info["down_events"].recovery_success, dtype=bool),
            )
            expert_switching_seen = jnp.logical_or(
                expert_switching_seen,
                jnp.asarray(state.info["expert_switching_used"], dtype=bool),
            )
            return (
                tick + jnp.asarray(1, dtype=jnp.int32),
                state,
                apex_seen,
                phase_transitioned,
                recovery_success,
                expert_switching_seen,
            )

        return jax.lax.while_loop(cond_fn, body_fn, carry)

    return jax.jit(jax.vmap(rollout_one, in_axes=(0, 0, 0)))


def _candidate_results_from_batch(
    batched_output,
    *,
    start_phases: list[int],
    real_count: int,
    max_ticks: int,
) -> list[dict]:
    (
        ticks,
        final_state,
        apex_seen,
        phase_transitioned,
        recovery_success,
        expert_switching_seen,
    ) = jax.device_get(batched_output)

    ticks = np.asarray(ticks)[:real_count]
    done = np.asarray(final_state.done).astype(bool)[:real_count]
    terminal_success = np.asarray(final_state.info["success"]).astype(bool)[:real_count]
    physical_failure = np.asarray(final_state.info["physical_failure"]).astype(bool)[
        :real_count
    ]
    timeout = np.asarray(final_state.info["timeout"]).astype(bool)[:real_count]
    apex_seen = np.asarray(apex_seen).astype(bool)[:real_count]
    phase_transitioned = np.asarray(phase_transitioned).astype(bool)[:real_count]
    recovery_success = np.asarray(recovery_success).astype(bool)[:real_count]
    expert_switching_seen = np.asarray(expert_switching_seen).astype(bool)[:real_count]

    if bool(np.any(expert_switching_seen)):
        raise ValueError("batched checkpoint rollout unexpectedly used expert switching")

    results: list[dict] = []
    for index in range(real_count):
        interactions = int(ticks[index])
        positive, outcome = gate_mod.classify_unified_continuation_outcome(
            start_phase=int(start_phases[index]),
            terminal_success=bool(terminal_success[index]),
            physical_failure=bool(physical_failure[index]),
            timeout=bool(timeout[index]),
            done=bool(done[index]),
            apex_seen=bool(apex_seen[index]),
            phase_transitioned=bool(phase_transitioned[index]),
            recovery_success=bool(recovery_success[index]),
            reached_rollout_horizon=(
                interactions >= int(max_ticks) and not bool(done[index])
            ),
        )
        results.append(
            {
                "success": bool(positive),
                "outcome_class": str(outcome),
                "environment_interactions": interactions,
            }
        )
    return results


def _evaluate_one(
    protocol: dict,
    baseline_record: dict,
    candidate_record: dict,
    baseline_rows: dict[tuple[str, str], dict],
    *,
    batch_size: int,
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
    batched_rollout = _make_batched_rollout(env, candidate_policy, max_ticks)
    records = []
    candidate_interactions = 0

    rows = [*core_bank, *boundary_bank]
    if len(rows) != 482:
        raise ValueError("checkpoint quickcheck bank must contain exactly 482 states")

    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch-size must be positive")
    batch_count = (len(rows) + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, len(rows), batch_size), start=1):
        chunk = rows[start : start + batch_size]
        states = []
        seeds: list[int] = []
        start_phases: list[int] = []
        baselines: list[dict] = []

        for local_index, row in enumerate(chunk):
            global_index = start + local_index
            key = (str(row["bank_role"]), str(row["state_sha256"]))
            baseline = baseline_rows.get(key)
            if baseline is None:
                raise ValueError(f"missing reused baseline row: {key}")

            if row["bank_role"] == "core":
                state = reset_tube(
                    np.int32(row["phase_index"]), np.int32(row["entry_index"])
                )
                if gate_mod._sha256_state(state) != row["state_sha256"]:
                    raise ValueError("checkpoint quickcheck core reset state drift")
            else:
                snapshot = load_unified_envelope_snapshot(Path(str(row["snapshot"])))
                state = fresh_unified_continuation_start(snapshot, env)

            states.append(state)
            seeds.append(base_seed + global_index * 10_000)
            start_phases.append(int(row["phase_index"]))
            baselines.append(baseline)

        real_count = len(states)
        # Pad only the final chunk so the compiled batch shape remains constant;
        # padded results are discarded immediately after the rollout.
        while len(states) < batch_size:
            states.append(states[-1])
            seeds.append(seeds[-1])
            start_phases.append(start_phases[-1])

        stacked_state = _stack_states(states)
        output = batched_rollout(
            stacked_state,
            jnp.asarray(seeds, dtype=jnp.int32),
            jnp.asarray(start_phases, dtype=jnp.int32),
        )
        candidate_results = _candidate_results_from_batch(
            output,
            start_phases=start_phases,
            real_count=real_count,
            max_ticks=max_ticks,
        )

        for local_index, candidate in enumerate(candidate_results):
            row = chunk[local_index]
            baseline = baselines[local_index]
            candidate_interactions += int(candidate["environment_interactions"])
            records.append(
                {
                    **row,
                    "baseline_success": bool(baseline["baseline_success"]),
                    "candidate_success": bool(candidate["success"]),
                    "baseline_outcome_class": str(
                        baseline["baseline_outcome_class"]
                    ),
                    "candidate_outcome_class": candidate["outcome_class"],
                    "baseline_environment_interactions": int(
                        baseline["baseline_environment_interactions"]
                    ),
                    "candidate_environment_interactions": int(
                        candidate["environment_interactions"]
                    ),
                }
            )

        print(
            f"batch {batch_index}/{batch_count}: "
            f"states {start + 1}-{start + real_count}/{len(rows)}",
            flush=True,
        )
        del output, stacked_state, states
        gc.collect()

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
        "schema": "jit_pi0_full_warmstart_checkpoint_quickcheck_v3",
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
        "rollout_mode": "gpu_vmap_while_loop",
        "batch_size": batch_size,
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
        "schema": "jit_pi0_full_warmstart_checkpoint_sweep_v3",
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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="parallel GPU rollout batch size; lower to 32 if device memory is tight",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing per-checkpoint report",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
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
            protocol,
            baseline_record,
            candidate_record,
            baseline_rows,
            batch_size=args.batch_size,
        )
        _write_json(out, report)
    finally:
        gate_mod.load_unified_formal_config = original_loader

    _print_result(report)
    _write_summary_if_complete()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
