"""Production-only workers for fixed-ground-prefix and continuation diagnostics."""
from __future__ import annotations

import importlib.metadata
from pathlib import Path
import traceback

import jax
import jax.numpy as jp
import numpy as np

from .jump_evidence_validation import (SCHEMA, canonical_sha256, compare_traces,
    file_sha, load_plan, read, select_sample_indices, write)
from .unified_envelope_snapshot import (UP_EVENT_FIELDS, DOWN_EVENT_FIELDS,
    capture_unified_envelope_snapshot, load_unified_envelope_snapshot,
    physical_state_sha256, restore_unified_envelope_snapshot,
    save_unified_envelope_snapshot, snapshot_context_sha256)

PHYSICS_FIELDS = ("qpos", "qvel", "ctrl", "observation", "history", "history_valid_count",
                  "last_action", "action", "rng", "active_phase", "source_tick", "parent_group_index",
                  "tube_entry_index", "tube_global_index", "done", "physical_failure",
                  "timeout", "end_code", "success")
EVENT_FIELDS = tuple("up/" + name for name in UP_EVENT_FIELDS if name != "episode_step") + tuple(
    "down/" + name for name in DOWN_EVENT_FIELDS)
CLOCK_FIELDS = ("start_phase", "episode_step", "phase_episode_step", "up/episode_step", "phase_transitioned", "episode_return")
PRESERVED_FIELDS = PHYSICS_FIELDS + EVENT_FIELDS + CLOCK_FIELDS


def scalar(value):
    return np.asarray(jax.device_get(value)).item()


def view(state):
    result = {"qpos": np.asarray(jax.device_get(state.data.qpos)).tolist(),
              "qvel": np.asarray(jax.device_get(state.data.qvel)).tolist(),
              "ctrl": np.asarray(jax.device_get(state.data.ctrl)).tolist(),
              "observation": np.asarray(jax.device_get(state.obs["state"])).tolist(),
              "history": np.asarray(jax.device_get(state.info["history"].frames)).tolist(),
              "history_valid_count": scalar(state.info["history"].valid_count),
              "last_action": np.asarray(jax.device_get(state.info["last_action"])).tolist(),
              "action": [0.0] * 4, "done": bool(scalar(state.done))}
    result["rng"] = np.asarray(jax.device_get(jax.random.key_data(state.info["rng"]))).tolist()
    for key in ("start_phase", "source_tick", "parent_group_index", "tube_entry_index", "tube_global_index", "active_phase", "physical_failure", "timeout", "end_code", "success",
                "episode_step", "phase_episode_step", "phase_transitioned", "episode_return"):
        result[key] = scalar(state.info[key])
    for prefix, fields in (("up", UP_EVENT_FIELDS), ("down", DOWN_EVENT_FIELDS)):
        for field in fields:
            result[prefix + "/" + field] = scalar(getattr(state.info[prefix + "_events"], field))
    if not all(np.isfinite(np.asarray(value)).all() for value in result.values()):
        raise ValueError("nonfinite runtime state in evidence audit")
    if bool(scalar(state.info["expert_switching_used"])):
        raise ValueError("audit unexpectedly used expert switching")
    return result


def simulator_start_differences(left, right):
    result = {}
    # These are diagnostics, not a claim that the snapshot serializes all MJX internals.
    for name in ("time", "qacc_warmstart", "act", "qfrc_applied", "xfrc_applied"):
        if not hasattr(left.data, name) or not hasattr(right.data, name):
            result[name] = {"available": False}
            continue
        a, b = np.asarray(jax.device_get(getattr(left.data, name))), np.asarray(jax.device_get(getattr(right.data, name)))
        result[name] = {"available": True, "exactly_equal": bool(np.array_equal(a, b)),
                        "max_abs_error": float(np.max(np.abs(a - b))) if a.size else 0.0}
    return result


def runtime(item):
    from .checkpoint import load_checkpoint
    from .ppo import make_checkpoint_policy
    from .unified_formal import build_unified_formal_environment
    from .unified_policy_freeze import load_frozen_unified_manifest
    from .unified_training import checkpoint_identity
    record = load_frozen_unified_manifest(Path(item["path"]))["policy"]
    if record != item["policy"]:
        raise ValueError("runtime frozen record differs from locked audit")
    config, artifact, env = build_unified_formal_environment(Path(record["formal_config"]))
    payload = load_checkpoint(Path(record["checkpoint"]), expected=checkpoint_identity(config, env))
    return env, make_checkpoint_policy(env, payload, deterministic=True)


class Steps:
    def __init__(self):
        self.calls = 0
        self.maximum = 0

    def bind(self, env):
        if "warp" in str(env._require_runtime_model().impl.value).lower():
            from .frontier_label_shard_runner import _build_memory_stable_step
            compiled = _build_memory_stable_step(env)
        else:
            compiled = jax.jit(env.step)

        def step(state, action):
            if self.calls >= self.maximum:
                raise RuntimeError("audit stage interaction ceiling exhausted")
            self.calls += 1  # Includes a dispatched call that subsequently raises.
            result = compiled(state, action)
            jax.block_until_ready(result)
            return result
        return step


def capture(state, env, record, parent, group):
    return capture_unified_envelope_snapshot(state, env=env, parent_trajectory=group,
        parent_state_sha256=parent, config_sha256=record["formal_config_sha256"],
        policy_actor_sha256=record["actor_sha256"], policy_payload_sha256=record["payload_sha256"],
        policy_iteration=record["iteration"])


def action_for(policy, state, key):
    result = policy(state.obs, key)
    action = result[0] if isinstance(result, tuple) else result
    array = np.asarray(jax.device_get(action), dtype=np.float32)
    if array.shape != (4,) or not np.isfinite(array).all():
        raise ValueError("invalid frozen Actor output")
    return jp.asarray(array)


def capture_worker(plan, output, steps):
    from .acquisition.causal_jump import physical_state_sha256_from_state
    env, policy = runtime(plan["proposer"])
    record = plan["proposer"]["policy"]
    step = steps.bind(env)
    state = jax.jit(env._reset_jump_start_unified)(jax.random.PRNGKey(plan["prefix_seed"]))
    jax.block_until_ready(state)
    if not bool(scalar(state.info["reset_from_jump_start"])) or abs(float(scalar(state.data.qpos[0])) - 2.5) > 1e-6:
        raise ValueError("audit did not start from declared x=2.5 ground reset")
    parent = physical_state_sha256_from_state(state)
    group = f"engineering_prefix_{plan['prefix_seed']}"
    start = capture(state, env, record, parent, group)
    save_unified_envelope_snapshot(output / "start_snapshot", start)
    from .geometry import extract_geometry
    geometry = extract_geometry(state.data, env._geometry)
    ground_geometry = {name: scalar(getattr(geometry, name)) for name in (
        "front_wheel_terrain_clearance", "rear_wheel_terrain_clearance",
        "maximum_wheel_penetration", "prohibited_contact")}
    write(output / "start_state.json", {"frame": view(state), "snapshot_context_sha256": snapshot_context_sha256(start),
        "rng": np.asarray(start.rng).tolist(), "xml_sha256": start.xml_sha256, "ground_geometry": ground_geometry,
        "compatibility_identity": start.compatibility_identity,
        "reset_key_seed": plan["prefix_seed"], "ground_contact_validation": "existing_fixed_ground_reset_implementation"})
    protocol = {"schema": SCHEMA, "purpose": "engineering_validation_only", "plan_sha256": plan["plan_sha256"],
                "prefix_seed": plan["prefix_seed"], "start_context_sha256": snapshot_context_sha256(start),
                "proposer_actor_sha256": record["actor_sha256"], "frozen_manifest_sha256": plan["proposer"]["file_sha256"],
                "maximum_environment_interactions": plan["horizon"], "selection_rule": plan["selection_rule"]}
    protocol["protocol_sha256"] = canonical_sha256(protocol)
    write(output / "protocol.json", protocol)  # Actual complete reset identity locked before env.step.
    actions, frames, snapshots = [], [], []
    key = jax.random.PRNGKey(plan["prefix_seed"])
    terminal = None
    for tick in range(plan["horizon"]):
        action = action_for(policy, state, jax.random.fold_in(key, tick))
        state = step(state, action)
        actions.append(np.asarray(jax.device_get(action)).tolist())
        current = view(state)
        current.update(prefix_ticks=tick + 1, phase_index=current["active_phase"],
                       valid_contact_seen=current["down/valid_contact_seen"], vz=current["qvel"][2])
        frames.append(current)
        if current["done"] or current["valid_contact_seen"]:
            snapshots.append(None)
            terminal = current
            break
        # Keep only compact CPU snapshots, never hundreds of live MJX States on GPU.
        snapshots.append(capture(state, env, record, parent, group))
    write(output / "prefix_actions.json", actions)
    write(output / "prefix_frames.json", frames)
    indices, eligible_counts = select_sample_indices(frames, plan["per_phase"])
    entries = []
    for index in indices:
        snapshot = snapshots[index]
        relative = f"snapshots/candidate_{len(entries):03d}"
        save_unified_envelope_snapshot(output / "bank" / relative, snapshot)
        phase = "upstream" if snapshot.active_phase == 0 else "downstream"
        entries.append({"candidate_id": f"audit_{len(entries):03d}", "candidate_kind": "reachable_unified_frontier_probe",
            "split": "train", "logical_role": "engineering_validation", "phase": phase,
            "phase_index": snapshot.active_phase, "source_bank": "bank", "snapshot": relative,
            "state_sha256": physical_state_sha256(snapshot), "snapshot_context_sha256": snapshot_context_sha256(snapshot),
            "parent_group_id": group, "parent_state_sha256": parent, "policy_iteration": record["iteration"],
            "policy_actor_sha256": record["actor_sha256"], "policy_payload_sha256": record["payload_sha256"],
            "protocol_sha256": protocol["protocol_sha256"], "prefix_ticks": frames[index]["prefix_ticks"],
            "capture_frame": frames[index], "tube_admission_authorized": False})
    catalog = {"schema": "jit_unified_boundary_catalog_v1", "status": "completed",
        "artifact_role": "unlabeled_policy_conditioned_frontier_candidates", "split": "train",
        "logical_role": "engineering_validation", "legacy_train_marker_is_not_training_authorization": True,
        "iteration": record["iteration"], "policy_name": record["name"], "policy_actor_sha256": record["actor_sha256"],
        "policy_payload_sha256": record["payload_sha256"], "frozen_unified_manifest_sha256": plan["proposer"]["file_sha256"],
        "protocol_sha256": protocol["protocol_sha256"], "candidate_count": len(entries), "entries": entries,
        "training_transitions": 0, "expert_switching_used": False, "test_data_used": False,
        "validation_data_used": False, "final_evaluation_data_used": False,
        "claim_boundary": {"unlabeled_acquisition_only": True, "tube_expansion_claim": False,
                           "jce_jel_claim": False, "certified_safe_set_claim": False}}
    write(output / "catalog.json", catalog)
    observed_transition = any(frame["phase_transitioned"] for frame in frames)
    return {"candidate_count": len(entries), "eligible_phase_counts": eligible_counts,
            "selected_prefix_ticks": [row["prefix_ticks"] for row in entries],
            "apex_transition_observed": observed_transition,
            "coverage_complete": len(entries) == 2 * plan["per_phase"] and observed_transition,
            "prefix_valid_landing_observed": bool(terminal and terminal["valid_contact_seen"]),
            "prefix_end_code": terminal["end_code"] if terminal else None,
            "engineering_panel_only": True, "catalog_file_sha256": file_sha(output / "catalog.json")}


def rollout(state, policy, step, *, seed, index, horizon):
    from .unified_continuation_labels import classify_first_valid_landing_outcome
    trace = [view(state)]
    candidate_key = jax.random.fold_in(jax.random.PRNGKey(seed), index)
    for tick in range(horizon):
        if trace[-1]["done"] or trace[-1]["down/valid_contact_seen"]:
            break
        action = action_for(policy, state, jax.random.fold_in(candidate_key, tick))
        trace[-1]["action"] = np.asarray(jax.device_get(action)).tolist()
        state = step(state, action)
        trace.append(view(state))
    final = trace[-1]
    valid = bool(final["down/valid_contact_seen"])
    positive, outcome = classify_first_valid_landing_outcome(valid_contact_seen=valid,
        physical_failure_before_landing=bool(final["physical_failure"]) and not valid,
        timeout=bool(final["timeout"]), done=bool(final["done"]),
        reached_rollout_horizon=len(trace) - 1 >= horizon and not final["done"])
    return trace, {"label": int(positive), "outcome_class": outcome, "ticks": len(trace) - 1,
                   "end_code": final["end_code"], "timeout": final["timeout"]}


def replay_worker(plan, output, steps, evaluator_index, sample_index):
    from .unified_continuation_labels import fresh_unified_continuation_state
    root = Path(plan["output"]) / "capture"
    row = read(root / "catalog.json")["entries"][sample_index]
    snapshot = load_unified_envelope_snapshot(root / row["source_bank"] / row["snapshot"])
    source_env, source_policy = runtime(plan["proposer"])
    source_step = steps.bind(source_env)
    raw = jax.jit(source_env._reset_jump_start_unified)(jax.random.PRNGKey(plan["prefix_seed"]))
    for action in read(root / "prefix_actions.json")[:row["prefix_ticks"]]:
        if bool(scalar(raw.done)):
            raise ValueError("reference prefix became terminal before selected frame")
        raw = source_step(raw, jp.asarray(action, jp.float32))
    actual = capture(raw, source_env, plan["proposer"]["policy"], row["parent_state_sha256"], row["parent_group_id"])
    prefix = compare_traces([row["capture_frame"]], [view(raw)], atol=plan["atol"], rtol=plan["rtol"], fields=PRESERVED_FIELDS)
    prefix["exact_stored_context_match"] = snapshot_context_sha256(actual) == row["snapshot_context_sha256"]
    prefix["passed"] &= prefix["exact_stored_context_match"]
    if plan["evaluators"][evaluator_index]["path"] == plan["proposer"]["path"]:
        env, policy = source_env, source_policy
    else:
        env, policy = runtime(plan["evaluators"][evaluator_index])
    step = steps.bind(env)
    restored = restore_unified_envelope_snapshot(snapshot, env)
    differences = simulator_start_differences(raw, restored)
    starts = {"raw": raw, "preserved": restored,
              "raw_fresh": fresh_unified_continuation_state(raw),
              "restored_fresh": fresh_unified_continuation_state(restored)}
    trajectories, outcomes = {}, {}
    for name, state in starts.items():
        trajectories[name], outcomes[name] = rollout(state, policy, step, seed=plan["label_seed"],
            index=sample_index, horizon=plan["horizon"])
    def compare(a, b, fields):
        result = compare_traces(trajectories[a], trajectories[b], atol=plan["atol"], rtol=plan["rtol"], fields=fields)
        result["outcomes_equal"] = outcomes[a] == outcomes[b]
        result["passed"] &= result["outcomes_equal"]
        return result
    write(output / "trajectories.json", trajectories)
    return {"candidate_id": row["candidate_id"], "phase": row["phase"], "prefix_ticks": row["prefix_ticks"],
            "evaluator": plan["evaluators"][evaluator_index]["policy"]["name"], "prefix_replay": prefix,
            "preserved_restore": compare("raw", "preserved", PRESERVED_FIELDS),
            "fresh_restore": compare("raw_fresh", "restored_fresh", PRESERVED_FIELDS),
            "counter_effect": compare("raw", "raw_fresh", PHYSICS_FIELDS + EVENT_FIELDS),
            "counter_effect_excludes_declared_clock_offsets_from_numeric_gate": list(CLOCK_FIELDS),
            "initial_clocks": {name: {field: trace[0][field] for field in CLOCK_FIELDS} for name, trace in trajectories.items()},
            "simulator_fields_at_restore": differences, "outcomes": outcomes,
            "all_simulator_internal_state_serialized": False,
            "suffix_horizon_semantics": "same frozen rollout ceiling; runtime retains or resets original phase clocks as named"}


def label_worker(plan, output, steps, evaluator_index, kind, shard_index, shard_count):
    from .unified_continuation_labels import label_unified_continuations
    from .unified_continuation_shards import label_unified_continuation_shard
    from .policy_family_landing import merge_policy_family_evaluator_shards
    catalog = Path(plan["output"]) / "capture/catalog.json"
    item = plan["evaluators"][evaluator_index]
    if kind == "merge":
        directories = [Path(plan["output"]) / f"evaluator_{evaluator_index}/shard_{i:03d}/result" for i in range(shard_count)]
        return merge_policy_family_evaluator_shards(catalog_path=catalog, shard_dirs=directories,
            output_dir=output / "result", evaluator_name=item["policy"]["name"],
            acquisition_frozen_policy=Path(plan["proposer"]["path"]), evaluator_frozen_policy=Path(item["path"]),
            max_ticks=plan["horizon"], protocol_seed=plan["label_seed"])
    env, policy = runtime(item)
    kwargs = dict(env=env, policy=policy, policy_record=item["policy"], frozen_manifest_sha256=item["file_sha256"],
                  max_ticks=plan["horizon"], protocol_seed=plan["label_seed"], compiled_step_fn=steps.bind(env),
                  acquisition_policy_record=plan["proposer"]["policy"],
                  acquisition_frozen_manifest_sha256=plan["proposer"]["file_sha256"], success_criterion="first_valid_landing")
    if kind == "serial":
        return label_unified_continuations(catalog, output / "result", **kwargs)
    return label_unified_continuation_shard(catalog, output / "result", shard_index=shard_index,
                                           shard_count=shard_count, **kwargs)


def run_worker(plan_path, output, *, kind, evaluator_index=0, sample_index=0, shard_index=0, shard_count=1):
    output = Path(output)
    steps = Steps()
    report = {"schema": SCHEMA, "worker": kind, "status": "running", "training_transitions": 0}
    code = 0
    try:
        plan = load_plan(plan_path)
        if kind == "capture":
            steps.maximum = plan["horizon"]
        elif kind == "replay":
            steps.maximum = 5 * plan["horizon"]
        elif kind in {"serial", "shard"}:
            count = read(Path(plan["output"]) / "capture/catalog.json")["candidate_count"]
            if kind == "shard":
                from .unified_continuation_shards import contiguous_shard_bounds
                start, stop = contiguous_shard_bounds(count, shard_index, shard_count)
                count = stop - start
            steps.maximum = count * plan["horizon"]
        if kind != "merge" and (jax.default_backend() != "gpu" or len(jax.local_devices()) != 1):
            raise RuntimeError("production audit requires exactly one visible JAX GPU; CPU is not a pass")
        report["runtime"] = {"jax_backend": jax.default_backend(), "devices": [str(d) for d in jax.local_devices()],
                             "versions": {name: importlib.metadata.version(name) for name in ("jax", "numpy", "mujoco", "brax")}}
        if kind == "capture":
            payload = capture_worker(plan, output, steps)
        elif kind == "replay":
            payload = replay_worker(plan, output, steps, evaluator_index, sample_index)
        else:
            payload = label_worker(plan, output, steps, evaluator_index, kind, shard_index, shard_count)
        report.update(payload)
        report["status"] = "completed"
    except BaseException as exc:
        report.update(status="engineering_error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        code = 1
    finally:
        report["maximum_environment_interactions"] = steps.maximum
        report["environment_interactions"] = steps.calls  # Merge costs zero; do not recount its label totals.
        report["includes_dispatched_step_that_raised"] = True
        write(output / "report.json", report)
    return code
