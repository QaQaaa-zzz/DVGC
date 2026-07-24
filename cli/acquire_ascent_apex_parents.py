"""Acquire independent Ascent->Apex trajectory parents with bounded controls.

This is proposal discovery, not certification.  It first produces fresh
Takeoff->Ascent entries from the frozen Takeoff controller bank, then applies
predeclared local variations around the known hip-full/knee-half sequence.
Round B is a deterministic bounded shooting fallback and never changes the
entry detector or physical gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from cli.search_takeoff_actions import SEQUENCES, action_at
from cli.stage_label_pilot import sample_from_state
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.env import END_REASON, OrangeBikeDVGC
from dvgc.reset_geometry import GroundSupportSolver
from dvgc.rollout import restore_snapshot
from dvgc.runtime import build_inference, load_params, save_json
from dvgc.stage_reachability import evaluate_entry


FEATURE_NAMES = (
    "x", "y", "z", "roll", "pitch", "yaw", "vx", "vy", "vz",
    "wx", "wy", "wz", "steering", "hip", "knee", "wheel_speed",
)


def _sha(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _parent_id(row: dict, reset_hash: str, seed: int) -> str:
    payload = {
        "upstream_source_parent_id": row.get(
            "upstream_source_parent_id", row.get("source_parent_id", row["id"])
        ),
        "reset_protocol_hash": reset_hash,
        "initial_state_id": row["id"],
        "dynamics_seed": int(seed),
    }
    return _sha(payload)


def _balanced_rows(records: list[dict]) -> list[dict]:
    kinds = ("canonical_compressed", "reference_aligned_compressed")
    buckets = {kind: [row for row in records if row.get("candidate_kind") == kind]
               for kind in kinds}
    rows = []
    for i in range(max(map(len, buckets.values()))):
        for kind in kinds:
            if i < len(buckets[kind]):
                rows.append(buckets[kind][i])
    return rows


def _select_diverse_entries(entries: list[dict], target: int) -> list[dict]:
    """Select controller/source-balanced, max-min physical Ascent entries."""
    if len(entries) <= target:
        return list(entries)
    chosen, used = [], set()
    kinds = ("canonical_compressed", "reference_aligned_compressed")
    controllers = sorted({row["takeoff_controller"] for row in entries})
    # Seed the set with every available source/controller cell.
    for kind in kinds:
        for controller in controllers:
            row = next((row for row in entries
                        if row["upstream_source_kind"] == kind
                        and row["takeoff_controller"] == controller
                        and row["id"] not in used), None)
            if row is not None and len(chosen) < target:
                chosen.append(row); used.add(row["id"])
    dims = np.asarray([3, 4, 8, 9, 10, 11, 13, 14])
    matrix = np.asarray([row["physical_feature"] for row in entries], float)[:, dims]
    scale = np.maximum(np.std(matrix, axis=0), np.asarray([
        .03, .03, .2, .2, .2, .2, .05, .05,
    ]))
    by_id = {row["id"]: i for i, row in enumerate(entries)}
    while len(chosen) < target:
        counts = Counter(row["upstream_source_kind"] for row in chosen)
        desired_kind = min(kinds, key=lambda kind: counts[kind])
        candidates = [row for row in entries if row["id"] not in used
                      and row["upstream_source_kind"] == desired_kind]
        if not candidates:
            candidates = [row for row in entries if row["id"] not in used]
        chosen_z = matrix[[by_id[row["id"]] for row in chosen]] / scale
        def distance(row):
            z = matrix[by_id[row["id"]]] / scale
            return float(np.min(np.linalg.norm(chosen_z - z[None, :], axis=1)))
        row = max(candidates, key=distance)
        chosen.append(row); used.add(row["id"])
    return chosen


def _round_a() -> list[dict]:
    return [
        {
            "round": "A", "hip_amplitude": hip, "knee_ratio": ratio,
            "start_tick": start, "duration": duration,
        }
        for start in (0, 3, 6)
        for duration in (16, 28, 44)
        for hip in (.70, .85, 1.0)
        for ratio in (.35, .50, .65)
    ]


def _round_b(seed: int, count: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        rows.append({
            "round": "B",
            "hip_amplitude": float(rng.uniform(.60, 1.0)),
            "knee_ratio": float(rng.uniform(.25, .75)),
            "start_tick": int(rng.integers(0, 9)),
            "duration": int(rng.integers(12, 56)),
        })
    return rows


def _local_action(spec: dict, tick: int) -> jp.ndarray:
    if tick < spec["start_tick"]:
        return jp.zeros((4,), jp.float32)
    if tick < spec["start_tick"] + spec["duration"]:
        return jp.asarray(
            [0., 0., spec["hip_amplitude"],
             spec["hip_amplitude"] * spec["knee_ratio"]], jp.float32
        )
    return jp.asarray([0., 0., -.15, -.15], jp.float32)


def _reason(state) -> str:
    code = int(np.asarray(jax.device_get(state.info["end_code"])))
    return END_REASON.get(code, f"unknown_{code}")


def _takeoff_entries(args, cfg, env, step, source, policies):
    reset_hash = source.metadata["reset_protocol_sha256"]
    inferences = []
    for name, path in policies:
        inferences.append((
            name,
            build_inference(env, load_params(path / "params.pkl"), deterministic=True),
            file_sha256(path / "params.pkl"),
        ))
    bounded = (
        ("bounded:hip_full_knee_half", SEQUENCES["hip_full_knee_half"]),
        ("bounded:reference_hold_then_extend", SEQUENCES["hold_then_extend"]),
    )
    entries, attempts = [], []
    kind_parents = Counter()
    pool_target = min(
        len(source.records),
        max(args.target_parents,
            args.target_parents * int(args.entry_pool_multiplier)),
    )
    for source_index, row in enumerate(_balanced_rows(source.records)):
        if len(entries) >= pool_target:
            break
        seed = args.seed + source_index * 1000
        parent_id = _parent_id(row, reset_hash, seed)
        policy_controllers = [
            (name, lambda state, key, tick, infer=infer: infer(state.obs, key)[0], phash)
            for name, infer, phash in inferences
        ]
        if policy_controllers:
            rotate = source_index % len(policy_controllers)
            policy_controllers = policy_controllers[rotate:] + policy_controllers[:rotate]
        controllers = list(policy_controllers)
        if not args.takeoff_policy_only:
            controllers += [
            (name, lambda state, key, tick, sequence=sequence: action_at(sequence, tick), None)
            for name, sequence in bounded
            ]
        accepted = False
        for ci, (name, action_fn, policy_hash) in enumerate(controllers):
            state = restore_snapshot(env, row, jax.random.PRNGKey(seed + ci))
            previous_vz = float(np.asarray(state.data.qvel[2]))
            reason = "horizon_exhaustion"
            first_entry_tick = None
            for tick in range(args.takeoff_horizon):
                key = jax.random.PRNGKey(seed + ci * 100 + tick)
                action = action_fn(state, key, tick)
                state = step(state, action)
                sample = sample_from_state(env, state, previous_vz)
                entry = evaluate_entry("takeoff", sample, cfg)
                if entry["valid"] and first_entry_tick is None:
                    first_entry_tick = tick + 1
                # The event detector may fire on the first clearance tick,
                # before the phase machine's fixed airborne confirmation
                # window has elapsed.  Downstream proposal snapshots are
                # captured only after the real phase/latches have transitioned
                # to Flight; the frozen Takeoff success definition is not
                # changed.
                phase = int(np.asarray(state.info["phase"]))
                if first_entry_tick is not None and phase == 2:
                    snapshot = env.snapshot_record(state, "flight")
                    snapshot.update({
                        "id": hashlib.sha256(
                            f"ascent-entry:{parent_id}:{name}:{tick+1}".encode()
                        ).hexdigest()[:32],
                        "candidate_kind": "fresh_takeoff_to_ascent_entry",
                        "flight_subinterval": "ascent",
                        "trajectory_parent_id": parent_id,
                        "upstream_source_parent_id": row["id"],
                        "upstream_source_kind": row["candidate_kind"],
                        "upstream_reference_index": row.get("reference_index"),
                        "reset_protocol_sha256": reset_hash,
                        "dynamics_seed": seed,
                        "takeoff_controller": name,
                        "takeoff_controller_policy_hash": policy_hash,
                        "takeoff_entry_tick": first_entry_tick,
                        "flight_confirmation_tick": tick + 1,
                        "takeoff_source_state_id": row["id"],
                    })
                    entries.append(snapshot)
                    kind_parents[row["candidate_kind"]] += 1
                    attempts.append({
                        "source_id": row["id"], "source_kind": row["candidate_kind"],
                        "trajectory_parent_id": parent_id, "controller": name,
                        "success": True, "entry_tick": first_entry_tick,
                        "flight_confirmation_tick": tick + 1,
                    })
                    accepted = True
                    break
                if float(np.asarray(state.done)) > .5:
                    reason = _reason(state)
                    break
                previous_vz = float(sample["physical_feature"][8])
            if accepted:
                break
            attempts.append({
                "source_id": row["id"], "source_kind": row["candidate_kind"],
                "trajectory_parent_id": parent_id, "controller": name,
                "success": False, "failure_reason": reason,
            })
    selected = _select_diverse_entries(entries, args.target_parents)
    source_mix = dict(Counter(row["upstream_source_kind"] for row in selected))
    controller_mix = dict(Counter(row["takeoff_controller"] for row in selected))
    return selected, attempts, source_mix, controller_mix, len(entries)


def _search_parent(args, cfg, env, step, row, specs, round_name):
    outcomes, snapshots = [], []
    for si, spec in enumerate(specs):
        seed = args.seed + (20_000_000 if round_name == "B" else 10_000_000)
        seed += int(row["dynamics_seed"]) % 1_000_000 + si
        state = restore_snapshot(env, row, jax.random.PRNGKey(seed))
        previous_vz = float(np.asarray(state.data.qvel[2]))
        reason = "horizon_exhaustion"
        trace, recent = [], []
        success = False
        for tick in range(args.ascent_horizon):
            action = _local_action(spec, tick)
            state = step(state, action)
            sample = sample_from_state(env, state, previous_vz)
            entry = evaluate_entry("ascent", sample, cfg)
            feature = np.asarray(sample["physical_feature"], float)
            trace.append({
                "tick": tick + 1, "feature": feature.tolist(),
                "action": np.asarray(action).tolist(),
                "valid_apex_entry": bool(entry["valid"]),
                "entry_reasons": entry["reasons"],
            })
            recent.append((tick + 1, state, np.asarray(action), sample))
            recent = recent[-3:]
            if entry["valid"]:
                success = True
                reason = "next_stage_entry"
                # Keep the event and up to two immediately preceding real
                # trajectory snapshots; these remain one trajectory parent.
                for rel, (capture_tick, capture_state, capture_action, capture_sample) in enumerate(recent):
                    snapshot = env.snapshot_record(capture_state, "flight")
                    snapshot.update({
                        "id": hashlib.sha256(
                            f"dynamic-apex:{row['trajectory_parent_id']}:{round_name}:"
                            f"{si}:{capture_tick}".encode()
                        ).hexdigest()[:32],
                        "candidate_kind": "apex_dynamically_reached",
                        "apex_support_class": "dynamically_reached_candidate",
                        "flight_subinterval": "apex",
                        "trajectory_parent_id": row["trajectory_parent_id"],
                        "upstream_source_parent_id": row["upstream_source_parent_id"],
                        "source_ascent_entry_id": row["id"],
                        "source_takeoff_state_id": row["takeoff_source_state_id"],
                        "source_reset_protocol_sha256": row["reset_protocol_sha256"],
                        "source_dynamics_seed": row["dynamics_seed"],
                        "generation_round": round_name,
                        "generation_seed": seed,
                        "generation_proposal_index": si,
                        "generation_parameters": dict(spec),
                        "generation_entry_tick": tick + 1,
                        "snapshot_tick": capture_tick,
                        "apex_snapshot_stratum": (
                            "event" if rel == len(recent) - 1 else "pre_event"
                        ),
                        "continuation_action": capture_action.tolist(),
                        "entry_quality": evaluate_entry("ascent", capture_sample, cfg),
                    })
                    snapshots.append(snapshot)
                break
            if float(np.asarray(state.done)) > .5:
                reason = _reason(state)
                break
            previous_vz = float(feature[8])
        min_apex_residual = min(
            abs(frame["feature"][8]) + max(0., .4015 - frame["feature"][2])
            + max(0., frame["feature"][2] - .7015)
            for frame in trace
        ) if trace else None
        outcomes.append({
            "trajectory_parent_id": row["trajectory_parent_id"],
            "upstream_source_parent_id": row["upstream_source_parent_id"],
            "upstream_source_kind": row["upstream_source_kind"],
            "source_ascent_entry_id": row["id"],
            "round": round_name, "proposal_index": si, "seed": seed,
            "parameters": dict(spec), "success": success,
            "entry_tick": tick + 1 if success else None,
            "failure_reason": None if success else reason,
            "minimum_apex_residual": min_apex_residual,
            "action_saturation_fraction": float(np.mean([
                np.mean(np.abs(frame["action"]) >= .999) for frame in trace
            ])) if trace else 0.,
            "trace": trace,
        })
        if success:
            break
    return outcomes, snapshots


def _reproduce_reference_parents(args, cfg, env, step, bank):
    selected = [row for row in bank.records if row.get("reference_index") in
                (131, 140, 144, 160, 162, 172)]
    rows = []
    for row in selected:
        runs = []
        for seed in (args.seed + 30_000_000, args.seed + 31_000_000):
            state = restore_snapshot(env, row, jax.random.PRNGKey(seed))
            previous_vz = float(np.asarray(state.data.qvel[2]))
            trace, success, reason = [], False, "horizon_exhaustion"
            for tick in range(args.ascent_horizon):
                action = action_at(SEQUENCES["hip_full_knee_half"], tick)
                state = step(state, action)
                sample = sample_from_state(env, state, previous_vz)
                entry = evaluate_entry("ascent", sample, cfg)
                trace.append({
                    "tick": tick + 1,
                    "physical_feature": np.asarray(sample["physical_feature"]).tolist(),
                    "action": np.asarray(action).tolist(),
                    "entry_reasons": entry["reasons"],
                })
                if entry["valid"]:
                    success, reason = True, "next_stage_entry"
                    break
                if float(np.asarray(state.done)) > .5:
                    reason = _reason(state)
                    break
                previous_vz = float(sample["physical_feature"][8])
            runs.append({
                "seed": seed, "success": success,
                "entry_tick": tick + 1 if success else None,
                "reason": reason, "trace": trace,
            })
        rows.append({
            "reference_index": row["reference_index"], "candidate_id": row["id"],
            "initial_physical_feature": np.asarray(row["physical_feature"]).tolist(),
            "runs": runs,
        })
    parent131 = next(row for row in rows if row["reference_index"] == 131)
    base = np.asarray(parent131["initial_physical_feature"], float)
    for row in rows:
        delta = np.asarray(row["initial_physical_feature"], float) - base
        row["difference_from_parent_131"] = dict(zip(FEATURE_NAMES, delta.tolist()))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--takeoff-bank", required=True)
    p.add_argument("--reference-ascent-bank", required=True)
    p.add_argument("--descent-support-bank", required=True)
    p.add_argument("--policy", action="append", default=[], help="name=policy_dir")
    p.add_argument("--output-root", required=True)
    p.add_argument("--target-parents", type=int, default=12)
    p.add_argument("--entry-pool-multiplier", type=int, default=1)
    p.add_argument("--takeoff-policy-only", action="store_true")
    p.add_argument("--required-successful-parents", type=int, default=2)
    p.add_argument("--round-b-proposals", type=int, default=96)
    p.add_argument("--takeoff-horizon", type=int, default=80)
    p.add_argument("--ascent-horizon", type=int, default=100)
    p.add_argument("--seed", type=int, default=10_610_000)
    p.add_argument("--config", default="configs/default.json")
    args = p.parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    source = SnapshotBank.load(args.takeoff_bank)
    reference_bank = SnapshotBank.load(args.reference_ascent_bank)
    descent_support = SnapshotBank.load(args.descent_support_bank)
    cfg = load_config(args.config, {
        "training_stage": "flight", "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "stage_reachability_objective": "",
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    step = jax.jit(env.step)
    policies = []
    for spec in args.policy:
        name, path = spec.split("=", 1)
        policies.append((name, Path(path)))
    entry_path = root / "fresh_ascent_entries.pkl"
    entry_report_path = root / "fresh_ascent_entries_report.json"
    reference_replay_source = "inline_fixed_reference_replay"
    if entry_path.exists():
        entries = SnapshotBank.load(entry_path).records
        entry_report = (json.loads(entry_report_path.read_text())
                        if entry_report_path.exists() else {})
        reference_replay = entry_report.get("reference_parent_replay")
        reference_replay_source = entry_report.get(
            "reference_parent_replay_source", reference_replay_source
        )
        if reference_replay is None:
            completed_audit = (
                root.parents[1] / "apex/interface_v5/parent_robustness_v2.json"
            )
            if completed_audit.exists():
                audit = json.loads(completed_audit.read_text())
                reference_replay = [
                    {"parent_id": parent_id, **summary}
                    for parent_id, summary in audit["parents"].items()
                ]
                reference_replay_source = {
                    "artifact": str(completed_audit.resolve()),
                    "artifact_sha256": file_sha256(completed_audit),
                    "reuse_reason": (
                        "same frozen-runtime parent robustness audit already "
                        "completed atomically"
                    ),
                }
            else:
                reference_replay = _reproduce_reference_parents(
                    args, cfg, env, step, reference_bank
                )
        takeoff_attempts = entry_report.get("fresh_takeoff_attempts", [])
        source_mix = dict(Counter(row["upstream_source_kind"] for row in entries))
        controller_mix = dict(Counter(row["takeoff_controller"] for row in entries))
        entry_pool = entry_report.get(
            "fresh_ascent_entry_pool_before_diversity_selection"
        )
    else:
        reference_replay = _reproduce_reference_parents(
            args, cfg, env, step, reference_bank
        )
        entries, takeoff_attempts, source_mix, controller_mix, entry_pool = _takeoff_entries(
            args, cfg, env, step, source, policies
        )
        SnapshotBank(entries, {
            "artifact_role": "fresh_parent_disjoint_ascent_entry_proposals",
            "certified_tube": False, "safe_claim_allowed": False,
            "source_takeoff_bank_sha256": file_sha256(args.takeoff_bank),
            "generation_seed": args.seed,
        }).save(entry_path)
        save_json(entry_report_path, {
            "status": "PASS",
            "reference_parent_replay": reference_replay,
            "reference_parent_replay_source": reference_replay_source,
            "fresh_takeoff_attempts": takeoff_attempts,
            "fresh_ascent_entry_pool_before_diversity_selection": entry_pool,
            "fresh_ascent_entries": len(entries),
            "fresh_ascent_entry_source_mix": source_mix,
            "fresh_ascent_entry_controller_mix": controller_mix,
            "entry_bank_sha256": file_sha256(entry_path),
        })
    shard_root = root / "parent_search_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    round_a, apex_snapshots = [], []
    for row in entries:
        result_path = shard_root / f"{row['trajectory_parent_id']}_round_a.json"
        snapshot_path = shard_root / f"{row['trajectory_parent_id']}_round_a.pkl"
        if result_path.exists() and snapshot_path.exists():
            outcomes = json.loads(result_path.read_text())["outcomes"]
            snapshots = SnapshotBank.load(snapshot_path).records
        else:
            outcomes, snapshots = _search_parent(
                args, cfg, env, step, row, _round_a(), "A"
            )
            save_json(result_path, {
                "status": "PASS", "trajectory_parent_id": row["trajectory_parent_id"],
                "outcomes": outcomes,
            })
            SnapshotBank(snapshots, {
                "artifact_role": "per_parent_dynamic_apex_search_shard",
                "round": "A", "trajectory_parent_id": row["trajectory_parent_id"],
            }).save(snapshot_path)
        round_a.extend(outcomes); apex_snapshots.extend(snapshots)
    success_parents = {
        row["trajectory_parent_id"] for row in round_a if row["success"]
    }
    round_b = []
    if len(success_parents) < int(args.required_successful_parents):
        for pi, row in enumerate(entries):
            result_path = shard_root / f"{row['trajectory_parent_id']}_round_b.json"
            snapshot_path = shard_root / f"{row['trajectory_parent_id']}_round_b.pkl"
            if result_path.exists() and snapshot_path.exists():
                outcomes = json.loads(result_path.read_text())["outcomes"]
                snapshots = SnapshotBank.load(snapshot_path).records
            else:
                outcomes, snapshots = _search_parent(
                    args, cfg, env, step, row,
                    _round_b(args.seed + 40_000_000 + pi, args.round_b_proposals),
                    "B",
                )
                save_json(result_path, {
                    "status": "PASS",
                    "trajectory_parent_id": row["trajectory_parent_id"],
                    "outcomes": outcomes,
                })
                SnapshotBank(snapshots, {
                    "artifact_role": "per_parent_dynamic_apex_search_shard",
                    "round": "B", "trajectory_parent_id": row["trajectory_parent_id"],
                }).save(snapshot_path)
            round_b.extend(outcomes); apex_snapshots.extend(snapshots)
        success_parents.update(
            row["trajectory_parent_id"] for row in round_b if row["success"]
        )
    # One trajectory may contribute several event-neighbour snapshots, but
    # parent count is always computed from upstream lineage, never seed count.
    support_metadata = dict(descent_support.metadata)
    support_metadata["support_features"] = [
        row["physical_feature"] for row in descent_support.records
    ]
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    hip = model.joint("hip_joint")
    knee = model.joint("knee_joint")
    hip_q = int(model.jnt_qposadr[hip.id])
    knee_q = int(model.jnt_qposadr[knee.id])
    geometry = GroundSupportSolver(cfg.xml_path)
    kept = []
    parent_counts = Counter()
    validation_rejections = Counter()
    for row in apex_snapshots:
        parent = row["trajectory_parent_id"]
        if parent_counts[parent] >= 4:
            continue
        qpos = np.asarray(row["qpos"], float)
        qvel = np.asarray(row["qvel"], float)
        ctrl = np.asarray(row["ctrl"], float)
        reason = None
        if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()
                and np.isfinite(ctrl).all()):
            reason = "nonfinite"
        elif not (model.jnt_range[hip.id, 0] <= qpos[hip_q] <=
                  model.jnt_range[hip.id, 1]):
            reason = "hip_joint_limit"
        elif not (model.jnt_range[knee.id, 0] <= qpos[knee_q] <=
                  model.jnt_range[knee.id, 1]):
            reason = "knee_joint_limit"
        else:
            contact = geometry.measure(qpos, qvel, ctrl)
            if contact["body_contacts"] or contact["wheel_contacts"]:
                reason = "terrain_contact"
            elif row.get("source_phase") != "flight":
                reason = "wrong_phase"
        if reason is None:
            restored = restore_snapshot(
                env, row, jax.random.PRNGKey(int(row["generation_seed"]) + 700_000)
            )
            sample = sample_from_state(env, restored, float(qvel[2]))
            if evaluate_entry("apex", sample, cfg, support_metadata)["valid"]:
                reason = "premature_descent_entry"
        if reason is None:
            continuation = jp.asarray(row["continuation_action"], jp.float32)
            probe = restored
            for _ in range(5):
                probe = step(probe, continuation)
                if float(np.asarray(probe.done)) > .5:
                    reason = "five_step_reset_shock"
                    break
        if reason is not None:
            validation_rejections[reason] += 1
            continue
        row["contact_summary"] = contact
        row["five_step_reset_shock"] = False
        kept.append(row); parent_counts[parent] += 1
    SnapshotBank(kept, {
        "artifact_role": "dynamic_apex_proposal_support",
        "certified_tube": False, "safe_claim_allowed": False,
        "source_ascent_entry_bank_sha256": file_sha256(root / "fresh_ascent_entries.pkl"),
        "generation_seed": args.seed, "per_parent_snapshot_cap": 4,
    }).save(root / "dynamic_apex_proposals.pkl")
    all_search = round_a + round_b
    payload = {
        "status": ("PASS" if len(success_parents) >= int(args.required_successful_parents)
                   else "STAGE_LOCAL_BLOCKER"),
        "artifact_role": "ascent_apex_independent_parent_acquisition",
        "claim_scope": "proposal_discovery_only",
        "chain_final_semantics": {
            "chain_success": "valid generic Apex entry",
            "downstream_support_entry": "not yet evaluated",
            "final_recovery": "not yet evaluated",
        },
        "inputs": {
            "takeoff_bank": str(Path(args.takeoff_bank).resolve()),
            "takeoff_bank_sha256": file_sha256(args.takeoff_bank),
            "reference_ascent_bank_sha256": file_sha256(args.reference_ascent_bank),
            "policies": [{
                "id": name, "path": str(path.resolve()),
                "params_sha256": file_sha256(path / "params.pkl"),
            } for name, path in policies],
        },
        "reference_parent_replay": reference_replay,
        "reference_parent_replay_source": reference_replay_source,
        "fresh_takeoff_attempts": takeoff_attempts,
        "fresh_ascent_entries": len(entries),
        "fresh_ascent_entry_pool_before_diversity_selection": entry_pool,
        "fresh_ascent_entry_source_mix": source_mix,
        "fresh_ascent_entry_controller_mix": controller_mix,
        "independent_upstream_parents": len({
            row["trajectory_parent_id"] for row in entries
        }),
        "round_a": {
            "proposals_evaluated": len(round_a),
            "successful_parents": len({
                row["trajectory_parent_id"] for row in round_a if row["success"]
            }),
            "termination_reasons": dict(Counter(
                "next_stage_entry" if row["success"] else row["failure_reason"]
                for row in round_a
            )),
        },
        "round_b": {
            "executed": bool(round_b), "proposals_evaluated": len(round_b),
            "successful_parents": len({
                row["trajectory_parent_id"] for row in round_b if row["success"]
            }),
            "termination_reasons": dict(Counter(
                "next_stage_entry" if row["success"] else row["failure_reason"]
                for row in round_b
            )),
        },
        "successful_parent_count": len(success_parents),
        "successful_parent_ids": sorted(success_parents),
        "dynamic_apex_snapshots": len(kept),
        "dynamic_apex_parent_count": len(parent_counts),
        "dynamic_apex_validation_rejections": dict(validation_rejections),
        "late_ascent_training_authorized": len(success_parents) >= 2,
        "apex_training_authorized": False,
        "stage_local_blocker": (
            None if len(success_parents) >= int(args.required_successful_parents)
            else "ascent_multi_parent_controller_gap"
        ),
        "search_outcomes": all_search,
        "artifacts": {
            "fresh_ascent_entries": str((root / "fresh_ascent_entries.pkl").resolve()),
            "dynamic_apex_proposals": str((root / "dynamic_apex_proposals.pkl").resolve()),
        },
    }
    save_json(root / "report.json", payload)
    print(json.dumps({
        "status": payload["status"], "fresh_entries": len(entries),
        "source_mix": source_mix, "round_a_parents":
        payload["round_a"]["successful_parents"], "round_b_parents":
        payload["round_b"]["successful_parents"],
        "successful_parents": len(success_parents),
        "dynamic_apex_snapshots": len(kept),
    }, indent=2))


if __name__ == "__main__":
    main()
