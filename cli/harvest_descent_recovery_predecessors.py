"""Harvest and certify v4 predecessors from successful Descent recovery prefixes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.build_backward_tube_proposal_index import state_hash
from cli.freeze_descent_predecessor_assets import verify_frozen_assets
from cli.run_backward_descent_nominal_pilot import (
    C_L, EXPECTED, PI_D, PI_L, PERTURBATIONS, _batched, _load_record,
    _micro_states, _outcome, _restore,
)
from cli.runtime_gate import source_fingerprint
from dvgc.backward_search import active_prefix_exact, make_descent_landing_rollout
from dvgc.backward_tube import (
    BackwardTubeNode, balanced_p1_launch_gate, balanced_p1_launch_subset,
    canonical_hash, p0_decision, p1_decision, validate_parent_lineage,
)
from dvgc.bank import SnapshotBank
from dvgc.config import ACTION_MAPPING_VERSION, file_sha256, load_config
from dvgc.descent_predecessor import (
    expand_residual_knots, predecessor_priority, remaining_residual_suffix,
    require_forward_lineage,
)
from dvgc.entry import entry_feature_from_physical, normalized_nearest
from dvgc.policy import load_bundle
from dvgc.ppo_integrity import normalizer_summary
from dvgc.rollout import restore_snapshot_mode
from dvgc.runtime import build_inference, save_json
from dvgc.snapshot_timing import validate_snapshot_v4


PRIOR = Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json")
INDEX = Path("runs/backward_recovery_tube_fast_track_v1/proposal_state_index.json")
ROOT = Path("runs/descent_diverse_p1_predecessor_recovery_v1")
CONFIG = Path("configs/default.json")
HORIZON = 200


def _atomic_pickle(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, pickle.HIGHEST_PROTOCOL)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _effective_provenance(cfg, policy_cfg, params) -> dict[str, str]:
    effective = canonical_hash({"base_config": file_sha256(CONFIG), "policy_config": policy_cfg,
                                "overrides": {"episode_length": HORIZON, "training_stage": "flight",
                                              "domain_randomization": False, "obs_noise_enable": False}})
    return {
        "xml_sha256": file_sha256(cfg.xml_path), "config_sha256": effective,
        "action_mapping_version": ACTION_MAPPING_VERSION,
        "policy_params_sha256": file_sha256(PI_D / "params.pkl"),
        "policy_config_sha256": file_sha256(PI_D / "config.json"),
        "policy_manifest_sha256": file_sha256(PI_D / "manifest.json"),
        "normalizer_sha256": normalizer_summary(params[0])["sha256"],
        "source_fingerprint": source_fingerprint(Path.cwd()),
    }


def _validate_v4(env, record, provenance, command):
    restored = jax.jit(lambda key: restore_snapshot_mode(
        env, record, key, observation_mode="timing_explicit_independent_reconstruction"))
    return validate_snapshot_v4(
        record,
        expected_shapes={"qpos": (env.mj_model.nq,), "qvel": (env.mj_model.nv,),
                         "act": (env.mj_model.na,), "ctrl_previous": (env.mj_model.nu,),
                         "qacc_warmstart": (env.mj_model.nv,), "sensordata": (env.mj_model.nsensordata,)},
        expected_hashes=provenance,
        actor_action_fn=lambda _obs: command,
        ctrl_from_action_fn=lambda action: np.asarray(env._action_to_ctrl(
            jnp.asarray(action), jnp.asarray(record["physical_state_t"]["qpos"])[env._joint_qpos["knee_joint"]])),
        current_frame_fn=lambda _record: np.asarray(restored(jax.random.PRNGKey(0)).obs["state"]).reshape(
            env._actor_history_steps, env._actor_frame_dim)[-1],
    )


def _controller_rows(report):
    rows = {}
    for row in report["rows"]:
        if row["P0"]["pass"]:
            rows[(row["proposal"]["physical_state_sha256"], row["controller"])] = row
    return rows


def _trajectory(env, dpolicy, lpolicy, node, controller_row, provenance, seed, *, capture):
    record = _load_record(controller_row["proposal"])
    state = _restore(env, record, jax.random.PRNGKey(seed))
    expanded = expand_residual_knots(controller_row.get("residual_knots"))
    step = jax.jit(env.step)
    snapshots, trace = [], []
    handed = False; entry_tick = -1; final = False; end_code = 0
    for tick in range(HORIZON):
        da, _ = dpolicy(state.obs, jax.random.PRNGKey(seed + tick))
        la, _ = lpolicy(state.obs, jax.random.PRNGKey(seed + 100000 + tick))
        base = la if handed else da
        residual = expanded[tick] if tick < len(expanded) and not handed else np.zeros(4, np.float32)
        command = np.asarray(jnp.clip(base + jnp.asarray(residual), -1., 1.), np.float32)
        fifo_valid = int(np.asarray(state.info["actor_packet_fifo_valid"]))
        if capture and not handed and fifo_valid == 3:
            v4 = env.snapshot_record_v4(state, "flight", jnp.asarray(command), provenance)
            validation = _validate_v4(env, v4, provenance, command)
            if not validation["valid"]:
                raise RuntimeError(f"v4 identity failure {node.node_id}:{tick}:{validation['failed']}")
            snapshots.append({"snapshot_v4": v4, "source_tick": tick,
                              "actor_packet_fifo_valid": fifo_valid,
                              "command": command, "expanded_residual": expanded.copy()})
        before_qpos = np.asarray(state.data.qpos)
        state = step(state, jnp.asarray(command))
        chain = bool(np.asarray(state.info["chain_ever"]))
        if chain and not handed:
            handed = True; entry_tick = tick + 1
        final |= bool(np.asarray(state.info["recovery_success"]))
        end_code = int(np.asarray(state.info["end_code"]))
        trace.append((before_qpos.copy(), command.copy(), np.asarray(state.data.qpos).copy(), end_code, handed, final))
        if bool(np.asarray(state.done)):
            break
    if entry_tick < 0 or not final:
        raise RuntimeError(f"frozen successful source no longer reaches Final: {node.node_id}")
    proposals = []
    for item in snapshots:
        if item["source_tick"] >= entry_tick:
            continue
        record = item["snapshot_v4"]; physical = record["physical_state_t"]
        identifier = state_hash(physical["qpos"], physical["qvel"], physical["ctrl_previous"], physical["qacc_warmstart"])
        proposals.append({
            **item, "physical_state_hash": identifier, "candidate_id": node.candidate_id,
            "source_node_id": node.node_id, "source_was_p1": node.p1,
            "source_controller_type": node.controller_type,
            "source_controller_artifact_sha256": node.controller_artifact_sha256,
            "downstream_entry_tick": entry_tick,
            "relative_tick_to_downstream_entry": item["source_tick"] - entry_tick,
            "controller_suffix": remaining_residual_suffix(expanded, item["source_tick"]),
            "construction_method": "forward_mjx_active_prefix", "state_splicing": False,
            "reverse_integration": False, "controller_suffix_available": True,
            "source_trajectory_exact_final_recovery": True,
        })
    return proposals, trace


def _setup():
    valid, failed = verify_frozen_assets(ROOT)
    if not valid:
        raise SystemExit(f"frozen asset identity failure: {failed}")
    report = json.loads(PRIOR.read_text()); index = json.loads(INDEX.read_text())
    dparams, policy_cfg, _ = load_bundle(PI_D, verify_files=True)
    lparams, _, _ = load_bundle(PI_L, verify_files=True)
    cfg = load_config(CONFIG, {**policy_cfg, "episode_length": 750, "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False,
        "expert_chain_termination": False, "training_stage": "flight"})
    if file_sha256(C_L) != EXPECTED["C_L"] or file_sha256(PI_D / "params.pkl") != EXPECTED["pi_D"] or file_sha256(PI_L / "params.pkl") != EXPECTED["pi_L"]:
        raise SystemExit("frozen policy/C_L identity failure")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")
    entry = SnapshotBank.load(C_L)
    env = __import__("dvgc.env", fromlist=["OrangeBikeDVGC"]).OrangeBikeDVGC(
        cfg, snapshot_bank=SnapshotBank(), cert_bank=entry)
    return report, index, cfg, entry, env, dparams, lparams, policy_cfg


def _source_order(nodes):
    counts = Counter(node.candidate_id for node in nodes if node.p1)
    dominant = counts.most_common(1)[0][0]
    return sorted(nodes, key=lambda node: (
        node.candidate_id == dominant, not node.p1, -node.entry_tick,
        counts.get(node.candidate_id, 0), node.candidate_id, node.node_id))


def _harvest(mode):
    report, index, cfg, entry, env, dparams, lparams, policy_cfg = _setup()
    nodes = [BackwardTubeNode(**row) for row in report["nodes"]]
    controllers = _controller_rows(report)
    dpolicy = build_inference(env, dparams, deterministic=True)
    lpolicy = build_inference(env, lparams, deterministic=True)
    provenance = _effective_provenance(cfg, policy_cfg, dparams)
    ordered = _source_order(nodes)
    pilot_path = ROOT / "source_a_pilot_snapshots.pkl"
    if mode == "pilot":
        if pilot_path.exists(): raise SystemExit("pilot already complete")
        ordered = ordered[:1]; all_proposals = []
    else:
        if (ROOT / "trajectory_harvested_snapshots.pkl").exists(): raise SystemExit("full harvest already complete")
        if not pilot_path.exists(): raise SystemExit("2-5% Source-A pilot must complete first")
        all_proposals = pickle.loads(pilot_path.read_bytes())
        pilot_sources = {row["source_node_id"] for row in all_proposals}
        ordered = [node for node in ordered if node.node_id not in pilot_sources]
    controller_rows = []
    for position, node in enumerate(ordered):
        key = (node.source_state_hash, node.controller_type)
        if key not in controllers: raise RuntimeError(f"controller artifact missing: {key}")
        proposals, _ = _trajectory(env, dpolicy, lpolicy, node, controllers[key], provenance,
                                   51_000_000 + position, capture=True)
        all_proposals.extend(proposals); controller_rows.append(node.node_id)
    dedup = {}; existing = {node.source_state_hash for node in nodes}; duplicate_counts = Counter()
    for row in all_proposals:
        require_forward_lineage(row)
        if row["physical_state_hash"] in existing:
            duplicate_counts["existing_P0_P1_identity"] += 1; continue
        if row["physical_state_hash"] in dedup:
            duplicate_counts["harvest_identity"] += 1; continue
        dedup[row["physical_state_hash"]] = row
    proposals = list(dedup.values())
    output_pkl = pilot_path if mode == "pilot" else ROOT / "trajectory_harvested_snapshots.pkl"
    _atomic_pickle(output_pkl, proposals)
    report_path = ROOT / ("source_a_pilot_report.json" if mode == "pilot" else "trajectory_harvested_proposals.json")
    save_json(report_path, {"status": "PASS", "mode": mode, "source_trajectories": len(controller_rows),
        "proposal_count": len(proposals), "duplicate_rejections": dict(duplicate_counts),
        "proposal_artifact": str(output_pkl), "proposal_artifact_sha256": file_sha256(output_pkl),
        "source_node_ids": controller_rows, "v4_identity": "100%", "heldout_used": False,
        "delay": False, "PPO": False,
        "rows": [{k: v for k, v in row.items() if k not in {"snapshot_v4", "command", "expanded_residual", "controller_suffix"}}
                 for row in proposals]})
    print(json.dumps({"mode": mode, "source_trajectories": len(controller_rows),
                      "proposals": len(proposals), "duplicates": dict(duplicate_counts)}, indent=2))


def _certify():
    report, index, cfg, entry, env, dparams, lparams, policy_cfg = _setup()
    proposals = pickle.loads((ROOT / "trajectory_harvested_snapshots.pkl").read_bytes())
    for artifact_index, proposal in enumerate(proposals):
        proposal["_artifact_index"] = artifact_index
    old_nodes = [BackwardTubeNode(**row) for row in report["nodes"]]
    old_p1 = [node for node in old_nodes if node.p1]
    counts = Counter(node.candidate_id for node in old_p1); dominant = counts.most_common(1)[0][0]
    proposals.sort(key=lambda row: predecessor_priority(row, dominant, counts))
    rollout = make_descent_landing_rollout(env, dparams, lparams, horizon=HORIZON,
                                            residual_ticks=8, ticks_per_knot=1)
    index_rows = index["rows"]
    quantiles = np.quantile([row["distance_to_nearest_downstream_safe_node"] for row in index_rows], [.25, .5, .75])
    safe = [row for row in entry.records if row["final"]["label"] == "safe"]
    entries = np.asarray([row["entry_feature"] for row in safe], np.float64)
    ids = [row["id"] for row in safe]; matcher = entry.metadata["entry_matcher"]
    center, scale = np.asarray(matcher["center"]), np.asarray(matcher["scale"])
    prereg = json.loads((ROOT / "preregistration.json").read_text())
    feature_scale = np.asarray(prereg["feature_scale"])
    old_feature = {}
    proposal_by_hash = {row["physical_state_sha256"]: row for row in index_rows}
    for node in old_p1:
        old_feature[node.node_id] = np.asarray(_load_record(proposal_by_hash[node.source_state_hash])["physical_feature"])
    new_nodes, certification_rows = [], []
    for position, proposal in enumerate(proposals):
        if proposal["candidate_id"] == dominant: continue
        record = proposal["snapshot_v4"]; seed = 52_000_000 + position
        commands = jnp.asarray(np.asarray(proposal["controller_suffix"], np.float32)[None])
        state = _batched(env, record, 1, seed)
        first = jax.device_get(rollout(state, commands, jax.random.PRNGKey(seed)))
        second = jax.device_get(rollout(state, commands, jax.random.PRNGKey(seed)))
        exact, mismatch = active_prefix_exact(first, second)
        repeats = [_outcome(first, 0, exact, mismatch), _outcome(second, 0, exact, mismatch)]
        p0 = p0_decision(repeats); branches = []
        p1 = {"pass": False, "reasons": ["p0_not_passed"], "successes": 0, "branches": 0}
        if p0["pass"]:
            micro = _micro_states(env, record, seed + 1000)
            branch_commands = jnp.repeat(commands, 4, axis=0)
            raw = jax.device_get(rollout(micro, branch_commands, jax.random.PRNGKey(seed + 2000)))
            branches = [_outcome(raw, i) | {"perturbation_vx_vz": PERTURBATIONS[i].tolist()} for i in range(4)]
            p1 = p1_decision(p0, branches, repeats[0]["failure_type"])
        feature = np.asarray(record["physical_feature"], np.float64)
        valid = bool(record.get("had_valid_landing", False)); support = bool(record.get("contact_age", 0) > 0)
        entry_feature = entry_feature_from_physical(feature, valid_landing=valid, support=support,
                                                    contact_age=int(record.get("contact_age", 0)), cfg=cfg)
        distance, nearest, _ = normalized_nearest(entry_feature, entries, center, scale)
        layer = 1 + sum(distance > value for value in quantiles)
        region = "late" if layer == 1 else "middle" if layer in (2, 3) else "early"
        if p0["pass"]:
            artifact_hash = canonical_hash({"suffix": np.asarray(proposal["controller_suffix"]).tolist(),
                                            "source_controller": proposal["source_controller_artifact_sha256"]})
            node = BackwardTubeNode(
                node_id=hashlib.sha256(f"harvest:{proposal['physical_state_hash']}:{artifact_hash}".encode()).hexdigest()[:32],
                phase="descent", layer=layer, region=region, candidate_id=proposal["candidate_id"],
                source_state_hash=proposal["physical_state_hash"],
                physical_state={"source_artifact": str(ROOT / "trajectory_harvested_snapshots.pkl"),
                                "source_index": proposal["_artifact_index"], "snapshot_sha256": canonical_hash(record)},
                actor_observation=np.asarray(record["actor_observation_t"]).tolist(),
                parent_node_id=ids[nearest], parent_tube="canonical_C_L",
                controller_type="reused_successful_controller_suffix",
                controller_artifact_sha256=artifact_hash, entry_tick=repeats[0]["downstream_entry_tick"],
                downstream_entry_state={"qpos": np.asarray(first["entry_qpos"])[0].tolist(),
                                        "qvel": np.asarray(first["entry_qvel"])[0].tolist(),
                                        "nearest_C_L_node_id": ids[nearest]},
                final_recovery=True, p0=True, p1=bool(p1["pass"]), branch_results=tuple(branches),
                provenance_hashes={"xml": EXPECTED["xml"], "C_L": EXPECTED["C_L"],
                                   "pi_D": EXPECTED["pi_D"], "pi_L": EXPECTED["pi_L"],
                                   "source_node": proposal["source_node_id"]})
            node.validate(); new_nodes.append(node)
        certification_rows.append({"physical_state_hash": proposal["physical_state_hash"],
            "candidate_id": proposal["candidate_id"], "source_node_id": proposal["source_node_id"],
            "source_tick": proposal["source_tick"], "relative_tick": proposal["relative_tick_to_downstream_entry"],
            "P0": p0, "P1": p1, "repeats": repeats, "branches": branches,
            "distance_to_C_L": distance, "layer": layer, "region": region})
        current_p1 = old_p1 + [node for node in new_nodes if node.p1]
        feature_map = dict(old_feature)
        feature_map.update({node.node_id: np.asarray(proposals[[x["physical_state_hash"] for x in proposals].index(node.source_state_hash)]["snapshot_v4"]["physical_feature"])
                            for node in new_nodes if node.p1})
        subset, excluded = balanced_p1_launch_subset(current_p1, feature_map, feature_scale)
        gate = balanced_p1_launch_gate(subset)
        save_json(ROOT / "source_a_certification.partial.json", {"completed": len(certification_rows),
            "new_P0": sum(node.p0 for node in new_nodes), "new_P1": sum(node.p1 for node in new_nodes),
            "balanced_gate": gate})
        if gate["status"] == "PASS": break
    current_p1 = old_p1 + [node for node in new_nodes if node.p1]
    feature_map = dict(old_feature)
    by_hash = {row["physical_state_hash"]: row for row in proposals}
    feature_map.update({node.node_id: np.asarray(by_hash[node.source_state_hash]["snapshot_v4"]["physical_feature"])
                        for node in new_nodes if node.p1})
    subset, excluded = balanced_p1_launch_subset(current_p1, feature_map, feature_scale)
    gate = balanced_p1_launch_gate(subset)
    lineage = validate_parent_lineage(old_nodes + new_nodes, set(ids))
    save_json(ROOT / "source_a_certification_results.json", {
        "status": "PASS", "certified_proposals": len(certification_rows),
        "new_P0": sum(node.p0 for node in new_nodes), "new_P1": sum(node.p1 for node in new_nodes),
        "new_P1_candidates": dict(Counter(node.candidate_id for node in new_nodes if node.p1)),
        "nodes": [node.to_dict() for node in new_nodes], "rows": certification_rows,
        "lineage": lineage, "balanced_gate": gate, "heldout_used": False, "delay": False, "PPO": False,
    })
    save_json(ROOT / "balanced_p1_launch_subset_v1_final.json", {
        "status": gate["status"], "node_ids": [node.node_id for node in subset],
        "excluded": excluded, "gate": gate,
        "full_P1_nodes": len(current_p1), "balanced_subset_nodes": len(subset),
    })
    print(json.dumps({"certified": len(certification_rows), "new_P0": sum(node.p0 for node in new_nodes),
                      "new_P1": sum(node.p1 for node in new_nodes), "gate": gate}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot", "full", "certify"))
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode in {"pilot", "full"}: _harvest(args.mode)
    else: _certify()


if __name__ == "__main__":
    main()
