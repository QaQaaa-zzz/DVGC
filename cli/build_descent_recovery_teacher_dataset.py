"""Build the balanced behavior-anchor dataset for the passed Descent launch set."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from cli.freeze_descent_predecessor_assets import verify_frozen_assets
from cli.run_backward_descent_nominal_pilot import C_L, PI_D, PI_L, _load_record, _restore
from dvgc.backward_tube import BackwardTubeNode
from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.descent_predecessor import expand_residual_knots
from dvgc.policy import load_bundle
from dvgc.runtime import build_inference, save_json


PRIOR = Path("runs/backward_recovery_tube_fast_track_v1/descent_cem3_tier2/descent_cem_pilot_report.json")


def _atomic_pickle(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, pickle.HIGHEST_PROTOCOL); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _generic_record(node, root, harvested):
    artifact = Path(node.physical_state["source_artifact"])
    if artifact == root / "trajectory_harvested_snapshots.pkl":
        return harvested[int(node.physical_state["source_index"])]["snapshot_v4"]
    return _load_record(node.physical_state)


def _old_controller_rows():
    report = json.loads(PRIOR.read_text()); result = {}
    for row in report["rows"]:
        if row["P0"]["pass"]:
            result[(row["proposal"]["physical_state_sha256"], row["controller"])] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", required=True)
    args = parser.parse_args(); root = Path(args.run)
    valid, failed = verify_frozen_assets(root)
    if not valid: raise SystemExit(f"frozen asset identity failure: {failed}")
    output = root / "descent_recovery_teacher_dataset_v1_balanced.pkl"
    if output.exists(): raise SystemExit("teacher dataset already exists")
    subset = [BackwardTubeNode(**row) for row in json.loads(
        (root / "balanced_p1_launch_subset_v1_frozen.json").read_text())["nodes"]]
    full = [BackwardTubeNode(**row) for row in json.loads((root / "full_p1_bank_v1.json").read_text())["nodes"]]
    old_report = json.loads(PRIOR.read_text()); old_all = [BackwardTubeNode(**row) for row in old_report["nodes"]]
    new = {row["node_id"]: row for row in json.loads((root / "source_a_certification_results.json").read_text())["nodes"]}
    harvested = pickle.loads((root / "trajectory_harvested_snapshots.pkl").read_bytes())
    controllers = _old_controller_rows()
    dparams, policy_cfg, _ = load_bundle(PI_D, verify_files=True); lparams, _, _ = load_bundle(PI_L, verify_files=True)
    cfg = load_config("configs/default.json", {**policy_cfg, "use_bank_resets": False,
        "domain_randomization": False, "obs_noise_enable": False, "training_stage": "flight"})
    from dvgc.env import OrangeBikeDVGC
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), cert_bank=SnapshotBank.load(C_L))
    dpolicy, lpolicy = build_inference(env, dparams, deterministic=True), build_inference(env, lparams, deterministic=True)
    teacher, replay_checks = [], []
    for index, node in enumerate(subset):
        record = _generic_record(node, root, harvested)
        state = _restore(env, record, jax.random.PRNGKey(61_000_000 + index))
        base1, _ = dpolicy(state.obs, jax.random.PRNGKey(0)); base2, _ = dpolicy(state.obs, jax.random.PRNGKey(0))
        if node.node_id in new:
            source = harvested[int(node.physical_state["source_index"])]
            residual = np.asarray(source["controller_suffix"], np.float32)[0]
        else:
            row = controllers[(node.source_state_hash, node.controller_type)]
            residual = expand_residual_knots(row.get("residual_knots"))[0]
        target = np.asarray(jnp.clip(base1 + jnp.asarray(residual), -1., 1.), np.float32)
        teacher.append({"node_id": node.node_id, "candidate_id": node.candidate_id, "layer": node.layer,
            "region": node.region, "observation": np.asarray(state.obs["state"], np.float32),
            "target_action": target, "frozen_action": np.asarray(base1, np.float32),
            "controller_residual": residual, "action_order": ["steer", "drive", "hip", "knee"]})
        replay_checks.append(bool(np.array_equal(np.asarray(base1), np.asarray(base2))))
    # Candidate-balanced P0-only states anchor the original pi_D behavior.
    p0_only = [node for node in old_all if node.p0 and not node.p1]
    grouped = defaultdict(list)
    for node in sorted(p0_only, key=lambda value: (value.candidate_id, value.node_id)): grouped[node.candidate_id].append(node)
    # A small frontier rehearsal set: exactly one deterministic medoid proxy
    # (lexicographically first node) per P0-only candidate.  Do not exhaust a
    # populous candidate after sparse candidates run out.
    anchor_nodes = [grouped[candidate][0] for candidate in sorted(grouped)]
    anchors = []
    for index, node in enumerate(anchor_nodes):
        state = _restore(env, _generic_record(node, root, harvested), jax.random.PRNGKey(62_000_000 + index))
        action, _ = dpolicy(state.obs, jax.random.PRNGKey(0))
        anchors.append({"node_id": node.node_id, "candidate_id": node.candidate_id, "layer": node.layer,
            "region": node.region, "observation": np.asarray(state.obs["state"], np.float32),
            "target_action": np.asarray(action, np.float32)})
    # Downstream transition is retained as a separate frozen-pi_L contract,
    # not regressed into pi_D's head.
    downstream = []
    for index, record in enumerate([row for row in SnapshotBank.load(C_L).records if row["final"]["label"] == "safe"][:16]):
        state = _restore(env, record, jax.random.PRNGKey(63_000_000 + index)); action, _ = lpolicy(state.obs, jax.random.PRNGKey(0))
        downstream.append({"entry_id": record["id"], "observation": np.asarray(state.obs["state"], np.float32),
                           "pi_L_action": np.asarray(action, np.float32)})
    payload = {"teacher": teacher, "anchors": anchors, "downstream_pi_L_transition": downstream,
               "normalizer_sha256": file_sha256(PI_D / "params.pkl"), "heldout_used": False, "delay": False}
    _atomic_pickle(output, payload)
    counts = Counter(row["candidate_id"] for row in teacher)
    anchor_counts = Counter(row["candidate_id"] for row in anchors)
    audit = {"status": "PASS" if all(replay_checks) and len(teacher) == 16 and max(counts.values()) / len(teacher) <= .35 and max(anchor_counts.values()) / len(anchors) <= .35 else "FAIL",
        "teacher_samples": len(teacher), "anchor_samples": len(anchors), "downstream_transition_samples": len(downstream),
        "candidate_counts": dict(sorted(counts.items())), "maximum_candidate_share": max(counts.values()) / len(teacher),
        "anchor_candidate_counts": dict(sorted(anchor_counts.items())),
        "anchor_maximum_candidate_share": max(anchor_counts.values()) / len(anchors),
        "layers": sorted({row["layer"] for row in teacher}), "regions": sorted({row["region"] for row in teacher}),
        "teacher_action_replay_exact": all(replay_checks), "action_order": ["steer", "drive", "hip", "knee"],
        "normalizer_fixed": True, "dataset": str(output), "dataset_sha256": file_sha256(output),
        "downstream_actions_used_as_pi_D_targets": False, "heldout_used": False, "delay": False,
        "supersedes_for_training": str(root / "descent_recovery_teacher_dataset_v1.pkl") if (root / "descent_recovery_teacher_dataset_v1.pkl").exists() else None,
    }
    save_json(root / "descent_recovery_teacher_dataset_v1_balanced_audit.json", audit)
    if audit["status"] != "PASS": raise SystemExit(f"teacher dataset gate failed: {audit}")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__": main()
