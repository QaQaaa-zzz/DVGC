"""Capture exact-replay Apex feedback prefixes as trajectory teacher data.

The frozen Apex support stores the actions actually selected by the bounded
receding-horizon controller.  A sequence is admitted here only when the same
audited branch reaches the local Descent event at the same tick in two exact
replays.  Only the pre-entry prefix is exported; post-entry/failure actions are
not teacher labels.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from pathlib import Path

import jax
import mujoco
import numpy as np

from cli.build_phase_balanced_teacher_dataset import successful_sequence_medoid
from cli.discover_apex_feedback_bridge import (
    TERMINAL_FEATURES, TERMINAL_INDEX, _state_score,
)
from cli.runtime_gate import source_fingerprint
from dvgc.bank import SnapshotBank
from dvgc.certification import DYNAMICS_VARIANTS
from dvgc.config import file_sha256, load_config
from dvgc.env import OrangeBikeDVGC
from dvgc.policy import load_bundle
from dvgc.rollout import restore_snapshot
from dvgc.runtime import save_json


def _atomic_pickle(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _selected_evidence(record: dict) -> tuple[dict, np.ndarray, dict]:
    evidence = list(record.get("certified_teacher_action_evidence", []))
    sequence, audit = successful_sequence_medoid(evidence)
    selected = [row for row in evidence if (
        row.get("branch_index") == audit["selected_branch_index"]
        and row.get("seed") == audit["selected_seed"]
        and row.get("dynamics_variant") == audit["selected_dynamics_variant"]
    )]
    if len(selected) != 1:
        raise ValueError(f"{record.get('id')} has ambiguous selected Apex evidence")
    return selected[0], sequence, audit


def _matching_audit_branch(label: dict, evidence: dict, sequence: np.ndarray) -> dict:
    matches = [row for row in label.get("branches", []) if (
        row.get("branch_index") == evidence.get("branch_index")
        and row.get("seed") == evidence.get("seed")
        and row.get("dynamics_variant") == evidence.get("dynamics_variant")
    )]
    if len(matches) != 1 or matches[0].get("success") is not True:
        raise ValueError(f"selected Apex evidence is absent or unsuccessful for {label.get('candidate_id')}")
    audited = np.asarray(matches[0].get("action_sequence"), np.float32)
    if audited.shape != sequence.shape or not np.array_equal(audited, sequence):
        raise ValueError(f"selected Apex action sequence differs from audit for {label.get('candidate_id')}")
    return matches[0]


def _replay_prefix(env, step, model, support_metadata, target, center, scale,
                   record: dict, evidence: dict, sequence: np.ndarray) -> dict:
    state = restore_snapshot(env, record, jax.random.PRNGKey(int(evidence["seed"])))
    previous_vz = float(np.asarray(state.data.qvel[2]))
    samples = []; stable_count = 0; entry_tick = None
    for tick, action in enumerate(sequence, 1):
        observation = np.asarray(jax.device_get(state.obs["state"]), np.float32)
        action = np.asarray(action, np.float32)
        state = step(state, action)
        _, diagnostic = _state_score(
            env, state, previous_vz, support_metadata, model, target, center, scale,
            float(np.sum(np.square(action))),
        )
        samples.append({
            "tick": tick - 1,
            "observation": observation,
            "action": action,
        })
        stable_count = stable_count + 1 if diagnostic["stable"] else 0
        if stable_count >= 4:
            entry_tick = tick
            break
        if diagnostic["done"]:
            break
        previous_vz = float(diagnostic["feature"][8])
    return {"entry_tick": entry_tick, "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-bank", required=True)
    parser.add_argument("--apex-audit-labels", required=True)
    parser.add_argument("--support-bank", required=True)
    parser.add_argument("--terminal-bank", required=True)
    parser.add_argument("--descent-policy", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--config", default="configs/default.json")
    args = parser.parse_args()
    output, report_path = Path(args.output_dataset), Path(args.output_report)
    if output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite Apex trajectory teacher artifact")
    gate = json.loads(Path("docs/RUNTIME_GATE.json").read_text())
    if gate.get("status") != "PASS" or gate.get("source_fingerprint") != source_fingerprint(Path.cwd()):
        raise SystemExit("runtime gate stale")

    phase_bank = SnapshotBank.load(args.phase_bank)
    records = [row for row in phase_bank.records if row.get("phase_rsi_stage") == "apex"]
    labels_payload = json.loads(Path(args.apex_audit_labels).read_text())
    labels = {str(row["candidate_id"]): row for row in labels_payload.get("labels", [])}
    origin_ids = {str(row.get("origin_record_id", row["id"])) for row in records}
    if not records or set(labels) != origin_ids:
        raise SystemExit("Apex phase-bank IDs do not exactly match independent audit labels")
    support = SnapshotBank.load(args.support_bank)
    support_metadata = dict(support.metadata)
    support_metadata["support_features"] = [row["physical_feature"] for row in support.records]
    terminal = SnapshotBank.load(args.terminal_bank)
    center = np.asarray(terminal.metadata["normalization_center"], float)
    scale = np.asarray(terminal.metadata["normalization_scale"], float)
    target = np.asarray([[
        (row["physical_feature"][TERMINAL_INDEX[name]] - center[i]) / scale[i]
        for i, name in enumerate(TERMINAL_FEATURES)
    ] for row in terminal.records], float)
    _, policy_cfg, _ = load_bundle(args.descent_policy, verify_files=True)
    variants = {row["id"]: row for row in DYNAMICS_VARIANTS}
    runtimes = {}

    def runtime(variant_id: str):
        if variant_id not in variants:
            raise ValueError(f"unknown audited dynamics variant {variant_id!r}")
        if variant_id not in runtimes:
            variant = variants[variant_id]
            overrides = {key: value for key, value in variant.items() if key != "id"}
            cfg = load_config(args.config, {
                **policy_cfg, **overrides, "training_stage": "flight",
                "use_bank_resets": False, "domain_randomization": False,
                "obs_noise_enable": False, "stage_reachability_objective": "",
            })
            env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank(), stage_support_bank=support)
            runtimes[variant_id] = (env, jax.jit(env.step))
        return runtimes[variant_id]

    model = mujoco.MjModel.from_xml_path(str(load_config(args.config).xml_path))
    trajectories = []
    for record in records:
        evidence, sequence, medoid = _selected_evidence(record)
        origin_id = str(record.get("origin_record_id", record["id"]))
        audited = _matching_audit_branch(labels[origin_id], evidence, sequence)
        env, step = runtime(str(evidence["dynamics_variant"]))
        first = _replay_prefix(env, step, model, support_metadata, target, center, scale,
                               record, evidence, sequence)
        second = _replay_prefix(env, step, model, support_metadata, target, center, scale,
                                record, evidence, sequence)
        expected_tick = int(audited["time_to_next_stage"])
        exact = (first["entry_tick"] == second["entry_tick"] == expected_tick
                 and len(first["samples"]) == len(second["samples"])
                 and all(np.array_equal(a["observation"], b["observation"])
                         and np.array_equal(a["action"], b["action"])
                         for a, b in zip(first["samples"], second["samples"])))
        if not exact:
            raise SystemExit(f"Apex medoid exact replay failed for {record['id']}")
        trajectories.append({
            "record_id": str(record["id"]),
            "parent_id": str(record["reset_parent_id"]),
            "branch_index": evidence["branch_index"], "seed": evidence["seed"],
            "dynamics_variant": evidence["dynamics_variant"],
            "entry_tick": expected_tick, "exact_replay_count": 2,
            "local_entry_replay_verified": True,
            "medoid_audit": medoid, "samples": first["samples"],
        })
    payload = {
        "schema": "dvgc_apex_feedback_trajectory_teacher_v1",
        "artifact_role": "certified_local_feedback_trajectory_teacher",
        "formal_tube_or_jel": False, "PPO_authorization": False,
        "phase_bank_sha256": file_sha256(args.phase_bank),
        "apex_audit_labels_sha256": file_sha256(args.apex_audit_labels),
        "support_bank_sha256": file_sha256(args.support_bank),
        "terminal_bank_sha256": file_sha256(args.terminal_bank),
        "descent_policy_params_sha256": file_sha256(Path(args.descent_policy) / "params.pkl"),
        "xml_sha256": file_sha256(load_config(args.config).xml_path),
        "source_fingerprint": gate["source_fingerprint"],
        "trajectories": trajectories,
    }
    _atomic_pickle(output, payload)
    report = {
        "status": "PASS", "artifact_role": payload["artifact_role"],
        "formal_tube_or_jel": False, "PPO_authorization": False,
        "states": len(trajectories),
        "exact_replay_states": sum(row["local_entry_replay_verified"] for row in trajectories),
        "trajectory_examples": sum(len(row["samples"]) for row in trajectories),
        "entry_ticks": [row["entry_tick"] for row in trajectories],
        "phase_bank_sha256": payload["phase_bank_sha256"],
        "apex_audit_labels_sha256": payload["apex_audit_labels_sha256"],
        "output_dataset": str(output), "output_dataset_sha256": file_sha256(output),
    }
    save_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
