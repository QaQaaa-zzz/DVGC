"""Extract a provenance-locked, phase-balanced teacher dataset for consolidation.

Parameterized experts are evaluated only on the authoritative actor input saved
with each reset state.  The non-parametric Apex expert contributes a real
certified feedback sequence selected by medoid; synthetic averages are never
used as action labels.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

from dvgc.bank import SnapshotBank
from dvgc.config import file_sha256, load_config
from dvgc.runtime import save_json


STAGES = ("takeoff", "ascent", "apex", "descent", "landing")
PARAMETRIC_STAGES = {"takeoff", "ascent", "descent", "landing"}


def successful_sequence_medoid(evidence: list[dict]) -> tuple[np.ndarray, dict]:
    """Return one observed sequence, never an average of successful modes."""
    usable = []
    for row in evidence:
        sequence = row.get("action_sequence")
        if sequence is None:
            first = row.get("first_action")
            if first is None:
                continue
            sequence = [first]
        array = np.asarray(sequence, np.float32)
        if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
            continue
        usable.append((row, array))
    if not usable:
        raise ValueError("Apex state has no finite certified teacher action sequence")
    length = min(array.shape[0] for _, array in usable)
    flattened = np.stack([array[:length].reshape(-1) for _, array in usable])
    distances = np.linalg.norm(flattened[:, None, :] - flattened[None, :, :], axis=-1)
    index = int(np.argmin(distances.sum(axis=1)))
    selected_row, selected = usable[index]
    first_actions = np.stack([array[0] for _, array in usable])
    signs = np.sign(first_actions)
    nonzero = np.abs(first_actions) > 1e-6
    majority = np.sign(np.sum(signs, axis=0))
    agreement = np.mean((signs == majority) | ~nonzero, axis=0)
    audit = {
        "sequence_count": len(usable),
        "comparison_ticks": length,
        "selected_index": index,
        "selected_branch_index": selected_row.get("branch_index"),
        "selected_seed": selected_row.get("seed"),
        "selected_dynamics_variant": selected_row.get("dynamics_variant"),
        "first_action_coordinate_variance": np.var(first_actions, axis=0).tolist(),
        "first_action_direction_agreement": agreement.tolist(),
        "max_pairwise_sequence_l2": float(np.max(distances)),
        "mean_action_replay_forbidden": True,
    }
    return selected, audit


def _actor_observation(record: dict) -> np.ndarray:
    value = np.asarray(record.get("policy_state", {}).get("actor_observation"), np.float32)
    if value.shape != (140,) or not np.isfinite(value).all():
        raise ValueError(f"{record.get('id')} lacks a finite authoritative 140D actor observation")
    return value


def build_examples(
    records: list[dict],
    *,
    policy_actions: dict[str, Callable[[np.ndarray], np.ndarray]],
    allowed_policy_paths: set[str],
) -> tuple[list[dict], list[dict]]:
    examples, apex_audits = [], []
    for record in records:
        stage = str(record.get("phase_rsi_stage"))
        if stage not in STAGES:
            raise ValueError(f"invalid or absent phase_rsi_stage: {stage}")
        observation = _actor_observation(record)
        teacher = {}
        if stage == "apex":
            evidence = list(record.get("certified_teacher_action_evidence", []))
            expected = int(record.get("independent_branch_count", 0))
            seeds = [row.get("seed") for row in evidence]
            if expected != 32 or len(evidence) != expected or None in seeds or len(seeds) != len(set(seeds)):
                raise ValueError(
                    f"{record.get('id')} lacks complete unique 32-branch Apex teacher evidence"
                )
            sequence, multimodality = successful_sequence_medoid(evidence)
            action = sequence[0]
            teacher = {
                "teacher_type": "certified_feedback_sequence_medoid",
                "teacher_branch_index": multimodality["selected_branch_index"],
                "teacher_seed": multimodality["selected_seed"],
                "teacher_dynamics_variant": multimodality["selected_dynamics_variant"],
                "teacher_sequence": sequence,
            }
            apex_audits.append({"record_id": record["id"], **multimodality})
        else:
            if stage in {"takeoff", "ascent"}:
                path = str(record.get("selected_controller_path", ""))
            else:
                candidates = sorted(path for path in policy_actions if path.endswith(f"::{stage}"))
                if len(candidates) != 1:
                    raise ValueError(f"expected exactly one frozen {stage} policy, got {candidates}")
                path = candidates[0].rsplit("::", 1)[0]
            if path not in allowed_policy_paths or path not in policy_actions:
                raise ValueError(f"{stage} record references unaudited policy {path!r}")
            action = np.asarray(policy_actions[path](observation[None, :]), np.float32)[0]
            teacher = {"teacher_type": "frozen_policy_mode", "teacher_policy_path": path}
        action = np.asarray(action, np.float32)
        if action.shape != (4,) or not np.isfinite(action).all() or np.max(np.abs(action)) > 1.000001:
            raise ValueError(f"invalid teacher action for {record['id']}: {action}")
        examples.append({
            "record_id": str(record["id"]),
            "origin_record_id": str(record.get("origin_record_id", record["id"])),
            "phase": stage,
            "parent_id": str(record["reset_parent_id"]),
            "source_bank_sha256": str(record["origin_artifact_sha256"]),
            "source_artifact_role": str(record["origin_artifact_role"]),
            "observation": observation,
            "action": action,
            "training_weight": float(record["reset_weight"]),
            **teacher,
        })
    return examples, apex_audits


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


def build_frozen_policy_actions(
    policy_rows: list[dict], env, *, load_policy, build_tools, hash_params
) -> tuple[dict[str, Callable[[np.ndarray], np.ndarray]], dict[str, str]]:
    """Build one evaluator per complete expert bundle, including its normalizer."""
    actions, hashes = {}, {}
    for row in policy_rows:
        path = str(row["policy_path"])
        params, _, _ = load_policy(path)
        actual_hash = hash_params(path)
        if actual_hash != row["params_sha256"]:
            raise ValueError(f"expert params changed after compatibility audit: {path}")
        _, expert_action, _ = build_tools(env, params)
        evaluator = lambda obs, action_fn=expert_action, actor=params[1]: np.asarray(
            action_fn(actor, obs)
        )
        actions[path] = evaluator
        actions[f"{path}::{row['stage']}"] = evaluator
        hashes[path] = actual_hash
    return actions, hashes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-bank", required=True)
    parser.add_argument("--expert-compatibility", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    output, report_path = Path(args.output_dataset), Path(args.output_report)
    if output.exists() or report_path.exists():
        raise SystemExit("refusing to overwrite phase-balanced teacher dataset")
    bank = SnapshotBank.load(args.phase_bank)
    if (bank.metadata.get("artifact_role") != "phase_balanced_tube_rsi_reset_bank"
            or bank.metadata.get("formal_tube_or_jel") is not False):
        raise SystemExit("input is not the provenance-locked phase-balanced reset bank")
    compatibility = json.loads(Path(args.expert_compatibility).read_text())
    if (compatibility.get("status") != "PASS"
            or compatibility.get("shared_actor_distillation_authorized") is not True):
        raise SystemExit("frozen expert compatibility audit is not PASS")

    # Import JAX/runtime modules only in the executable path; pure tests remain CPU-only.
    import jax
    from dvgc.descent_supervised import build_actor_tools
    from dvgc.env import OrangeBikeDVGC
    from dvgc.policy import load_bundle

    policy_rows = compatibility["parameterized_experts"]
    allowed_paths = {str(row["policy_path"]) for row in policy_rows}
    if not allowed_paths:
        raise SystemExit("compatibility audit contains no parameterized experts")
    _first_params, first_cfg, _ = load_bundle(policy_rows[0]["policy_path"], verify_files=True)
    cfg = load_config(overrides={
        **first_cfg, "use_bank_resets": False, "domain_randomization": False,
        "obs_noise_enable": False,
    })
    env = OrangeBikeDVGC(cfg, snapshot_bank=SnapshotBank())
    try:
        policy_actions, policy_hashes = build_frozen_policy_actions(
            policy_rows, env,
            load_policy=lambda path: load_bundle(path, verify_files=True),
            build_tools=build_actor_tools,
            hash_params=lambda path: file_sha256(Path(path) / "params.pkl"),
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    examples, apex_audits = build_examples(
        bank.records, policy_actions=policy_actions, allowed_policy_paths=allowed_paths
    )
    masses = {stage: sum(row["training_weight"] for row in examples if row["phase"] == stage)
              for stage in STAGES}
    if any(not np.isclose(masses[stage], .2, atol=1e-7) for stage in STAGES):
        raise SystemExit(f"teacher dataset is not phase balanced: {masses}")
    payload = {
        "schema": "dvgc_phase_balanced_distillation_teacher_v1",
        "artifact_role": "phase_balanced_distillation_teacher_dataset",
        "formal_tube_or_jel": False,
        "phase_bank_sha256": file_sha256(args.phase_bank),
        "expert_compatibility_sha256": file_sha256(args.expert_compatibility),
        "expert_params_sha256s": policy_hashes,
        "examples": examples,
    }
    _atomic_pickle(output, payload)
    report = {
        "status": "PASS", "artifact_role": payload["artifact_role"],
        "formal_tube_or_jel": False, "PPO_authorization": False,
        "examples": len(examples), "phase_counts": dict(Counter(row["phase"] for row in examples)),
        "phase_weight_mass": masses, "policy_example_counts": dict(Counter(
            row.get("teacher_policy_path", "apex_feedback_medoid") for row in examples)),
        "apex_multimodality_audit": apex_audits,
        "output_dataset": str(output), "output_dataset_sha256": file_sha256(output),
        "phase_bank_sha256": payload["phase_bank_sha256"],
        "expert_compatibility_sha256": payload["expert_compatibility_sha256"],
    }
    save_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key != "apex_multimodality_audit"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
