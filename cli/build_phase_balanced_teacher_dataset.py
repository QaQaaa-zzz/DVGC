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
    policy_identities: dict[str, str] | None = None,
    apex_trajectories: dict[str, dict] | None = None,
) -> tuple[list[dict], list[dict]]:
    policy_identities = policy_identities or {}
    examples, apex_audits = [], []
    for record in records:
        stage = str(record.get("phase_rsi_stage"))
        if stage not in STAGES:
            raise ValueError(f"invalid or absent phase_rsi_stage: {stage}")
        observation = _actor_observation(record)
        teacher = {}
        if stage == "apex":
            if apex_trajectories is not None:
                trajectory = apex_trajectories.get(str(record["id"]))
                if (trajectory is None
                        or trajectory.get("local_entry_replay_verified") is not True
                        or int(trajectory.get("exact_replay_count", 0)) < 2):
                    raise ValueError(f"{record.get('id')} lacks exact-replay Apex trajectory evidence")
                samples = list(trajectory.get("samples", []))
                if not samples or [int(row.get("tick", -1)) for row in samples] != list(range(len(samples))):
                    raise ValueError(f"{record.get('id')} has invalid Apex trajectory ticks")
                weight = float(record["reset_weight"]) / len(samples)
                for sample in samples:
                    sample_observation = np.asarray(sample.get("observation"), np.float32)
                    sample_action = np.asarray(sample.get("action"), np.float32)
                    if (sample_observation.shape != (140,) or not np.isfinite(sample_observation).all()
                            or sample_action.shape != (4,) or not np.isfinite(sample_action).all()
                            or np.max(np.abs(sample_action)) > 1.000001):
                        raise ValueError(f"invalid Apex trajectory sample for {record['id']}")
                    examples.append({
                        "record_id": f"{record['id']}:tick:{sample['tick']}",
                        "origin_record_id": str(record.get("origin_record_id", record["id"])),
                        "phase": stage, "parent_id": str(record["reset_parent_id"]),
                        "source_bank_sha256": str(record["origin_artifact_sha256"]),
                        "source_artifact_role": str(record["origin_artifact_role"]),
                        "observation": sample_observation, "action": sample_action,
                        "training_weight": weight,
                        "teacher_type": "certified_feedback_trajectory_medoid",
                        "teacher_branch_index": trajectory.get("branch_index"),
                        "teacher_seed": trajectory.get("seed"),
                        "teacher_dynamics_variant": trajectory.get("dynamics_variant"),
                        "teacher_trajectory_tick": int(sample["tick"]),
                        "teacher_entry_tick": int(trajectory["entry_tick"]),
                    })
                apex_audits.append({
                    "record_id": record["id"], "trajectory_examples": len(samples),
                    "entry_tick": int(trajectory["entry_tick"]),
                    "exact_replay_count": int(trajectory["exact_replay_count"]),
                })
                continue
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
            source_identity = record.get("policy_identity_hash")
            if (stage == "descent" and source_identity is not None
                    and policy_identities.get(path) != f"adapter:{source_identity}"):
                raise ValueError(
                    f"Descent teacher identity does not match certified Tube controller: "
                    f"{policy_identities.get(path)!r} != adapter:{source_identity}"
                )
            action = np.asarray(policy_actions[path](observation[None, :]), np.float32)[0]
            teacher = {
                "teacher_type": ("frozen_policy_with_adapter_mode"
                                 if policy_identities.get(path, "").startswith("adapter:")
                                 else "frozen_policy_mode"),
                "teacher_policy_path": path,
                "teacher_controller_identity": policy_identities.get(path),
                "source_certified_controller_identity": source_identity,
            }
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
    policy_rows: list[dict], env, *, load_policy, build_tools, hash_params,
    load_adapter=None,
) -> tuple[dict[str, Callable[[np.ndarray], np.ndarray]], dict[str, str], dict[str, str]]:
    """Build one evaluator per complete expert bundle, including its normalizer."""
    actions, hashes, identities = {}, {}, {}
    for row in policy_rows:
        path = str(row["policy_path"])
        params, _, manifest = load_policy(path)
        actual_hash = hash_params(path)
        if actual_hash != row["params_sha256"]:
            raise ValueError(f"expert params changed after compatibility audit: {path}")
        _, expert_action, _ = build_tools(env, params)
        evaluator = lambda obs, action_fn=expert_action, actor=params[1]: np.asarray(
            action_fn(actor, obs)
        )
        identity = f"params:{actual_hash}"
        if manifest.get("adapter_sha256") is not None:
            if load_adapter is None:
                raise ValueError(f"expert {path} declares an adapter but no verified loader is available")
            adapter, adapter_identity = load_adapter(path, manifest, actual_hash)
            base_evaluator = evaluator
            evaluator = lambda obs, base=base_evaluator, correction=adapter: np.asarray(
                correction(obs, base(obs))
            )
            identity = f"adapter:{adapter_identity}"
        actions[path] = evaluator
        actions[f"{path}::{row['stage']}"] = evaluator
        hashes[path] = actual_hash
        identities[path] = identity
    return actions, hashes, identities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-bank", required=True)
    parser.add_argument("--expert-compatibility", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--apex-trajectory-teacher", default="")
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
    from dvgc.backward_search import (
        compact_observation_command_adapter, compact_observation_residual_adapter,
    )
    import jax.numpy as jnp

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
        def load_verified_adapter(path, manifest, base_hash):
            adapter_path = Path(path).parent / "adapter.pkl"
            if not adapter_path.is_file() or file_sha256(adapter_path) != manifest["adapter_sha256"]:
                raise ValueError(f"frozen adapter identity mismatch: {adapter_path}")
            with adapter_path.open("rb") as stream:
                artifact = pickle.load(stream)
            if (artifact.get("base_policy_sha256") != base_hash
                    or artifact.get("policy_identity_hash") != manifest.get("policy_identity_hash")):
                raise ValueError(f"adapter/base-policy provenance mismatch: {adapter_path}")
            common = (
                jnp.asarray(artifact["prototypes"]), jnp.asarray(artifact["targets"]),
                jnp.asarray(artifact["normalizer_mean"]), jnp.asarray(artifact["normalizer_std"]),
                float(artifact["radius"]),
            )
            if artifact.get("command_source") == "recorded":
                adapter = compact_observation_command_adapter(
                    *common, float(artifact.get("core_radius", 0.0))
                )
            else:
                adapter = compact_observation_residual_adapter(*common)
            return adapter, artifact["policy_identity_hash"]

        policy_actions, policy_hashes, policy_identities = build_frozen_policy_actions(
            policy_rows, env,
            load_policy=lambda path: load_bundle(path, verify_files=True),
            build_tools=build_actor_tools,
            hash_params=lambda path: file_sha256(Path(path) / "params.pkl"),
            load_adapter=load_verified_adapter,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    apex_trajectories = None
    apex_trajectory_sha256 = None
    if args.apex_trajectory_teacher:
        trajectory_path = Path(args.apex_trajectory_teacher)
        with trajectory_path.open("rb") as stream:
            trajectory_payload = pickle.load(stream)
        if (trajectory_payload.get("schema") != "dvgc_apex_feedback_trajectory_teacher_v1"
                or trajectory_payload.get("artifact_role") != "certified_local_feedback_trajectory_teacher"
                or trajectory_payload.get("formal_tube_or_jel") is not False
                or trajectory_payload.get("phase_bank_sha256") != file_sha256(args.phase_bank)):
            raise SystemExit("Apex trajectory teacher provenance/schema mismatch")
        trajectories = list(trajectory_payload.get("trajectories", []))
        apex_trajectories = {str(row["record_id"]): row for row in trajectories}
        expected_apex_ids = {str(row["id"]) for row in bank.records
                             if row.get("phase_rsi_stage") == "apex"}
        if len(apex_trajectories) != len(trajectories) or set(apex_trajectories) != expected_apex_ids:
            raise SystemExit("Apex trajectory teacher record IDs are incomplete or duplicated")
        apex_trajectory_sha256 = file_sha256(trajectory_path)

    examples, apex_audits = build_examples(
        bank.records, policy_actions=policy_actions, allowed_policy_paths=allowed_paths,
        policy_identities=policy_identities, apex_trajectories=apex_trajectories,
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
        "expert_controller_identities": policy_identities,
        "apex_trajectory_teacher_sha256": apex_trajectory_sha256,
        "examples": examples,
    }
    _atomic_pickle(output, payload)
    report = {
        "status": "PASS", "artifact_role": payload["artifact_role"],
        "formal_tube_or_jel": False, "PPO_authorization": False,
        "examples": len(examples), "phase_counts": dict(Counter(row["phase"] for row in examples)),
        "phase_weight_mass": masses, "policy_example_counts": dict(Counter(
            row.get("teacher_policy_path", "apex_feedback_medoid") for row in examples)),
        "teacher_type_counts": dict(Counter(row["teacher_type"] for row in examples)),
        "expert_controller_identities": policy_identities,
        "apex_multimodality_audit": apex_audits,
        "output_dataset": str(output), "output_dataset_sha256": file_sha256(output),
        "phase_bank_sha256": payload["phase_bank_sha256"],
        "expert_compatibility_sha256": payload["expert_compatibility_sha256"],
        "apex_trajectory_teacher_sha256": apex_trajectory_sha256,
    }
    save_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items()
                      if key != "apex_multimodality_audit"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
