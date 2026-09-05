"""Run predeclaration and closed interaction accounting."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import mediapy as media
import numpy as np

from .config import canonical_sha256, file_sha256, resolve_config_payload
from .constants import REWARD_COMPONENT_KEYS


@dataclass(frozen=True)
class InteractionAccounting:
    training: int
    brax_evaluation: int
    fixed_evaluation: int
    diagnostic: int

    def __post_init__(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("interaction counts must be nonnegative")

    @property
    def total(self) -> int:
        return sum(asdict(self).values())


@dataclass(frozen=True)
class RunDeclaration:
    run_id: str
    purpose: str
    output_dir: Path
    config_sha256: str
    xml_sha256: str
    reference_sha256: str
    training_transition_ceiling: int
    stopping_conditions: tuple[str, ...]
    resume_command: str
    parent_checkpoint: str | None = None
    starting_training_transition: int = 0
    resume_semantics: str = "fresh"
    segment_seed: int | None = None


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def predeclare_run(
    declaration: RunDeclaration, *, resolved_config: Mapping[str, Any]
) -> Path:
    if not declaration.run_id or Path(declaration.run_id).name != declaration.run_id:
        raise ValueError("run_id must be one safe path component")
    if declaration.training_transition_ceiling <= 0:
        raise ValueError("training_transition_ceiling must be positive")
    if declaration.starting_training_transition < 0:
        raise ValueError("starting training transition must be nonnegative")
    if declaration.starting_training_transition == 0:
        if declaration.parent_checkpoint is not None:
            raise ValueError("fresh run must not declare a parent checkpoint")
        if declaration.resume_semantics != "fresh":
            raise ValueError("fresh run must declare fresh resume semantics")
    else:
        if not declaration.parent_checkpoint:
            raise ValueError("warm start requires a parent checkpoint")
        if declaration.resume_semantics != "parameter_warm_start_optimizer_reset":
            raise ValueError("warm start resume semantics must declare optimizer reset")
        if declaration.segment_seed is None:
            raise ValueError("warm start requires a segment seed")
    run_dir = Path(declaration.output_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = asdict(declaration)
    manifest["output_dir"] = str(run_dir.resolve())
    manifest["stopping_conditions"] = list(declaration.stopping_conditions)
    is_formal = declaration.purpose == "formal_propulsion_ascent_ppo"
    manifest["claim_boundary"] = {
        "engineering_integrity_only": not is_formal,
        "formal_training_evidence": is_formal,
        "learnability_claim": False,
        "trained_expert_claim": False,
        "tube_or_safety_claim": False,
    }
    _atomic_json(run_dir / "run_manifest.json", manifest)
    _atomic_json(run_dir / "resolved_config.json", dict(resolved_config))
    _atomic_json(run_dir / "status.json", {"status": "predeclared"})
    (run_dir / "resume_command.txt").write_text(
        declaration.resume_command.rstrip() + "\n", encoding="utf-8"
    )
    return run_dir


def mark_run_running(
    run_dir: Path,
    *,
    process_id: int,
    metadata: Mapping[str, Any],
) -> None:
    if process_id <= 0:
        raise ValueError("process_id must be positive")
    if any(key in {"status", "process_id"} for key in metadata):
        raise ValueError("running metadata contains a reserved field")
    status_path = Path(run_dir) / "status.json"
    current = json.loads(status_path.read_text(encoding="utf-8"))
    if current != {"status": "predeclared"}:
        raise ValueError("run is not predeclared")
    _atomic_json(
        status_path,
        {"status": "running", "process_id": int(process_id), **dict(metadata)},
    )


def close_run(
    run_dir: Path,
    *,
    status: str,
    accounting: InteractionAccounting,
    reason: str,
) -> None:
    if status not in {"completed", "engineering_error", "aborted"}:
        raise ValueError("run status is not terminal")
    path = Path(run_dir)
    current = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if current.get("status") not in {"predeclared", "running"}:
        raise ValueError("run is already closed")
    _atomic_json(
        path / "status.json",
        {
            "status": status,
            "reason": reason,
            "interaction_accounting": asdict(accounting),
            "environment_transitions": accounting.total,
        },
    )


def _payload_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_reset_source(
    payload: Mapping[str, Any], *, expected_airborne_rsi: bool, context: str
) -> None:
    key = "metric__reset__slash__source_airborne_rsi"
    if key not in payload:
        raise ValueError(f"{context} reset source metric is missing")
    values = np.asarray(payload[key], dtype=np.float64)
    expected = 1.0 if expected_airborne_rsi else 0.0
    if values.ndim != 1 or values.size == 0 or not np.all(values == expected):
        raise ValueError(f"{context} reset source mismatch")


def _verify_episode_npz(
    npz_path: Path,
    *,
    captured_state_count: int,
    expected_airborne_rsi: bool,
    context: str,
) -> None:
    vector_widths = {"qpos": 12, "qvel": 11, "ctrl": 4, "action": 4}
    scalar_names = (
        "reward",
        "terminated",
        "truncated",
        "end_code",
        "success",
        "physical_failure",
        "timeout",
    )
    reward_names = tuple(
        f"reward_component__{name}" for name in REWARD_COMPONENT_KEYS
    )
    with np.load(npz_path, allow_pickle=False) as payload:
        for name, width in vector_widths.items():
            if name not in payload.files:
                raise ValueError(f"{context} required array is missing: {name}")
            if payload[name].shape != (captured_state_count, width):
                raise ValueError(f"{context} {name} sample count mismatch")
            if not np.isfinite(payload[name]).all():
                raise ValueError(f"{context} {name} contains nonfinite values")
        for name in scalar_names + reward_names:
            if name not in payload.files:
                raise ValueError(f"{context} required array is missing: {name}")
            if payload[name].shape != (captured_state_count,):
                raise ValueError(f"{context} {name} sample count mismatch")
        _verify_reset_source(
            payload,
            expected_airborne_rsi=expected_airborne_rsi,
            context=context,
        )
        reset_key = "metric__reset__slash__source_airborne_rsi"
        if payload[reset_key].shape != (captured_state_count,):
            raise ValueError(f"{context} reset source sample count mismatch")


def _verify_diagnostic_npz(
    npz_path: Path,
    *,
    captured_state_count: int,
    context: str,
    expected_airborne_rsi: bool | None = None,
    episode_npz_path: Path | None = None,
) -> None:
    vector_widths = {"qpos": 12, "qvel": 11, "ctrl": 4, "action": 4}
    scalar_names = (
        "time_seconds",
        "reward_clipped",
        "reward_unclipped",
        "reward_scaled",
        "terminal_terminated",
        "terminal_truncated",
        "terminal_end_code",
        "terminal_success",
        "terminal_physical_failure",
        "terminal_timeout",
    ) + tuple(f"reward_component__{name}" for name in REWARD_COMPONENT_KEYS)
    with np.load(npz_path, allow_pickle=False) as payload:
        for name, width in vector_widths.items():
            if name not in payload.files or payload[name].shape != (
                captured_state_count,
                width,
            ):
                raise ValueError(f"{context} {name} sample count mismatch")
        for name in scalar_names:
            if name not in payload.files or payload[name].shape != (
                captured_state_count,
            ):
                raise ValueError(f"{context} {name} sample count mismatch")
        if expected_airborne_rsi is not None:
            reset_key = "metric__reset__source_airborne_rsi"
            if reset_key not in payload.files or payload[reset_key].shape != (
                captured_state_count,
            ):
                raise ValueError(f"{context} reset source sample count mismatch")
            expected = 1.0 if expected_airborne_rsi else 0.0
            if not np.all(payload[reset_key] == expected):
                raise ValueError(f"{context} reset source mismatch")
        if episode_npz_path is not None:
            pairs = {
                "qpos": "qpos",
                "qvel": "qvel",
                "ctrl": "ctrl",
                "action": "action",
                "reward_clipped": "reward",
                "terminal_terminated": "terminated",
                "terminal_truncated": "truncated",
                "terminal_end_code": "end_code",
                "terminal_success": "success",
                "terminal_physical_failure": "physical_failure",
                "terminal_timeout": "timeout",
            }
            pairs.update(
                {
                    f"reward_component__{name}": f"reward_component__{name}"
                    for name in REWARD_COMPONENT_KEYS
                }
            )
            with np.load(episode_npz_path, allow_pickle=False) as episode:
                for diagnostic_name, episode_name in pairs.items():
                    if episode_name not in episode.files or not np.array_equal(
                        payload[diagnostic_name], episode[episode_name]
                    ):
                        raise ValueError(
                            f"{context} does not match representative episode: "
                            f"{diagnostic_name}"
                        )


def _verify_checkpoint_sidecar(
    checkpoint: Path,
    *,
    transition: int,
    config_sha256: str,
    xml_sha256: str,
) -> str:
    identity_path = checkpoint / "identity.json"
    payload_path = checkpoint / "payload.pkl"
    if not identity_path.is_file() or not payload_path.is_file():
        raise ValueError(f"formal checkpoint artifact is missing at {transition}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("config_sha256") != config_sha256:
        raise ValueError("formal checkpoint config identity mismatch")
    if identity.get("xml_sha256") != xml_sha256:
        raise ValueError("formal checkpoint XML identity mismatch")
    if identity.get("training_transitions") != transition:
        raise ValueError("formal checkpoint transition mismatch")
    digest = _payload_sha256(payload_path)
    if identity.get("payload_sha256") != digest:
        raise ValueError("formal checkpoint payload hash mismatch")
    return digest


def _verify_video_artifacts(
    video_report: Mapping[str, Any],
    *,
    artifact_dir: Path,
    schema: str,
    context: str,
    expected_airborne_rsi: bool | None = None,
    expected_seeds: tuple[int, ...] = (),
) -> None:
    artifact_keys = ("video", "state_trace")
    if schema.endswith(("_v2", "_v3", "_v4")):
        artifact_keys = (
            "video",
            "state_trace",
            "diagnostic_plot",
            "diagnostic_data",
        )
    resolved_paths: dict[str, Path] = {}
    suffixes = {
        "video": ".mp4",
        "state_trace": ".npz",
        "diagnostic_plot": ".png",
        "diagnostic_data": ".npz",
    }
    for artifact_key in artifact_keys:
        artifact = Path(str(video_report.get(artifact_key, "")))
        if not artifact.is_file() or artifact.resolve().parent != artifact_dir.resolve():
            raise ValueError(f"{context} artifact is invalid: {artifact_key}")
        if artifact.suffix.lower() != suffixes[artifact_key]:
            raise ValueError(f"{context} artifact type is invalid: {artifact_key}")
        resolved_paths[artifact_key] = artifact.resolve()

    if resolved_paths["video"] == resolved_paths["state_trace"]:
        raise ValueError(f"{context} video and state trace must be distinct")

    if not schema.endswith(("_v2", "_v3", "_v4")):
        return
    if resolved_paths["state_trace"] != resolved_paths["diagnostic_data"]:
        raise ValueError(f"{context} state trace must equal diagnostic data")
    if len(
        {
            resolved_paths["video"],
            resolved_paths["diagnostic_plot"],
            resolved_paths["diagnostic_data"],
        }
    ) != 3:
        raise ValueError(f"{context} video, plot, and data must be distinct")
    for artifact_key, hash_key in (
        ("video", "video_sha256"),
        ("diagnostic_plot", "diagnostic_plot_sha256"),
        ("diagnostic_data", "diagnostic_data_sha256"),
    ):
        expected = video_report.get(hash_key)
        actual = _payload_sha256(resolved_paths[artifact_key])
        if expected != actual:
            raise ValueError(f"{context} {artifact_key} hash mismatch")
    captured = video_report.get("captured_state_count")
    encoded = video_report.get("encoded_frame_count")
    if not isinstance(captured, int) or not isinstance(encoded, int):
        raise ValueError(f"{context} frame counts are invalid")
    representative_episode: Path | None = None
    if expected_airborne_rsi is not None:
        representative_seed = video_report.get("representative_seed")
        if representative_seed not in expected_seeds:
            raise ValueError(f"{context} representative seed is invalid")
        representative_episode = (
            artifact_dir / f"seed_{representative_seed}.npz"
        ).resolve()
        declared_episode = Path(
            str(video_report.get("representative_episode_npz", ""))
        )
        if (
            not representative_episode.is_file()
            or declared_episode.resolve() != representative_episode
        ):
            raise ValueError(f"{context} representative episode path is invalid")
        expected_episode_hash = video_report.get(
            "representative_episode_npz_sha256"
        )
        if expected_episode_hash != _payload_sha256(representative_episode):
            raise ValueError(f"{context} representative episode hash mismatch")
        if video_report.get("reset_source_airborne_rsi") is not expected_airborne_rsi:
            raise ValueError(f"{context} reset source declaration mismatch")
    _verify_diagnostic_npz(
        resolved_paths["diagnostic_data"],
        captured_state_count=captured,
        context=f"{context} diagnostic data",
        expected_airborne_rsi=expected_airborne_rsi,
        episode_npz_path=representative_episode,
    )
    try:
        decoded = media.read_video(resolved_paths["video"])
    except Exception as exc:
        raise ValueError(f"{context} video decode failed") from exc
    if len(decoded) != encoded:
        raise ValueError(f"{context} decoded video frame count mismatch")
    try:
        image = media.read_image(resolved_paths["diagnostic_plot"])
    except Exception as exc:
        raise ValueError(f"{context} diagnostic plot decode failed") from exc
    if np.asarray(image).ndim not in {2, 3}:
        raise ValueError(f"{context} diagnostic plot shape is invalid")


def _verify_apex_split_artifacts(
    video_report: Mapping[str, Any], *, artifact_dir: Path, context: str
) -> None:
    full_path = Path(str(video_report["diagnostic_data"])).resolve()
    segment_paths = {
        "pre": Path(str(video_report.get("pre_apex_data", ""))).resolve(),
        "post": Path(str(video_report.get("post_apex_data", ""))).resolve(),
    }
    if len({full_path, *segment_paths.values()}) != 3:
        raise ValueError(f"{context} Apex data paths must be distinct")
    for name, segment in segment_paths.items():
        if (
            not segment.is_file()
            or segment.parent != artifact_dir.resolve()
            or segment.suffix.lower() != ".npz"
        ):
            raise ValueError(f"{context} {name}-Apex artifact is invalid")
        if video_report.get(f"{name}_apex_data_sha256") != _payload_sha256(
            segment
        ):
            raise ValueError(f"{context} {name}-Apex artifact hash mismatch")

    captured = int(video_report["captured_state_count"])
    apex_index = int(video_report.get("apex_frame_index", -2))
    expected_pre = captured if apex_index == -1 else apex_index + 1
    expected_post = 0 if apex_index == -1 else captured - apex_index
    if not (-1 <= apex_index < captured):
        raise ValueError(f"{context} Apex frame index is invalid")
    if video_report.get("pre_apex_sample_count") != expected_pre:
        raise ValueError(f"{context} pre-Apex sample count mismatch")
    if video_report.get("post_apex_sample_count") != expected_post:
        raise ValueError(f"{context} post-Apex sample count mismatch")
    pre_transitions = int(video_report.get("pre_apex_environment_transitions", -1))
    post_transitions = int(video_report.get("post_apex_environment_transitions", -1))
    if pre_transitions + post_transitions != captured - 1:
        raise ValueError(f"{context} Apex transition accounting mismatch")
    expected_pre_transitions = captured - 1 if apex_index == -1 else apex_index
    expected_post_transitions = 0 if apex_index == -1 else captured - apex_index - 1
    if (pre_transitions, post_transitions) != (
        expected_pre_transitions,
        expected_post_transitions,
    ):
        raise ValueError(f"{context} Apex transition split mismatch")

    with (
        np.load(full_path, allow_pickle=False) as full,
        np.load(segment_paths["pre"], allow_pickle=False) as pre,
        np.load(segment_paths["post"], allow_pickle=False) as post,
    ):
        required_full = {"apex_frame_index", "segment_pre_apex", "segment_post_apex"}
        if not required_full <= set(full.files):
            raise ValueError(f"{context} Apex masks are missing")
        if int(np.asarray(full["apex_frame_index"])[0]) != apex_index:
            raise ValueError(f"{context} Apex index declaration mismatch")
        expected_indices = {
            "pre": np.arange(expected_pre, dtype=np.int32),
            "post": (
                np.empty(0, dtype=np.int32)
                if apex_index == -1
                else np.arange(apex_index, captured, dtype=np.int32)
            ),
        }
        expected_pre_mask = np.zeros(captured, dtype=bool)
        expected_pre_mask[expected_indices["pre"]] = True
        expected_post_mask = np.zeros(captured, dtype=bool)
        expected_post_mask[expected_indices["post"]] = True
        if not np.array_equal(full["segment_pre_apex"], expected_pre_mask):
            raise ValueError(f"{context} pre-Apex mask mismatch")
        if not np.array_equal(full["segment_post_apex"], expected_post_mask):
            raise ValueError(f"{context} post-Apex mask mismatch")
        for name, segment, expected in (
            ("pre", pre, expected_indices["pre"]),
            ("post", post, expected_indices["post"]),
        ):
            if (
                "apex_frame_index" not in segment.files
                or segment["apex_frame_index"].shape != (1,)
                or int(segment["apex_frame_index"][0]) != apex_index
            ):
                raise ValueError(f"{context} {name}-Apex index mismatch")
            if "source_frame_index" not in segment.files or not np.array_equal(
                segment["source_frame_index"], expected
            ):
                raise ValueError(f"{context} {name}-Apex source indices mismatch")
            for key in full.files:
                values = np.asarray(full[key])
                if key in required_full or values.ndim < 1 or values.shape[0] != captured:
                    continue
                if key not in segment.files or not np.array_equal(
                    segment[key], values[expected]
                ):
                    raise ValueError(f"{context} {name}-Apex {key} mismatch")
        if apex_index >= 0 and not np.array_equal(pre["qpos"][-1], post["qpos"][0]):
            raise ValueError(f"{context} Apex boundary state mismatch")


def _verify_training_curves(path: Path, *, target: int) -> dict[str, Any]:
    report_path = path / "training_curves.json"
    if not report_path.is_file():
        raise ValueError("formal v4 training curve report is missing")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    plot = Path(str(payload.get("plot_path", ""))).resolve()
    data = Path(str(payload.get("data_path", ""))).resolve()
    if (
        not plot.is_file()
        or plot.parent != path.resolve()
        or plot.suffix.lower() != ".png"
        or not data.is_file()
        or data.parent != path.resolve()
        or data.suffix.lower() != ".npz"
        or plot == data
    ):
        raise ValueError("formal v4 training curve artifact is invalid")
    if payload.get("plot_sha256") != _payload_sha256(plot):
        raise ValueError("formal v4 training curve plot hash mismatch")
    if payload.get("data_sha256") != _payload_sha256(data):
        raise ValueError("formal v4 training curve data hash mismatch")
    required = {
        "episode_training_transitions",
        "episode_mean_reward",
        "episode_mean_length",
        "episode_airborne_rsi_fraction",
        "ppo_training_transitions",
        "ppo_kl_mean",
        "ppo_policy_loss",
        "ppo_value_loss",
        "ppo_total_loss",
        "ppo_policy_mean_std",
        "ppo_sps",
    }
    episode_jsonl = path / "episode_metrics.jsonl"
    ppo_jsonl = path / "metrics.jsonl"
    if not episode_jsonl.is_file() or not ppo_jsonl.is_file():
        raise ValueError("formal v4 raw training metrics are missing")
    episode_rows = [
        json.loads(line)
        for line in episode_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    ppo_rows = [
        json.loads(line)
        for line in ppo_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    with np.load(data, allow_pickle=False) as arrays:
        if not required <= set(arrays.files):
            raise ValueError("formal v4 training curve series are incomplete")
        episode_count = int(payload.get("episode_sample_count", -1))
        ppo_count = int(payload.get("ppo_sample_count", -1))
        episode_steps = np.asarray(arrays["episode_training_transitions"])
        ppo_steps = np.asarray(arrays["ppo_training_transitions"])
        if episode_steps.shape != (episode_count,) or ppo_steps.shape != (ppo_count,):
            raise ValueError("formal v4 training curve sample count mismatch")
        if episode_count <= 0 or ppo_count <= 0:
            raise ValueError("formal v4 training curves must be nonempty")
        if len(episode_rows) != episode_count or len(ppo_rows) != ppo_count:
            raise ValueError("formal v4 raw/curve sample count mismatch")
        if np.any(np.diff(episode_steps) <= 0) or np.any(np.diff(ppo_steps) <= 0):
            raise ValueError("formal v4 training curve steps are not increasing")
        if int(ppo_steps[-1]) != target:
            raise ValueError("formal v4 PPO curve does not reach target")
        for key in required - {
            "episode_training_transitions",
            "ppo_training_transitions",
        }:
            values = np.asarray(arrays[key])
            expected_count = episode_count if key.startswith("episode_") else ppo_count
            if values.shape != (expected_count,) or not np.isfinite(values).all():
                raise ValueError(f"formal v4 training curve is invalid: {key}")
        raw_bindings = {
            "episode_training_transitions": (
                episode_rows,
                "training_transitions",
                False,
            ),
            "episode_mean_reward": (episode_rows, "episode/sum_reward", True),
            "episode_mean_length": (episode_rows, "episode/length", True),
            "ppo_training_transitions": (ppo_rows, "training_transitions", False),
            "ppo_kl_mean": (ppo_rows, "training/kl_mean", True),
            "ppo_policy_loss": (ppo_rows, "training/policy_loss", True),
            "ppo_value_loss": (ppo_rows, "training/v_loss", True),
            "ppo_total_loss": (ppo_rows, "training/total_loss", True),
            "ppo_policy_mean_std": (
                ppo_rows,
                "training/policy_dist_mean_std",
                True,
            ),
            "ppo_sps": (ppo_rows, "training/sps", True),
        }
        for array_name, (rows, key, nested) in raw_bindings.items():
            try:
                raw = np.asarray(
                    [row["metrics"][key] if nested else row[key] for row in rows]
                )
            except (KeyError, TypeError) as exc:
                raise ValueError(f"formal v4 raw metric is missing: {key}") from exc
            if not np.array_equal(np.asarray(arrays[array_name]), raw):
                raise ValueError(f"formal v4 raw metric mismatch: {key}")
        reset_counts = np.asarray(
            [row["metrics"]["episode/reset/source_airborne_rsi"] for row in episode_rows],
            dtype=np.float64,
        )
        expected_fraction = reset_counts / np.asarray(arrays["episode_mean_length"])
        if not np.allclose(
            arrays["episode_airborne_rsi_fraction"], expected_fraction,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("formal v4 raw RSI fraction mismatch")
    try:
        image = media.read_image(plot)
    except Exception as exc:
        raise ValueError("formal v4 training curve plot decode failed") from exc
    if np.asarray(image).ndim not in {2, 3}:
        raise ValueError("formal v4 training curve plot shape is invalid")
    return payload


def _verify_formal_run(
    path: Path,
    *,
    manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    status: Mapping[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    if status["status"] != "completed":
        return report
    schema = str(resolved.get("schema", ""))
    if schema not in {
        "jit_phase_u_formal_v1",
        "jit_phase_u_formal_v2",
        "jit_phase_u_formal_v3",
        "jit_phase_u_formal_v4",
    }:
        raise ValueError("formal run does not use the formal config schema")
    if schema in {
        "jit_phase_u_formal_v2",
        "jit_phase_u_formal_v3",
        "jit_phase_u_formal_v4",
    }:
        resolve_config_payload(resolved)
    if schema == "jit_phase_u_formal_v4":
        if (
            type(manifest.get("starting_training_transition")) is not int
            or manifest["starting_training_transition"] != 0
        ):
            raise ValueError(
                "formal v4 fresh-start provenance requires starting transition 0"
            )
        if manifest.get("parent_checkpoint") is not None:
            raise ValueError(
                "formal v4 fresh-start provenance forbids a parent checkpoint"
            )
        if manifest.get("resume_semantics") != "fresh":
            raise ValueError(
                "formal v4 fresh-start provenance requires fresh resume semantics"
            )
        expected_seed = int(resolved["ppo"]["seed"])
        if (
            type(manifest.get("segment_seed")) is not int
            or manifest["segment_seed"] != expected_seed
        ):
            raise ValueError(
                "formal v4 fresh-start provenance segment seed mismatch"
            )
        resume_command = manifest.get("resume_command")
        if (
            not isinstance(resume_command, str)
            or "--restore-checkpoint" in resume_command
        ):
            raise ValueError(
                "formal v4 fresh-start provenance forbids restore-bearing resume commands"
            )
        resume_command_path = path / "resume_command.txt"
        expected_persisted_command = resume_command + "\n"
        if (
            not resume_command_path.is_file()
            or resume_command_path.read_text(encoding="utf-8")
            != expected_persisted_command
        ):
            raise ValueError(
                "formal v4 fresh-start provenance resume command artifact mismatch"
            )
    repository_root = Path(__file__).resolve().parents[3]
    model = resolved["model"]
    if model.get("xml_sha256") != manifest["xml_sha256"]:
        raise ValueError("formal config/XML manifest identity mismatch")
    if model.get("reference_sha256") != manifest["reference_sha256"]:
        raise ValueError("formal config/reference manifest identity mismatch")
    if file_sha256(repository_root / model["xml_path"]) != manifest["xml_sha256"]:
        raise ValueError("authoritative XML identity drift")
    if (
        file_sha256(repository_root / model["reference_path"])
        != manifest["reference_sha256"]
    ):
        raise ValueError("reference trajectory identity drift")

    target = int(resolved["ppo"]["requested_transitions"])
    if schema == "jit_phase_u_formal_v1" and target != 998_400:
        raise ValueError("formal target must equal 998400")
    start = int(manifest.get("starting_training_transition", 0))
    expected_segment = target - start
    accounting = status["interaction_accounting"]
    if accounting["training"] != expected_segment:
        raise ValueError("formal segment training accounting mismatch")
    if int(manifest["training_transition_ceiling"]) != expected_segment:
        raise ValueError("formal segment training ceiling mismatch")
    if accounting["brax_evaluation"] != 0:
        raise ValueError("formal run contains undeclared Brax evaluation transitions")
    if schema not in {"jit_phase_u_formal_v3", "jit_phase_u_formal_v4"} and accounting["diagnostic"] != 0:
        raise ValueError("legacy formal run contains undeclared diagnostic transitions")

    formal = resolved["formal"]
    expected_checkpoints = tuple(
        int(step)
        for step in formal["checkpoint_transitions"]
        if int(step) >= start
    )
    checkpoint_hashes = {}
    for step in expected_checkpoints:
        checkpoint_hashes[str(step)] = _verify_checkpoint_sidecar(
            path / "checkpoints" / f"transition_{step}",
            transition=step,
            config_sha256=manifest["config_sha256"],
            xml_sha256=manifest["xml_sha256"],
        )

    expected_seeds = tuple(int(seed) for seed in resolved["ppo"]["held_out_seeds"])
    episode_horizon = int(resolved["ppo"]["episode_horizon"])
    expected_evaluations = tuple(
        int(step)
        for step in formal["fixed_evaluation_transitions"]
        if int(step) > start
    )
    fixed_total = 0
    evaluation_summaries: dict[str, Any] = {}
    for step in expected_evaluations:
        panel_dir = path / "evaluations" / f"transition_{step}"
        summary_path = panel_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError(f"formal evaluation summary is missing at {step}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("absolute_transition") != step:
            raise ValueError("formal evaluation absolute transition mismatch")
        if tuple(summary.get("held_out_seeds", ())) != expected_seeds:
            raise ValueError("formal evaluation held-out seeds mismatch")
        if summary.get("rollouts") != len(expected_seeds):
            raise ValueError("formal evaluation rollout count mismatch")
        panel_total = 0
        for seed in expected_seeds:
            metadata_path = panel_dir / f"seed_{seed}.json"
            npz_path = panel_dir / f"seed_{seed}.npz"
            if not metadata_path.is_file() or not npz_path.is_file():
                raise ValueError(f"formal evaluation trace is missing for seed {seed}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("seed") != seed:
                raise ValueError("formal trace seed mismatch")
            transitions = metadata.get("environment_transitions")
            if (
                not isinstance(transitions, int)
                or not 1 <= transitions <= episode_horizon
            ):
                raise ValueError("formal trace transition count is invalid")
            if metadata.get("captured_state_count") != transitions + 1:
                raise ValueError("formal trace transition/state count mismatch")
            digest = _payload_sha256(npz_path)
            if metadata.get("npz_sha256") != digest:
                raise ValueError("formal trace payload hash mismatch")
            if schema in {"jit_phase_u_formal_v3", "jit_phase_u_formal_v4"}:
                _verify_episode_npz(
                    npz_path,
                    captured_state_count=transitions + 1,
                    expected_airborne_rsi=False,
                    context="formal natural evaluation",
                )
            panel_total += transitions
        if summary.get("environment_transitions") != panel_total:
            raise ValueError("formal trace transition total mismatches panel summary")
        fixed_total += panel_total
        evaluation_summaries[str(step)] = {
            "environment_transitions": panel_total,
            "apex_success_rate": summary.get("apex_success_rate"),
            "physical_failure_rate": summary.get("physical_failure_rate"),
            "end_reason_counts": summary.get("end_reason_counts", {}),
        }
    if accounting["fixed_evaluation"] != fixed_total:
        raise ValueError("formal fixed evaluation accounting mismatch")

    diagnostic_total = 0
    diagnostic_summaries: dict[str, Any] = {}
    if schema in {"jit_phase_u_formal_v3", "jit_phase_u_formal_v4"}:
        for step in expected_evaluations:
            panel_dir = path / "diagnostics" / "airborne_rsi" / f"transition_{step}"
            summary_path = panel_dir / "summary.json"
            if not summary_path.is_file():
                raise ValueError(
                    f"formal airborne RSI diagnostic summary is missing at {step}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("absolute_transition") != step:
                raise ValueError("formal RSI diagnostic transition mismatch")
            if tuple(summary.get("held_out_seeds", ())) != expected_seeds:
                raise ValueError("formal RSI diagnostic seeds mismatch")
            if summary.get("rollouts") != len(expected_seeds):
                raise ValueError("formal RSI diagnostic rollout count mismatch")
            panel_total = 0
            for seed in expected_seeds:
                metadata_path = panel_dir / f"seed_{seed}.json"
                npz_path = panel_dir / f"seed_{seed}.npz"
                if not metadata_path.is_file() or not npz_path.is_file():
                    raise ValueError(
                        f"formal RSI diagnostic trace is missing for seed {seed}"
                    )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                transitions = metadata.get("environment_transitions")
                if metadata.get("seed") != seed:
                    raise ValueError("formal RSI diagnostic trace seed mismatch")
                if (
                    not isinstance(transitions, int)
                    or not 1 <= transitions <= episode_horizon
                ):
                    raise ValueError("formal RSI diagnostic transition count is invalid")
                if metadata.get("captured_state_count") != transitions + 1:
                    raise ValueError("formal RSI diagnostic state count mismatch")
                if metadata.get("npz_sha256") != _payload_sha256(npz_path):
                    raise ValueError("formal RSI diagnostic trace hash mismatch")
                _verify_episode_npz(
                    npz_path,
                    captured_state_count=transitions + 1,
                    expected_airborne_rsi=True,
                    context="formal RSI diagnostic",
                )
                panel_total += transitions
            if summary.get("environment_transitions") != panel_total:
                raise ValueError("formal RSI diagnostic panel total mismatch")
            diagnostic_total += panel_total
            diagnostic_summaries[str(step)] = {
                "environment_transitions": panel_total,
                "apex_success_rate": summary.get("apex_success_rate"),
                "physical_failure_rate": summary.get("physical_failure_rate"),
                "end_reason_counts": summary.get("end_reason_counts", {}),
            }
        if accounting["diagnostic"] != diagnostic_total:
            raise ValueError("formal RSI diagnostic accounting mismatch")

        final_diagnostic = (
            path / "diagnostics" / "airborne_rsi" / f"transition_{target}"
        )
        diagnostic_video_path = final_diagnostic / "video_report.json"
        if not diagnostic_video_path.is_file():
            raise ValueError("formal final RSI diagnostic video report is missing")
        diagnostic_video = json.loads(
            diagnostic_video_path.read_text(encoding="utf-8")
        )
        _verify_video_artifacts(
            diagnostic_video,
            artifact_dir=final_diagnostic,
            schema=schema,
            context="formal final RSI diagnostic",
            expected_airborne_rsi=True,
            expected_seeds=expected_seeds,
        )
        if schema == "jit_phase_u_formal_v4":
            _verify_apex_split_artifacts(
                diagnostic_video,
                artifact_dir=final_diagnostic,
                context="formal final RSI diagnostic",
            )
        diagnostic_video_transitions = diagnostic_video.get(
            "environment_transitions"
        )
        diagnostic_captured = diagnostic_video.get("captured_state_count")
        diagnostic_encoded = diagnostic_video.get("encoded_frame_count")
        if (
            not isinstance(diagnostic_video_transitions, int)
            or diagnostic_captured != diagnostic_video_transitions + 1
            or diagnostic_encoded != diagnostic_captured
        ):
            raise ValueError(
                "formal final RSI diagnostic state/frame accounting mismatch"
            )

    final_panel = path / "evaluations" / f"transition_{target}"
    video_report_path = final_panel / "video_report.json"
    if not video_report_path.is_file():
        raise ValueError("formal final representative video report is missing")
    video_report = json.loads(video_report_path.read_text(encoding="utf-8"))
    _verify_video_artifacts(
        video_report,
        artifact_dir=final_panel,
        schema=schema,
        context=(
            "formal final natural"
            if schema in {"jit_phase_u_formal_v3", "jit_phase_u_formal_v4"}
            else "formal final"
        ),
        expected_airborne_rsi=(
            False
            if schema in {"jit_phase_u_formal_v3", "jit_phase_u_formal_v4"}
            else None
        ),
        expected_seeds=expected_seeds,
    )
    if schema == "jit_phase_u_formal_v4":
        _verify_apex_split_artifacts(
            video_report,
            artifact_dir=final_panel,
            context="formal final natural",
        )
    video_transitions = video_report.get("environment_transitions")
    captured = video_report.get("captured_state_count")
    encoded = video_report.get("encoded_frame_count")
    if (
        not isinstance(video_transitions, int)
        or captured != video_transitions + 1
        or encoded != captured
    ):
        raise ValueError("formal final video state/frame accounting mismatch")

    formal_report_path = path / "formal_report.json"
    if not formal_report_path.is_file():
        raise ValueError("formal report is missing")
    formal_report = json.loads(formal_report_path.read_text(encoding="utf-8"))
    if formal_report.get("starting_training_transition") != start:
        raise ValueError("formal report starting transition mismatch")
    if formal_report.get("completed_training_transitions") != target:
        raise ValueError("formal report target mismatch")
    if formal_report.get("segment_training_transitions") != expected_segment:
        raise ValueError("formal report segment count mismatch")
    if formal_report.get("fixed_evaluation_transitions") != fixed_total:
        raise ValueError("formal report fixed evaluation count mismatch")
    if formal_report.get("diagnostic_transitions", 0) != diagnostic_total:
        raise ValueError("formal report diagnostic count mismatch")
    if tuple(formal_report.get("checkpoint_transitions", ())) != expected_checkpoints:
        raise ValueError("formal report checkpoint schedule mismatch")
    if tuple(formal_report.get("evaluated_transitions", ())) != expected_evaluations:
        raise ValueError("formal report evaluation schedule mismatch")
    if formal_report.get("checkpoint_restored") is not True:
        raise ValueError("formal report lacks final checkpoint restore evidence")
    for value in formal_report.get("final_metrics", {}).values():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("formal report contains a nonfinite metric")

    training_curves: dict[str, Any] = {}
    if schema == "jit_phase_u_formal_v4":
        training_curves = _verify_training_curves(path, target=target)

    report.update(
        {
            "absolute_training_transition": target,
            "formal_checkpoint_transitions": list(expected_checkpoints),
            "formal_checkpoint_payload_sha256": checkpoint_hashes,
            "formal_evaluated_transitions": list(expected_evaluations),
            "fixed_evaluation_transitions": fixed_total,
            "diagnostic_transitions": diagnostic_total,
            "evaluation_summaries": evaluation_summaries,
            "airborne_rsi_diagnostic_summaries": diagnostic_summaries,
            "checkpoint_restored": True,
            "training_curves": training_curves,
        }
    )
    return report


def verify_run(run_dir: Path) -> dict[str, Any]:
    """Verifies a terminal run's immutable identities and interaction ledger."""

    path = Path(run_dir)
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    resolved = json.loads((path / "resolved_config.json").read_text(encoding="utf-8"))
    status = json.loads((path / "status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"completed", "engineering_error", "aborted"}:
        raise ValueError("run status is not terminal")
    actual_config_sha256 = canonical_sha256(resolved)
    if manifest.get("config_sha256") != actual_config_sha256:
        raise ValueError("run config_sha256 mismatch")
    accounting = status.get("interaction_accounting", {})
    required_counts = {"training", "brax_evaluation", "fixed_evaluation", "diagnostic"}
    if set(accounting) != required_counts:
        raise ValueError("interaction accounting fields are incomplete")
    if any(not isinstance(accounting[key], int) or accounting[key] < 0 for key in required_counts):
        raise ValueError("interaction accounting values must be nonnegative integers")
    total = sum(accounting.values())
    if status.get("environment_transitions") != total:
        raise ValueError("total environment transition accounting mismatch")
    ceiling = int(manifest["training_transition_ceiling"])
    if accounting["training"] > ceiling:
        raise ValueError("training transition ceiling exceeded")

    report: dict[str, Any] = {
        "status": status["status"],
        "training_transitions": accounting["training"],
        "total_environment_transitions": total,
        "config_sha256": actual_config_sha256,
    }
    purpose = manifest.get("purpose")
    if purpose == "formal_propulsion_ascent_ppo":
        return _verify_formal_run(
            path,
            manifest=manifest,
            resolved=resolved,
            status=status,
            report=report,
        )
    is_smoke = purpose == "compile_update_checkpoint_restore_engineering_smoke"
    if not is_smoke:
        return report
    if status["status"] != "completed":
        return report
    if accounting["training"] != ceiling:
        raise ValueError("completed engineering smoke did not use one exact declared block")
    schema = str(resolved.get("schema", ""))
    if schema not in {
        "jit_phase_u_engineering_smoke_v1",
        "jit_phase_u_engineering_smoke_v2",
        "jit_phase_u_engineering_smoke_v3",
        "jit_phase_u_engineering_smoke_v4",
    }:
        raise ValueError("engineering smoke does not use a supported config schema")
    if schema in {
        "jit_phase_u_engineering_smoke_v2",
        "jit_phase_u_engineering_smoke_v3",
        "jit_phase_u_engineering_smoke_v4",
    }:
        resolve_config_payload(resolved)

    repository_root = Path(__file__).resolve().parents[3]
    model_config = resolved["model"]
    xml_path = repository_root / model_config["xml_path"]
    reference_path = repository_root / model_config["reference_path"]
    if file_sha256(xml_path) != manifest["xml_sha256"]:
        raise ValueError("authoritative XML identity drift")
    if file_sha256(reference_path) != manifest["reference_sha256"]:
        raise ValueError("reference trajectory identity drift")

    checkpoint = path / "checkpoints" / f"transition_{accounting['training']}"
    identity_path = checkpoint / "identity.json"
    payload_path = checkpoint / "payload.pkl"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if identity.get("config_sha256") != manifest["config_sha256"]:
        raise ValueError("final checkpoint config identity mismatch")
    if identity.get("xml_sha256") != manifest["xml_sha256"]:
        raise ValueError("final checkpoint XML identity mismatch")
    if identity.get("training_transitions") != accounting["training"]:
        raise ValueError("final checkpoint transition mismatch")
    if identity.get("payload_sha256") != _payload_sha256(payload_path):
        raise ValueError("final checkpoint payload hash mismatch")

    smoke = json.loads((path / "smoke_report.json").read_text(encoding="utf-8"))
    if smoke.get("completed_training_transitions") != accounting["training"]:
        raise ValueError("smoke report training count mismatch")
    if smoke.get("diagnostic_transitions") != accounting["diagnostic"]:
        raise ValueError("smoke report diagnostic count mismatch")
    if smoke.get("checkpoint_restored") is not True:
        raise ValueError("smoke report lacks checkpoint restore evidence")
    diagnostic = json.loads(
        (path / "diagnostic_summary.json").read_text(encoding="utf-8")
    )
    if diagnostic.get("environment_transitions") != accounting["diagnostic"]:
        raise ValueError("diagnostic summary count mismatch")
    video = json.loads((path / "video_report.json").read_text(encoding="utf-8"))
    if video.get("environment_transitions") != accounting["diagnostic"]:
        raise ValueError("video transition count mismatch")
    if video.get("captured_state_count") != accounting["diagnostic"] + 1:
        raise ValueError("video captured state count mismatch")
    if video.get("encoded_frame_count") != video.get("captured_state_count"):
        raise ValueError("video encoded frame count mismatch")
    _verify_video_artifacts(
        video,
        artifact_dir=path,
        schema=schema,
        context="smoke diagnostic",
    )
    report.update(
        {
            "checkpoint_restored": True,
            "checkpoint_payload_sha256": identity["payload_sha256"],
            "diagnostic_end_reason_counts": diagnostic["end_reason_counts"],
            "captured_state_count": video["captured_state_count"],
            "encoded_frame_count": video["encoded_frame_count"],
        }
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-run")
    verify.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "verify-run":
        print(json.dumps(verify_run(args.run_dir), indent=2, sort_keys=True))
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
