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

from .config import canonical_sha256, file_sha256, resolve_config_payload


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
) -> None:
    artifact_keys = ("video", "state_trace")
    if schema.endswith("_v2"):
        artifact_keys = (
            "video",
            "state_trace",
            "diagnostic_plot",
            "diagnostic_data",
        )
    resolved_paths: dict[str, Path] = {}
    for artifact_key in artifact_keys:
        artifact = Path(str(video_report.get(artifact_key, "")))
        if not artifact.is_file() or artifact.resolve().parent != artifact_dir.resolve():
            raise ValueError(f"{context} artifact is invalid: {artifact_key}")
        resolved_paths[artifact_key] = artifact.resolve()

    if not schema.endswith("_v2"):
        return
    if resolved_paths["state_trace"] != resolved_paths["diagnostic_data"]:
        raise ValueError(f"{context} state trace must equal diagnostic data")
    for artifact_key, hash_key in (
        ("video", "video_sha256"),
        ("diagnostic_plot", "diagnostic_plot_sha256"),
        ("diagnostic_data", "diagnostic_data_sha256"),
    ):
        expected = video_report.get(hash_key)
        actual = _payload_sha256(resolved_paths[artifact_key])
        if expected != actual:
            raise ValueError(f"{context} {artifact_key} hash mismatch")


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
    if schema not in {"jit_phase_u_formal_v1", "jit_phase_u_formal_v2"}:
        raise ValueError("formal run does not use the formal config schema")
    if schema == "jit_phase_u_formal_v2":
        resolve_config_payload(resolved)
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
    if target != 998_400:
        raise ValueError("formal target must equal 998400")
    start = int(manifest.get("starting_training_transition", 0))
    expected_segment = target - start
    accounting = status["interaction_accounting"]
    if accounting["training"] != expected_segment:
        raise ValueError("formal segment training accounting mismatch")
    if int(manifest["training_transition_ceiling"]) != expected_segment:
        raise ValueError("formal segment training ceiling mismatch")
    if accounting["brax_evaluation"] != 0 or accounting["diagnostic"] != 0:
        raise ValueError("formal run contains undeclared evaluation or diagnostic transitions")

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
            if not isinstance(transitions, int) or not 1 <= transitions <= 200:
                raise ValueError("formal trace transition count is invalid")
            if metadata.get("captured_state_count") != transitions + 1:
                raise ValueError("formal trace transition/state count mismatch")
            digest = _payload_sha256(npz_path)
            if metadata.get("npz_sha256") != digest:
                raise ValueError("formal trace payload hash mismatch")
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

    final_panel = path / "evaluations" / f"transition_{target}"
    video_report_path = final_panel / "video_report.json"
    if not video_report_path.is_file():
        raise ValueError("formal final representative video report is missing")
    video_report = json.loads(video_report_path.read_text(encoding="utf-8"))
    _verify_video_artifacts(
        video_report,
        artifact_dir=final_panel,
        schema=schema,
        context="formal final",
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
    if tuple(formal_report.get("checkpoint_transitions", ())) != expected_checkpoints:
        raise ValueError("formal report checkpoint schedule mismatch")
    if tuple(formal_report.get("evaluated_transitions", ())) != expected_evaluations:
        raise ValueError("formal report evaluation schedule mismatch")
    if formal_report.get("checkpoint_restored") is not True:
        raise ValueError("formal report lacks final checkpoint restore evidence")
    for value in formal_report.get("final_metrics", {}).values():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("formal report contains a nonfinite metric")

    report.update(
        {
            "absolute_training_transition": target,
            "formal_checkpoint_transitions": list(expected_checkpoints),
            "formal_checkpoint_payload_sha256": checkpoint_hashes,
            "formal_evaluated_transitions": list(expected_evaluations),
            "fixed_evaluation_transitions": fixed_total,
            "evaluation_summaries": evaluation_summaries,
            "checkpoint_restored": True,
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
    if schema not in {"jit_phase_u_engineering_smoke_v1", "jit_phase_u_engineering_smoke_v2"}:
        raise ValueError("engineering smoke does not use a supported config schema")
    if schema == "jit_phase_u_engineering_smoke_v2":
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
