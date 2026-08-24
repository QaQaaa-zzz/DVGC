"""Run predeclaration and closed interaction accounting."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from .config import canonical_sha256, file_sha256


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
    run_dir = Path(declaration.output_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = asdict(declaration)
    manifest["output_dir"] = str(run_dir.resolve())
    manifest["stopping_conditions"] = list(declaration.stopping_conditions)
    manifest["claim_boundary"] = {
        "engineering_integrity_only": True,
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
    is_smoke = manifest.get("purpose") == "compile_update_checkpoint_restore_engineering_smoke"
    if not is_smoke:
        return report
    if status["status"] != "completed":
        return report
    if accounting["training"] != ceiling:
        raise ValueError("completed engineering smoke did not use one exact declared block")

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
    for artifact_key in ("video", "state_trace"):
        if not Path(video[artifact_key]).is_file():
            raise ValueError(f"video artifact is missing: {artifact_key}")
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
