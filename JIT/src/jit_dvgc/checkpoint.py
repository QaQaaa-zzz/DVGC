"""Identity-bound checkpoint serialization for engineering smoke runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Any


@dataclass(frozen=True)
class CheckpointIdentity:
    config_sha256: str
    xml_sha256: str
    actor_frame_fields: tuple[str, ...]
    action_order: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointPayload:
    identity: CheckpointIdentity
    training_transitions: int
    observation_normalizer: Any
    actor_params: Any
    critic_params: Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path: Path, payload: CheckpointPayload) -> None:
    if payload.training_transitions < 0:
        raise ValueError("training_transitions must be nonnegative")
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=False)
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(stream.name)
    target = directory / "payload.pkl"
    os.replace(temporary, target)
    identity = asdict(payload.identity)
    identity["actor_frame_fields"] = list(payload.identity.actor_frame_fields)
    identity["action_order"] = list(payload.identity.action_order)
    identity["training_transitions"] = payload.training_transitions
    identity["payload_sha256"] = _sha256(target)
    (directory / "identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_checkpoint(
    path: Path, *, expected: CheckpointIdentity
) -> CheckpointPayload:
    directory = Path(path)
    sidecar = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
    payload_path = directory / "payload.pkl"
    if _sha256(payload_path) != sidecar.get("payload_sha256"):
        raise ValueError("checkpoint payload_sha256 mismatch")
    with payload_path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, CheckpointPayload):
        raise ValueError("checkpoint payload has the wrong type")
    for field in ("config_sha256", "xml_sha256", "actor_frame_fields", "action_order"):
        if getattr(payload.identity, field) != getattr(expected, field):
            raise ValueError(f"checkpoint {field} mismatch")
    if int(sidecar.get("training_transitions", -1)) != payload.training_transitions:
        raise ValueError("checkpoint training_transitions mismatch")
    return payload
