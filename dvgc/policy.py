"""Immutable policy bundles with environment/config provenance."""
from __future__ import annotations

import hashlib
import json
import pickle
import shutil
from pathlib import Path
from typing import Any

from .config import POLICY_NETWORK_VERSION, config_hash, file_sha256

BUNDLE_VERSION = 1


def verify_manifest_artifact(
    manifest: dict[str, Any], role: str, path: str | Path | None, *, required: bool = True
) -> None:
    """Verify that a CLI artifact is exactly the one declared by a policy."""
    if role not in ("candidate_bank", "downstream_bank"):
        raise ValueError(f"Unsupported policy artifact role: {role}")
    expected = manifest.get(f"{role}_sha256")
    if not path:
        if required:
            raise ValueError(f"Policy requires its declared {role}")
        if expected is not None:
            raise ValueError(f"Policy declares {role}, but the CLI omitted it")
        return
    artifact = Path(path)
    if expected is None:
        raise ValueError(f"Policy manifest does not declare a {role}")
    if not artifact.is_file():
        raise ValueError(f"Policy {role} is missing: {artifact}")
    actual = file_sha256(artifact)
    if actual != expected:
        raise ValueError(
            f"Policy {role} mismatch: expected SHA-256 {expected}, got {actual}"
        )


def save_bundle(
    directory: str | Path,
    *,
    params: Any,
    config,
    xml_path: str | Path,
    candidate_bank: str | Path | None,
    downstream_bank: str | Path | None,
    policy_version: str,
    estimator_version: str = "event_filter_v1",
    extra: dict[str, Any] | None = None,
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "params.pkl").open("wb") as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    cfg_payload = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    (root / "config.json").write_text(json.dumps(cfg_payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "policy_version": str(policy_version),
        "estimator_version": str(estimator_version),
        "config_hash": config_hash(config),
        "action_mapping_version": str(config.action_mapping_version),
        "policy_network_version": POLICY_NETWORK_VERSION,
        "xml_path": str(Path(xml_path).resolve()),
        "xml_sha256": file_sha256(xml_path),
        "candidate_bank": str(Path(candidate_bank).resolve()) if candidate_bank else None,
        "candidate_bank_sha256": file_sha256(candidate_bank) if candidate_bank and Path(candidate_bank).exists() else None,
        "downstream_bank": str(Path(downstream_bank).resolve()) if downstream_bank else None,
        "downstream_bank_sha256": file_sha256(downstream_bank) if downstream_bank and Path(downstream_bank).exists() else None,
    }
    manifest.update(extra or {})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return root


def load_bundle(directory: str | Path, *, verify_files: bool = True) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("bundle_version", -1)) != BUNDLE_VERSION:
        raise ValueError(f"Unsupported policy bundle version: {manifest.get('bundle_version')}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    raw = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str).encode()
    if hashlib.sha256(raw).hexdigest() != manifest.get("config_hash"):
        raise ValueError("Policy config.json does not match its manifest")
    if config.get("action_mapping_version") != manifest.get("action_mapping_version"):
        raise ValueError("Policy action mapping does not match its manifest")
    if manifest.get("policy_network_version") != POLICY_NETWORK_VERSION:
        raise ValueError(
            "Policy network version is incompatible with this source: "
            f"expected {POLICY_NETWORK_VERSION}, got {manifest.get('policy_network_version')}"
        )
    if verify_files:
        xml_path = Path(manifest["xml_path"])
        if not xml_path.exists() or file_sha256(xml_path) != manifest["xml_sha256"]:
            raise ValueError("Policy XML is missing or has changed since training")
        for key in ("candidate_bank", "downstream_bank"):
            value = manifest.get(key)
            expected = manifest.get(f"{key}_sha256")
            if value and expected and (not Path(value).exists() or file_sha256(value) != expected):
                raise ValueError(f"Policy provenance mismatch: {key}")
    with (root / "params.pkl").open("rb") as f:
        params = pickle.load(f)
    return params, config, manifest


def copy_bundle(source: str | Path, destination: str | Path) -> Path:
    dst = Path(destination)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(source, dst)
    return dst
