#!/usr/bin/env python3
"""Build or verify the 2026-08-28 locked-artifact handoff manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_DIR = REPO_ROOT / "JIT" / "handoff" / "2026-08-28"
ROOTS_PATH = HANDOFF_DIR / "ARTIFACT_ROOTS.txt"
MANIFEST_PATH = HANDOFF_DIR / "LOCKED_ARTIFACTS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configured_roots() -> list[str]:
    return [
        line.strip()
        for line in ROOTS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def artifact_files(roots: list[str]) -> list[Path]:
    files: set[Path] = set()
    for relative in roots:
        path = (REPO_ROOT / relative).resolve()
        path.relative_to(REPO_ROOT)
        if not path.exists():
            raise FileNotFoundError(relative)
        if path.is_file():
            files.add(path)
        else:
            files.update(item for item in path.rglob("*") if item.is_file())
    return sorted(files, key=lambda item: item.relative_to(REPO_ROOT).as_posix())


def validate_bound_references(files: list[Path]) -> int:
    included = set(files)
    reference_count = 0
    entries_path = (
        REPO_ROOT
        / "JIT/runs/soft_tube/soft_tube_train_v1_20260828/entries.json"
    )
    for entry in json.loads(entries_path.read_text(encoding="utf-8")):
        snapshot = Path(entry["snapshot"])
        if not snapshot.is_absolute():
            snapshot = REPO_ROOT / snapshot
        snapshot = snapshot.resolve()
        snapshot.relative_to(REPO_ROOT)
        snapshot_files = [item for item in snapshot.rglob("*") if item.is_file()]
        if not snapshot_files or any(item not in included for item in snapshot_files):
            raise ValueError(f"Soft Tube snapshot is outside the archive: {snapshot}")
        reference_count += 1

    frozen_path = (
        REPO_ROOT
        / "JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/"
        "frozen_experts.json"
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    for expert in frozen["experts"].values():
        checkpoint = (REPO_ROOT / expert["checkpoint"]).resolve()
        checkpoint_files = [item for item in checkpoint.rglob("*") if item.is_file()]
        if not checkpoint_files or any(item not in included for item in checkpoint_files):
            raise ValueError(f"frozen checkpoint is outside the archive: {checkpoint}")
        reference_count += 1
    return reference_count


def build_manifest() -> dict[str, object]:
    roots = configured_roots()
    files = artifact_files(roots)
    reference_count = validate_bound_references(files)
    records = []
    for path in files:
        records.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema": "jit_locked_handoff_artifacts_v1",
        "handoff_date": "2026-08-28",
        "artifact_roots": roots,
        "file_count": len(records),
        "total_bytes": sum(record["bytes"] for record in records),
        "largest_file_bytes": max(record["bytes"] for record in records),
        "validated_bound_references": reference_count,
        "files": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="replace the manifest with hashes of the configured artifact roots",
    )
    args = parser.parse_args()
    actual = build_manifest()
    if args.write_manifest:
        MANIFEST_PATH.write_text(
            json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}: "
            f"{actual['file_count']} files, {actual['total_bytes']} bytes"
        )
        return 0
    expected = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if actual != expected:
        raise SystemExit("locked handoff artifact manifest mismatch")
    print(
        f"verified {actual['file_count']} files, {actual['total_bytes']} bytes, "
        f"largest={actual['largest_file_bytes']} bytes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
