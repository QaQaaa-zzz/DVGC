#!/usr/bin/env python3
"""Plan or explicitly execute a resumable JIT envelope-iteration workflow.

Runtime workflow state is namespaced by the immutable workflow content.  This
prevents a regenerated workflow from colliding with a stale ``state.json`` left
by an older config that reused the same human-readable work root/tag.  Exact
same workflow content resolves to the same state namespace and therefore keeps
normal resume semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from jit_dvgc.workflow import run_workflow


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"workflow config must be a JSON object: {path}")
    return dict(payload)


def _versioned_runtime_config(config_path: Path) -> Path:
    """Resolve one source config into a content-addressed state namespace."""
    source = _read_object(config_path)
    if source.get("schema") != "jit_iteration_workflow_v1":
        raise ValueError("unsupported iteration workflow schema")
    original_state_dir = str(source.get("state_dir", ""))
    if not original_state_dir:
        raise ValueError("workflow state_dir is required")

    # State location itself is deliberately excluded from the workflow-content
    # fingerprint.  Every scientific/engineering stage, command, assertion,
    # environment value, and workflow name remains part of the identity.
    identity_payload = {
        key: value for key, value in source.items() if key != "state_dir"
    }
    fingerprint = _canonical_sha256(identity_payload)
    versioned_state_dir = Path(original_state_dir) / f"config_{fingerprint[:16]}"
    runtime = {
        **source,
        "state_dir": str(versioned_state_dir),
    }

    versioned_state_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = versioned_state_dir / "workflow_config.json"
    serialized = json.dumps(
        runtime,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    if runtime_path.exists():
        existing = runtime_path.read_text(encoding="utf-8")
        if existing != serialized:
            raise ValueError(
                "content-addressed workflow runtime config drift; refusing unsafe resume"
            )
    else:
        runtime_path.write_text(serialized, encoding="utf-8")
    return runtime_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute pending stages; without this flag only print the resolved plan",
    )
    args = parser.parse_args()
    runtime_config = _versioned_runtime_config(args.config)
    result = run_workflow(runtime_config, execute=args.execute)
    if isinstance(result, dict):
        result = {
            **result,
            "source_workflow_config": str(args.config),
            "runtime_workflow_config": str(runtime_config),
        }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
