from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_checkpoint_train_evidence import (
    audit_upstream_checkpoint_train_evidence,
    freeze_upstream_checkpoint_train_evidence,
)
from jit_dvgc.upstream_checkpoint_train_source_audit import (
    audit_parent_diversity_source_integrity,
)
from jit_dvgc.upstream_matched_checkpoint_train_evidence import (
    CONFIG_SCHEMA as MATCHED_CONFIG_SCHEMA,
    audit_matched_checkpoint_train_evidence,
    freeze_matched_checkpoint_train_evidence,
)


def _schema(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint TRAIN freeze config must be a JSON object")
    return str(payload.get("schema", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze merged upstream TRAIN evidence across checkpoint domains."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()

    if _schema(args.config) == MATCHED_CONFIG_SCHEMA:
        result = (
            audit_matched_checkpoint_train_evidence(args.config)
            if args.audit_only
            else freeze_matched_checkpoint_train_evidence(args.config)
        )
    else:
        source_integrity = audit_parent_diversity_source_integrity(args.config)
        result = (
            audit_upstream_checkpoint_train_evidence(args.config)
            if args.audit_only
            else freeze_upstream_checkpoint_train_evidence(args.config)
        )
        result = {**result, "source_integrity": source_integrity}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
