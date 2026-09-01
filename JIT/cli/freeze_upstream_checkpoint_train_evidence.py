from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_checkpoint_train_evidence import (
    audit_upstream_checkpoint_train_evidence,
    freeze_upstream_checkpoint_train_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze merged upstream TRAIN evidence across checkpoint domains."
    )
    parser.add_argument("config", type=Path)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    result = (
        audit_upstream_checkpoint_train_evidence(args.config)
        if args.audit_only
        else freeze_upstream_checkpoint_train_evidence(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
