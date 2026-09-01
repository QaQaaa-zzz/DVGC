from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_parent_diversity_train import (
    audit_upstream_parent_diversity,
    execute_upstream_parent_diversity_train,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand upstream TRAIN parent checkpoint diversity under frozen pi_0."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.audit_only:
        payload = audit_upstream_parent_diversity(args.config)
    else:
        payload = execute_upstream_parent_diversity_train(
            args.config, resume=bool(args.resume)
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
