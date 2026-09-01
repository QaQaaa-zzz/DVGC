from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_checkpoint_domain_cv import run_upstream_checkpoint_domain_cv
from jit_dvgc.upstream_matched_checkpoint_domain_cv import (
    STATUS as MATCHED_STATUS,
    run_matched_checkpoint_domain_cv,
)


def _status(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint-domain CV config must be a JSON object")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("checkpoint-domain CV config protocol missing")
    return str(protocol.get("status", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TRAIN-only leave-one-checkpoint-domain-out C_up diagnostic."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = (
        run_matched_checkpoint_domain_cv(args.config)
        if _status(args.config) == MATCHED_STATUS
        else run_upstream_checkpoint_domain_cv(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
