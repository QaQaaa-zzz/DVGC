from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.downstream_shared_architecture_parent_cv import (
    CONFIG_SCHEMA as DOWNSTREAM_SHARED_CONFIG_SCHEMA,
    run_downstream_shared_architecture_parent_cv,
)
from jit_dvgc.upstream_continuation_field_train_cv import (
    run_upstream_train_logo_diagnostic,
)
from jit_dvgc.upstream_support_stratified_parent_cv import (
    CONFIG_SCHEMA as SUPPORT_STRATIFIED_CONFIG_SCHEMA,
    run_support_stratified_parent_cv,
)


def _schema(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuation TRAIN diagnostic config must be a JSON object")
    return str(payload.get("schema", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TRAIN-only parent-group generalization diagnostics for continuation fields."
        )
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    schema = _schema(args.config)
    if schema == SUPPORT_STRATIFIED_CONFIG_SCHEMA:
        result = run_support_stratified_parent_cv(args.config)
    elif schema == DOWNSTREAM_SHARED_CONFIG_SCHEMA:
        result = run_downstream_shared_architecture_parent_cv(args.config)
    else:
        result = run_upstream_train_logo_diagnostic(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
