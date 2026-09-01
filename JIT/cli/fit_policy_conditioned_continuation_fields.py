from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.policy_conditioned_continuation_field import (
    fit_policy_conditioned_continuation_fields,
)
from jit_dvgc.shared_continuation_field_refit import (
    CONFIG_SCHEMA as SHARED_REFIT_CONFIG_SCHEMA,
    fit_shared_continuation_fields,
)


def _schema(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("continuation field config must be a JSON object")
    return str(payload.get("schema", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit frozen-pi_k policy-conditioned continuation fields."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = (
        fit_shared_continuation_fields(args.config)
        if _schema(args.config) == SHARED_REFIT_CONFIG_SCHEMA
        else fit_policy_conditioned_continuation_fields(args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
