from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_continuation_field_train_cv import (
    run_upstream_train_logo_diagnostic,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TRAIN-only leave-one-parent-group-out diagnostics for the failed C_up^0."
        )
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_upstream_train_logo_diagnostic(args.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
