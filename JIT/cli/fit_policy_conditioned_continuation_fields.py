from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.policy_conditioned_continuation_field import (
    fit_policy_conditioned_continuation_fields,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit frozen-pi_k policy-conditioned continuation fields."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            fit_policy_conditioned_continuation_fields(args.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
