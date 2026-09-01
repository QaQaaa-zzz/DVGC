from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_checkpoint_domain_cv import run_upstream_checkpoint_domain_cv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TRAIN-only leave-one-checkpoint-domain-out C_up diagnostic."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_upstream_checkpoint_domain_cv(args.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
