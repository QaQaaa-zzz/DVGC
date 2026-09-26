from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.iteration_validation_evidence import freeze_iteration_validation_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze completed group-disjoint pi_k validation evidence."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_iteration_validation_evidence(args.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
