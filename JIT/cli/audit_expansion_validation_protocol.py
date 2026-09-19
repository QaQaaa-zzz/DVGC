from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.expansion_validation_protocol import audit_expansion_validation_protocol


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a group-disjoint expansion-validation declaration without interactions."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            audit_expansion_validation_protocol(args.config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
