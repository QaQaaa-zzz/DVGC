from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.iteration_train_evidence import freeze_iteration_train_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze completed pi_k-conditioned transition-band TRAIN evidence."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze_iteration_train_evidence(args.config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
