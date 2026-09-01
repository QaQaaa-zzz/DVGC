from __future__ import annotations

import argparse
import json
from pathlib import Path

from jit_dvgc.upstream_matched_panel_audit import audit_upstream_matched_panel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether legacy transition_4988928 TRAIN already contains the exact matched upstream panel."
    )
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = audit_upstream_matched_panel(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
