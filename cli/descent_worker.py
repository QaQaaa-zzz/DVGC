"""Execute one controller-owned command inside an isolated worker service."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    args = parser.parse_args()
    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    command = spec.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(value, str) for value in command):
        raise SystemExit("Worker spec command must be a non-empty string list")
    log = Path(spec["log"])
    log.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in spec.get("environment", {}).items()})
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
