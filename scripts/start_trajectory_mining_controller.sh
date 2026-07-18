#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TRAJECTORY_MINING_RUN:-}" ]]; then RUN="$TRAJECTORY_MINING_RUN"; else
  RUN=$(/usr/bin/python3 -c "import json; print(json.load(open('$ROOT/runs/ACTIVE_PIPELINE.json'))['run_path'])")
fi
cd "$ROOT"
exec systemd-run --user --unit=dvgc-trajectory-mining-controller \
  --description="DVGC successful-trajectory mining controller" --setenv="TRAJECTORY_MINING_RUN=$RUN" \
  --property=WorkingDirectory="$ROOT" --property=Restart=on-failure --property=RestartSec=10 \
  --property=RestartPreventExitStatus="40 41" --property=StartLimitIntervalSec=0 --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec '$ROOT/scripts/run_trajectory_mining_pipeline.sh' >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
