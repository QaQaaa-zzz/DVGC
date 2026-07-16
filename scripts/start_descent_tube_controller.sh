#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${DESCENT_TUBE_RUN:-runs/stage_experts/descent_tube_seed0_20260716T2330}"
cd "$ROOT"
exec systemd-run --user --unit=dvgc-descent-tube-controller \
  --description="DVGC exact descent Tube controller" --property=WorkingDirectory="$ROOT" \
  --property=Restart=on-failure --property=RestartSec=10 \
  --property=RestartPreventExitStatus="40 41" --property=StartLimitIntervalSec=0 \
  --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec '$ROOT/scripts/run_descent_tube_pipeline.sh' >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
