#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/qy/mujoco_playground/.venv/bin/python"
RUN="${DESCENT_LOCAL_RUN:-runs/stage_experts/descent_local_nonfinite_repair_seed0_20260716T1825}"
UNIT="${DESCENT_LOCAL_UNIT:-dvgc-descent-local-controller-v4}"

cd "$ROOT"
exec systemd-run --user --unit="$UNIT" \
  --description="DVGC persistent descent-local controller" \
  --property=WorkingDirectory="$ROOT" \
  --property=Restart=on-failure \
  --property=RestartSec=10 \
  --property=RestartPreventExitStatus="40 41" \
  --property=StartLimitIntervalSec=0 \
  --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec '$ROOT/scripts/run_descent_local_pipeline.sh' >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
