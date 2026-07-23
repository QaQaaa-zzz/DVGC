#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${STAGE_NEXT_V3_RUN:-runs/stage_next_reset_v3_seed0_20260723}"
cd "$ROOT"
exec systemd-run --user --unit=dvgc-stage-next-v3-controller \
  --description="DVGC corrected-reset stage-next controller" \
  --setenv="STAGE_NEXT_V3_RUN=$RUN" \
  --property=WorkingDirectory="$ROOT" --property=Restart=on-failure \
  --property=RestartSec=10 --property=RestartPreventExitStatus="40 41" \
  --property=StartLimitIntervalSec=0 --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.stage_next_v3_controller >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
