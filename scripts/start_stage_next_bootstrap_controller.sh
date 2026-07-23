#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${STAGE_NEXT_BOOTSTRAP_RUN:-runs/stage_next_bootstrap_seed0_20260720}"
cd "$ROOT"
exec systemd-run --user --unit=dvgc-stage-next-bootstrap-controller \
  --description="DVGC stage-to-next-stage bootstrap controller" \
  --setenv="STAGE_NEXT_BOOTSTRAP_RUN=$RUN" \
  --property=WorkingDirectory="$ROOT" --property=Restart=on-failure \
  --property=RestartSec=10 --property=RestartPreventExitStatus="40 41" \
  --property=StartLimitIntervalSec=0 --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.stage_next_bootstrap_controller >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
