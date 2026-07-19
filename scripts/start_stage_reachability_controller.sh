#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${STAGE_REACHABILITY_RUN:-runs/stage_reachability_seed0_20260719}"
cd "$ROOT"
exec systemd-run --user --unit=dvgc-stage-reachability-controller \
  --description="DVGC RA-L stage reachability controller" --setenv="STAGE_REACHABILITY_RUN=$RUN" \
  --property=WorkingDirectory="$ROOT" --property=Restart=on-failure --property=RestartSec=10 \
  --property=RestartPreventExitStatus="40 41" --property=StartLimitIntervalSec=0 --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec '$ROOT/scripts/run_stage_reachability_pipeline.sh' >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
