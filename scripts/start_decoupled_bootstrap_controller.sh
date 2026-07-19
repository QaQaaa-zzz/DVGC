#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${DECOUPLED_BOOTSTRAP_RUN:-runs/decoupled_bootstrap_seed0_20260720}"
cd "$ROOT"
exec systemd-run --user --unit=dvgc-decoupled-bootstrap-controller \
  --description="DVGC decoupled bootstrap expert controller" \
  --setenv="DECOUPLED_BOOTSTRAP_RUN=$RUN" \
  --property=WorkingDirectory="$ROOT" --property=Restart=on-failure \
  --property=RestartSec=10 --property=RestartPreventExitStatus="40 41" \
  --property=StartLimitIntervalSec=0 --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec '$ROOT/scripts/run_decoupled_bootstrap_pipeline.sh' >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
