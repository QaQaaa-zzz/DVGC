#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${DESCENT_ENVELOPE_RUN:?DESCENT_ENVELOPE_RUN must be set}"
cd "$ROOT"
exec systemd-run --user --unit=dvgc-descent-envelope-controller \
  --description="DVGC stable descent envelope controller" \
  --setenv="DESCENT_ENVELOPE_RUN=$RUN" \
  --property=WorkingDirectory="$ROOT" --property=Restart=on-failure --property=RestartSec=10 \
  --property=RestartPreventExitStatus="40 41" --property=StartLimitIntervalSec=0 \
  --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec '$ROOT/scripts/run_descent_envelope_pipeline.sh' >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
