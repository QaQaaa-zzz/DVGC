#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${JUMP_ENVELOPE_RUN:-runs/jump_envelope_seed0_20260719}"
cd "$ROOT"
exec systemd-run --user --unit=dvgc-jump-envelope-controller \
  --description="DVGC handoff-first jump envelope controller" --setenv="JUMP_ENVELOPE_RUN=$RUN" \
  --property=WorkingDirectory="$ROOT" --property=Restart=on-failure --property=RestartSec=10 \
  --property=RestartPreventExitStatus="40 41" --property=StartLimitIntervalSec=0 --property=RuntimeMaxSec=infinity \
  /bin/bash -lc "exec '$ROOT/scripts/run_jump_envelope_pipeline.sh' >> '$ROOT/$RUN/systemd-controller.stdout.log' 2>&1"
