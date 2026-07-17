#!/usr/bin/env bash
set -euo pipefail
while true; do
  clear || true
  bash /home/qy/DVGC/scripts/dvgc_status.sh "$@"
  sleep 5
done
