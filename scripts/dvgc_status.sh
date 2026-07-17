#!/usr/bin/env bash
set -euo pipefail
cd /home/qy/DVGC
exec /usr/bin/python3 -m cli.pipeline_watchdog --print-only "$@"
