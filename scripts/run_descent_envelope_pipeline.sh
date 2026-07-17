#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.descent_envelope_controller \
  --run "${DESCENT_ENVELOPE_RUN:?DESCENT_ENVELOPE_RUN must be set}"
