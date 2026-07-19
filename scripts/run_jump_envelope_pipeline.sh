#!/usr/bin/env bash
set -euo pipefail
cd /home/qy/DVGC
exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.jump_envelope_controller --run "${JUMP_ENVELOPE_RUN:?}"
