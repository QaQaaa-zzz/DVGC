#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.trajectory_mining_controller \
  --run "${TRAJECTORY_MINING_RUN:?TRAJECTORY_MINING_RUN must be set}"
