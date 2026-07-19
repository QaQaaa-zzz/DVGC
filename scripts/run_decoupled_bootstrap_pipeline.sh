#!/usr/bin/env bash
set -euo pipefail
cd /home/qy/DVGC
RUN="${DECOUPLED_BOOTSTRAP_RUN:-runs/decoupled_bootstrap_seed0_20260720}"
exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.decoupled_bootstrap_controller --run "$RUN"
