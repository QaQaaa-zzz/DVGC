#!/usr/bin/env bash
set -euo pipefail
cd /home/qy/DVGC
RUN="${STAGE_REACHABILITY_RUN:-runs/stage_reachability_seed0_20260719}"
exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.stage_reachability_controller --run "$RUN"
