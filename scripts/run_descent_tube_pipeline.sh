#!/usr/bin/env bash
set -euo pipefail
cd "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec /home/qy/mujoco_playground/.venv/bin/python -u -m cli.descent_tube_controller --run "${DESCENT_TUBE_RUN:-runs/stage_experts/descent_tube_seed0_20260716T2330}"
