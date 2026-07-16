#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/qy/mujoco_playground/.venv/bin/python}"
RUN="${DESCENT_LOCAL_RUN:-runs/stage_experts/descent_local_nonfinite_repair_seed0_20260716T1825}"

cd "$ROOT"
exec "$PYTHON" -u -m cli.descent_local_controller --run "$RUN"
