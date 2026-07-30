#!/usr/bin/env bash
set -euo pipefail

ROOT="${DVGC_ROOT:-/home/qy/DVGC}"
RUN="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v2"
AUDIT="$RUN/final_jel_audit_v5"
PY="/home/qy/mujoco_playground/.venv/bin/python"

cd "$ROOT"
mkdir -p "$RUN" "$AUDIT"

start_unit() {
  local unit="$1" script="$2" log="$3"
  if systemctl --user is-active --quiet "$unit"; then
    return
  fi
  if [[ "$(systemctl --user show "$unit" -p LoadState --value 2>/dev/null || true)" == "loaded" ]]; then
    systemctl --user --no-block start "$unit"
    return
  fi
  systemd-run --user --unit="${unit%.service}" \
    --property=Type=simple --property=Restart=no --property=RuntimeMaxSec=infinity \
    --property="StandardOutput=append:$ROOT/$log" \
    --property="StandardError=append:$ROOT/$log" \
    --working-directory="$ROOT" bash "$script"
}

start_unit dvgc-final-shared-policy-v2.service \
  scripts/run_final_shared_policy_pipeline.sh "$RUN/controller.log"
start_unit dvgc-final-shared-jel-audit-v2.service \
  scripts/run_final_shared_jel_audit.sh "$AUDIT/controller.log"

"$PY" -c '
import sys
from dvgc.runtime import save_json
save_json("runs/ACTIVE_PIPELINE.json", {
    "run_path": sys.argv[1],
    "controller_unit": "dvgc-final-shared-policy-v2.service",
    "start_script": "scripts/start_final_shared_v2_followons.sh",
    "status": "ACTIVE",
})
' "$RUN"
