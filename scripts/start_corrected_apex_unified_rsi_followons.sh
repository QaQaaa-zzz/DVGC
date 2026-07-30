#!/usr/bin/env bash
set -euo pipefail

ROOT="${DVGC_ROOT:-/home/qy/DVGC}"
RUN="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v3"
PY="/home/qy/mujoco_playground/.venv/bin/python"
cd "$ROOT"
mkdir -p "$RUN" "$RUN/final_jel_audit_v6"

start_unit() {
  local unit="$1" script="$2" log="$3"
  if systemctl --user is-active --quiet "$unit"; then return; fi
  if [[ "$(systemctl --user show "$unit" -p LoadState --value 2>/dev/null || true)" == "loaded" ]]; then
    systemctl --user --no-block start "$unit"
    return
  fi
  shift 3
  local env_args=()
  for value in "$@"; do env_args+=(--setenv="$value"); done
  systemd-run --user --unit="${unit%.service}" --property=Type=simple \
    --property=Restart=no --property=RuntimeMaxSec=infinity \
    --property="StandardOutput=append:$ROOT/$log" \
    --property="StandardError=append:$ROOT/$log" \
    --working-directory="$ROOT" "${env_args[@]}" bash "$script"
}

start_unit dvgc-final-shared-policy-v3.service \
  scripts/run_corrected_apex_unified_rsi_pipeline.sh "$RUN/controller.log"
start_unit dvgc-final-shared-jel-audit-v3.service \
  scripts/run_final_shared_jel_audit.sh "$RUN/final_jel_audit_v6/controller.log" \
  FINAL_SHARED_BASE="$RUN" \
  FINAL_SHARED_PILOT="$RUN/corrected_apex_contract_pilot_4096_seed2" \
  FINAL_SHARED_AUDIT="$RUN/final_jel_audit_v6" \
  FINAL_SHARED_POLICY_UNIT=dvgc-final-shared-policy-v3.service

"$PY" -c '
from dvgc.runtime import save_json
save_json("runs/ACTIVE_PIPELINE.json", {
  "run_path": "runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v3",
  "controller_unit": "dvgc-final-shared-policy-v3.service",
  "start_script": "scripts/start_corrected_apex_unified_rsi_followons.sh",
  "status": "ACTIVE",
})
'

