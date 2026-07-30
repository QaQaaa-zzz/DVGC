#!/usr/bin/env bash
set -euo pipefail

ROOT="${DVGC_ROOT:-/home/qy/DVGC}"
PY="/home/qy/mujoco_playground/.venv/bin/python"
BASE="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v3"
PHASE_ROOT="runs/safe_state_tube_rsi_seed0_20260729/phase_balanced_tube_rsi_v2"
PHASE_BANK="$PHASE_ROOT/bank.pkl"
PHASE_REPORT="$PHASE_ROOT/report.json"
ANCHOR="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v2/distillation_all_phase_v2/policy"
TEACHERS="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v2/distillation/teacher_dataset.pkl"
DESCENT_TUBE="runs/descent_reachability_network_v3/tube_v6_schema_normalization_20260729/descent_tube_v6.pkl"
DESCENT_SUPPORT="runs/stage_next_bootstrap_seed0_20260720/support_v2/descent_proposal_support_v1.pkl"
C_L="runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl"
PREFLIGHT="$BASE/preflight_apex_stable_contract_v1/report.json"
PILOT="$BASE/corrected_apex_contract_pilot_4096_seed2"
STATE="$BASE/controller_state.json"

cd "$ROOT"
mkdir -p "$BASE"
exec 9>"$BASE/.corrected_apex_unified_rsi.lock"
flock -n 9 || { echo "Corrected Apex unified RSI controller already active" >&2; exit 0; }

write_state() {
  local stage="$1" status="$2" next="$3" error="${4:-}"
  "$PY" -c 'import sys,time; from dvgc.runtime import save_json; save_json(sys.argv[1], {"updated_at":time.time(),"current_stage":sys.argv[2],"status":sys.argv[3],"next_automatic_action":sys.argv[4],"last_error":sys.argv[5],"PPO_authorization":sys.argv[2]=="corrected_apex_unified_rsi_pilot"})' \
    "$STATE" "$stage" "$status" "$next" "$error"
}

if [[ ! -s "$PREFLIGHT" ]]; then
  write_state corrected_apex_preflight active corrected_apex_unified_rsi_pilot
  mkdir -p "$(dirname "$PREFLIGHT")"
  "$PY" -m cli.preflight_phase_balanced_unified_rsi \
    --phase-bank "$PHASE_BANK" --phase-bank-report "$PHASE_REPORT" \
    --policy "$ANCHOR" --descent-tube "$DESCENT_TUBE" \
    --descent-entry-support-bank "$DESCENT_SUPPORT" --output "$PREFLIGHT"
fi

if [[ ! -s "$PILOT/report.json" ]]; then
  write_state corrected_apex_unified_rsi_pilot active fixed_final_promotion_decision
  set +e
  "$PY" -u -m cli.train_phase_balanced_unified_rsi_pilot \
    --phase-bank "$PHASE_BANK" --initial-policy "$ANCHOR" --anchor-policy "$ANCHOR" \
    --canonical-entry-bank "$C_L" --preflight "$PREFLIGHT" \
    --descent-tube "$DESCENT_TUBE" --descent-entry-support-bank "$DESCENT_SUPPORT" \
    --teacher-dataset "$TEACHERS" --run "$PILOT" --seed 2
  code=$?
  set -e
  if [[ $code -ne 0 ]]; then
    write_state corrected_apex_unified_rsi_pilot engineering_failure inspect_atomic_pilot "pilot exited with code $code"
    exit "$code"
  fi
fi

status="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PILOT/report.json")"
if [[ "$status" != "PASS_PROMOTE" ]]; then
  write_state corrected_apex_pilot_no_promotion gate_pause diagnose_fixed_final_without_more_budget "$status"
  exit 40
fi

write_state corrected_apex_pilot_complete pass final_shared_policy_jel_audit_v6
"$PY" -c '
from dvgc.runtime import save_json
save_json("runs/ACTIVE_PIPELINE.json", {
  "run_path": "runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v3/final_jel_audit_v6",
  "controller_unit": "dvgc-final-shared-jel-audit-v3.service",
  "start_script": "scripts/start_corrected_apex_unified_rsi_followons.sh",
  "status": "ACTIVE",
})
'
systemctl --user --no-block start dvgc-final-shared-jel-audit-v3.service

