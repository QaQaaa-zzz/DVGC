#!/usr/bin/env bash
set -euo pipefail

ROOT="${DVGC_ROOT:-/home/qy/DVGC}"
PY="/home/qy/mujoco_playground/.venv/bin/python"
PHASE_ROOT="runs/safe_state_tube_rsi_seed0_20260729/phase_balanced_tube_rsi_v2"
PHASE_BANK="$PHASE_ROOT/bank.pkl"
PHASE_REPORT="$PHASE_ROOT/report.json"
APEX_BANK="runs/safe_state_tube_rsi_seed0_20260729/apex/apex_entry_support_v1/apex_entry_support_v1.pkl"
APEX_REPORT="runs/safe_state_tube_rsi_seed0_20260729/apex/apex_entry_support_v1/report.json"
DESCENT_V6="runs/descent_reachability_network_v3/tube_v6_schema_normalization_20260729/descent_tube_v6.pkl"
DESCENT_V6_REPORT="runs/descent_reachability_network_v3/tube_v6_schema_normalization_20260729/report.json"
DESCENT_V6_VERIFY="runs/descent_reachability_network_v3/tube_v6_schema_normalization_20260729/verification_v2.json"
COMPAT="runs/safe_state_tube_rsi_seed0_20260729/phase_expert_compatibility_v1/report.json"
RUN="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v2"
AUDIT_RUN="$RUN/final_jel_audit_v2"
TEACHERS="$RUN/distillation/teacher_dataset.pkl"
TEACHER_REPORT="$RUN/distillation/teacher_report.json"
DISTILLED="$RUN/distillation/policy"
DISTILL_REPORT="$RUN/distillation/report.json"
PREFLIGHT="$RUN/preflight/report.json"
PILOT="$RUN/joint_rsi_pilot_4096_seed0"
STATE="$RUN/controller_state.json"
C_L="runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl"

cd "$ROOT"
mkdir -p "$RUN"
exec 9>"$RUN/.final_shared_pipeline.lock"
flock -n 9 || { echo "Final shared-policy pipeline already active" >&2; exit 0; }

write_state() {
  local stage="$1" status="$2" next="$3" error="${4:-}"
  "$PY" -c 'import sys,time; from dvgc.runtime import save_json; save_json(sys.argv[1], {"updated_at":time.time(),"current_stage":sys.argv[2],"status":sys.argv[3],"next_automatic_action":sys.argv[4],"last_error":sys.argv[5],"PPO_authorization":sys.argv[2] in ("joint_rsi_pilot","pilot_complete")})' \
    "$STATE" "$stage" "$status" "$next" "$error"
}

write_state waiting_for_apex active build_phase_balanced_tube_rsi_v2
while [[ ! -s "$APEX_BANK" || ! -s "$APEX_REPORT" ]]; do
  if ! systemctl --user is-active --quiet dvgc-apex-reachability-funnel-v3.service; then
    write_state apex_funnel_blocked gate_pause inspect_apex_funnel "Apex controller ended before atomic phase-balanced bank/report"
    exit 40
  fi
  sleep 120
done

if [[ ! -s "$DESCENT_V6" || ! -s "$DESCENT_V6_REPORT" || ! -s "$DESCENT_V6_VERIFY" ]]; then
  write_state descent_tube_v6_missing engineering_failure verify_descent_tube_v6 \
    "normalized Descent Tube v6 or its PASS verification is absent"
  exit 2
fi
if ! "$PY" -c '
import json, sys
from dvgc.config import file_sha256
tube, normalization_path, verification_path = sys.argv[1:]
normalization = json.load(open(normalization_path))
verification = json.load(open(verification_path))
actual = file_sha256(tube)
assert normalization.get("status") == "PASS"
assert verification.get("status") == "PASS"
assert normalization.get("output_bank_sha256") == actual
assert verification.get("tube_sha256") == actual
assert verification.get("policy_identity_hash") == normalization.get("policy_identity_hash")
' "$DESCENT_V6" "$DESCENT_V6_REPORT" "$DESCENT_V6_VERIFY"; then
  write_state descent_tube_v6_invalid gate_pause inspect_descent_tube_v6 \
    "normalized Descent Tube v6 verification/status/hash identity is invalid"
  exit 40
fi

if [[ ! -s "$PHASE_BANK" && ! -s "$PHASE_REPORT" ]]; then
  write_state phase_balanced_bank_v2 active teacher_extraction
  "$PY" -m cli.build_phase_balanced_tube_rsi_bank \
    --takeoff-bank runs/safe_state_tube_rsi_seed0_20260729/takeoff/takeoff_entry_support_v2.pkl \
    --ascent-bank runs/safe_state_tube_rsi_seed0_20260729/ascent/ascent_entry_support_v2.pkl \
    --apex-bank "$APEX_BANK" --descent-bank "$DESCENT_V6" \
    --landing-bank artifacts/landing_tube.pkl \
    --landing-completion-analysis runs/landing/landing_completion_analysis.json \
    --output-bank "$PHASE_BANK" --output-report "$PHASE_REPORT"
elif [[ ! -s "$PHASE_BANK" || ! -s "$PHASE_REPORT" ]]; then
  write_state phase_balanced_bank_v2 engineering_failure inspect_partial_phase_bank \
    "partial phase-balanced v2 bank/report; refusing overwrite"
  exit 2
fi

if [[ ! -s "$TEACHERS" && ! -s "$TEACHER_REPORT" ]]; then
  write_state teacher_extraction active bounded_distillation
  "$PY" -m cli.build_phase_balanced_teacher_dataset \
    --phase-bank "$PHASE_BANK" --expert-compatibility "$COMPAT" \
    --output-dataset "$TEACHERS" --output-report "$TEACHER_REPORT"
elif [[ ! -s "$TEACHERS" || ! -s "$TEACHER_REPORT" ]]; then
  write_state teacher_extraction engineering_failure inspect_partial_teacher_artifact "partial teacher artifact; refusing overwrite"
  exit 2
fi

if [[ ! -d "$DISTILLED" && ! -s "$DISTILL_REPORT" ]]; then
  write_state bounded_distillation active unified_rsi_preflight
  "$PY" -m cli.train_phase_balanced_distillation \
    --teacher-dataset "$TEACHERS" --base-policy runs/landing/refinement_seed0/policy \
    --output-policy "$DISTILLED" --output-report "$DISTILL_REPORT" \
    --steps 500 --learning-rate 3e-5
elif [[ ! -d "$DISTILLED" || ! -s "$DISTILL_REPORT" ]]; then
  write_state bounded_distillation engineering_failure inspect_partial_distillation "partial distillation artifact; refusing overwrite"
  exit 2
fi

distill_status="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$DISTILL_REPORT")"
if [[ "$distill_status" != "PASS" ]]; then
  write_state distillation_downstream_fidelity gate_pause \
    repair_distillation_without_PPO "$distill_status"
  exit 40
fi

if [[ ! -s "$PREFLIGHT" ]]; then
  write_state unified_rsi_preflight active joint_rsi_pilot
  "$PY" -m cli.preflight_phase_balanced_unified_rsi \
    --phase-bank "$PHASE_BANK" --phase-bank-report "$PHASE_REPORT" \
    --policy "$DISTILLED" --output "$PREFLIGHT"
fi

if [[ ! -s "$PILOT/report.json" ]]; then
  write_state joint_rsi_pilot active fixed_phase_evaluation
  set +e
  "$PY" -u -m cli.train_phase_balanced_unified_rsi_pilot \
    --phase-bank "$PHASE_BANK" --initial-policy "$DISTILLED" \
    --canonical-entry-bank "$C_L" --preflight "$PREFLIGHT" \
    --teacher-dataset "$TEACHERS" --run "$PILOT" --seed 0
  code=$?
  set -e
  if [[ $code -eq 40 ]]; then
    write_state distillation_retention_blocker gate_pause repair_distillation_without_PPO "Fixed Landing/Descent start gate failed; PPO was not started"
    exit 40
  elif [[ $code -ne 0 ]]; then
    write_state joint_rsi_pilot engineering_failure resume_from_atomic_checkpoint "pilot exited with code $code"
    exit "$code"
  fi
fi

status="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PILOT/report.json")"
if [[ "$status" == "PASS_PROMOTE" ]]; then
  write_state pilot_complete pass expanded_fixed_phase_evaluation
  "$PY" -c '
import sys
from dvgc.runtime import save_json
save_json("runs/ACTIVE_PIPELINE.json", {
    "run_path": sys.argv[1],
    "controller_unit": "dvgc-final-shared-jel-audit-v2.service",
    "start_script": "scripts/start_final_shared_v2_followons.sh",
    "status": "ACTIVE",
})
' "$AUDIT_RUN"
else
  write_state pilot_no_promotion gate_pause diagnose_reward_reset_action_drift_without_budget_increase "$status"
  exit 40
fi
