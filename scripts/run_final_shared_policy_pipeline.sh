#!/usr/bin/env bash
set -euo pipefail

ROOT="${DVGC_ROOT:-/home/qy/DVGC}"
PY="/home/qy/mujoco_playground/.venv/bin/python"
PHASE_ROOT="runs/safe_state_tube_rsi_seed0_20260729/phase_balanced_tube_rsi_v1"
PHASE_BANK="$PHASE_ROOT/bank.pkl"
PHASE_REPORT="$PHASE_ROOT/report.json"
COMPAT="runs/safe_state_tube_rsi_seed0_20260729/phase_expert_compatibility_v1/report.json"
RUN="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v1"
TEACHERS="$RUN/distillation/teacher_dataset.pkl"
TEACHER_REPORT="$RUN/distillation/teacher_report.json"
DISTILLED="$RUN/distillation/policy"
DISTILL_REPORT="$RUN/distillation/report.json"
PREFLIGHT="$RUN/preflight/report.json"
PILOT="$RUN/joint_rsi_pilot_5120_seed0"
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

write_state waiting_for_apex active build_phase_balanced_teacher_dataset
while [[ ! -s "$PHASE_BANK" || ! -s "$PHASE_REPORT" ]]; do
  if ! systemctl --user is-active --quiet dvgc-apex-reachability-funnel-v3.service; then
    write_state apex_funnel_blocked gate_pause inspect_apex_funnel "Apex controller ended before atomic phase-balanced bank/report"
    exit 40
  fi
  sleep 120
done

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
else
  write_state pilot_no_promotion gate_pause diagnose_reward_reset_action_drift_without_budget_increase "$status"
  exit 40
fi
