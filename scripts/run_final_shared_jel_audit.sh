#!/usr/bin/env bash
set -euo pipefail

ROOT="${DVGC_ROOT:-/home/qy/DVGC}"
PY="/home/qy/mujoco_playground/.venv/bin/python"
BASE="runs/safe_state_tube_rsi_seed0_20260729/final_shared_policy_v1"
PHASE_BANK="runs/safe_state_tube_rsi_seed0_20260729/phase_balanced_tube_rsi_v1/bank.pkl"
PILOT="$BASE/joint_rsi_pilot_5120_seed0"
POLICY="$PILOT/policy"
C_L="runs/stage_experts/flight_seed0_20260715T2045/bridge_recovery/entry_set_bridge.pkl"
AUDIT="$BASE/final_jel_audit_v1"
LEVEL4="$AUDIT/final_4branch.json"
LEVEL8_BANK="$AUDIT/funnel_4_to_8/candidates.pkl"
LEVEL8="$AUDIT/final_8branch.json"
LEVEL32_BANK="$AUDIT/funnel_8_to_32/candidates.pkl"
LEVEL32="$AUDIT/final_32branch_construction.json"
INDEPENDENT="$AUDIT/final_32branch_independent.json"
JEL="$AUDIT/final_shared_policy_jel.pkl"
REPORT="$AUDIT/report.json"
STATE="$AUDIT/controller_state.json"

cd "$ROOT"
mkdir -p "$AUDIT"
exec 9>"$AUDIT/.final_jel_audit.lock"
flock -n 9 || { echo "Final JEL audit controller already active" >&2; exit 0; }

write_state() {
  local stage="$1" status="$2" next="$3" error="${4:-}"
  "$PY" -c 'import sys,time; from dvgc.runtime import save_json; save_json(sys.argv[1], {"updated_at":time.time(),"current_stage":sys.argv[2],"status":sys.argv[3],"next_automatic_action":sys.argv[4],"last_error":sys.argv[5],"formal_jel_complete":sys.argv[3]=="pipeline_complete"})' \
    "$STATE" "$stage" "$status" "$next" "$error"
}

count_states() {
  "$PY" -c 'import sys; from dvgc.bank import SnapshotBank; print(len(SnapshotBank.load(sys.argv[1]).records))' "$1"
}

write_cost() {
  local bank="$1" branches="$2" output="$3" hypothesis="$4"
  if [[ ! -s "$output" ]]; then
    "$PY" -m cli.stage_cost_estimate --output "$output" \
      --unique-states "$(count_states "$bank")" --branches "$branches" --horizon 400 \
      --pilot-fraction .05 --throughput .59 --hypothesis "$hypothesis"
  fi
}

write_state waiting_for_promoted_pilot active final_4branch_screen
while [[ ! -s "$PILOT/report.json" ]]; do
  if ! systemctl --user is-active --quiet dvgc-final-shared-policy-v1.service; then
    write_state unified_pilot_missing gate_pause inspect_unified_pipeline "Unified controller ended before atomic pilot report"
    exit 40
  fi
  sleep 120
done
pilot_status="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$PILOT/report.json")"
if [[ "$pilot_status" != "PASS_PROMOTE" ]]; then
  write_state unified_pilot_not_promoted gate_pause diagnose_unified_pilot_without_formal_audit "$pilot_status"
  exit 40
fi

if [[ ! -s "$LEVEL4" ]]; then
  write_state final_4branch_screen active final_8branch_screen
  write_cost "$PHASE_BANK" 4 "$AUDIT/cost_4branch.json" \
    "promoted shared Actor has nonzero parent-diverse Final-Recovery coverage"
  "$PY" -u -m cli.audit_final_shared_policy_candidates \
    --policy "$POLICY" --candidate-bank "$PHASE_BANK" --canonical-entry-bank "$C_L" \
    --output "$LEVEL4" --branches 4 --seed 10900000 --namespace final-shared-construction-4
fi
if [[ ! -s "$LEVEL8_BANK" ]]; then
  if ! "$PY" -m cli.select_exact_branch_survivors --bank "$PHASE_BANK" --report "$LEVEL4" \
    --required-branches 4 --next-branches 8 --output-bank "$LEVEL8_BANK" \
    --output-report "$AUDIT/funnel_4_to_8/selection.json"; then
    write_state final_4branch_screen gate_pause diagnose_shared_actor_zero_exact_final_support "no exact 4/4 Final-Recovery state"
    exit 40
  fi
fi
if [[ ! -s "$LEVEL8" ]]; then
  write_state final_8branch_screen active final_32branch_construction
  write_cost "$LEVEL8_BANK" 8 "$AUDIT/cost_8branch.json" \
    "exact 4/4 shared-policy Final states retain Final-Recovery over eight branches"
  "$PY" -u -m cli.audit_final_shared_policy_candidates \
    --policy "$POLICY" --candidate-bank "$LEVEL8_BANK" --canonical-entry-bank "$C_L" \
    --output "$LEVEL8" --branches 8 --seed 11000000 --namespace final-shared-construction-8
fi
if [[ ! -s "$LEVEL32_BANK" ]]; then
  if ! "$PY" -m cli.select_exact_branch_survivors --bank "$LEVEL8_BANK" --report "$LEVEL8" \
    --required-branches 8 --next-branches 32 --output-bank "$LEVEL32_BANK" \
    --output-report "$AUDIT/funnel_8_to_32/selection.json"; then
    write_state final_8branch_screen gate_pause diagnose_shared_actor_nonrobust_final_support "no exact 8/8 Final-Recovery state"
    exit 40
  fi
fi
if [[ ! -s "$LEVEL32" ]]; then
  write_state final_32branch_construction active final_32branch_independent_audit
  write_cost "$LEVEL32_BANK" 32 "$AUDIT/cost_32branch_construction.json" \
    "exact 8/8 shared-policy Final states pass 32-branch construction certification"
  "$PY" -u -m cli.audit_final_shared_policy_candidates \
    --policy "$POLICY" --candidate-bank "$LEVEL32_BANK" --canonical-entry-bank "$C_L" \
    --output "$LEVEL32" --branches 32 --seed 11100000 --namespace final-shared-construction-32
fi
if [[ ! -s "$INDEPENDENT" ]]; then
  write_state final_32branch_independent_audit active freeze_final_shared_policy_jel
  write_cost "$LEVEL32_BANK" 32 "$AUDIT/cost_32branch_independent.json" \
    "construction-safe states retain Final-Recovery under a disjoint independent audit namespace"
  "$PY" -u -m cli.audit_final_shared_policy_candidates \
    --policy "$POLICY" --candidate-bank "$LEVEL32_BANK" --canonical-entry-bank "$C_L" \
    --output "$INDEPENDENT" --branches 32 --seed 11200000 --namespace final-shared-independent-32
fi
if [[ ! -s "$JEL" && ! -s "$REPORT" ]]; then
  write_state freeze_final_shared_policy_jel active pipeline_complete
  if ! "$PY" -m cli.freeze_final_shared_policy_jel \
    --bank "$LEVEL32_BANK" --construction-report "$LEVEL32" \
    --independent-audit-report "$INDEPENDENT" --output-bank "$JEL" \
    --output-report "$REPORT" --branches 32; then
    write_state freeze_final_shared_policy_jel gate_pause diagnose_shared_actor_independent_recertification_failure "no state passed both disjoint 32-branch Final audits"
    exit 40
  fi
elif [[ ! -s "$JEL" || ! -s "$REPORT" ]]; then
  write_state freeze_final_shared_policy_jel engineering_failure inspect_partial_jel "partial final JEL artifact; refusing overwrite"
  exit 2
fi
write_state pipeline_complete pipeline_complete none
