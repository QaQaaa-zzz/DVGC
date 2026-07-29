#!/usr/bin/env bash
set -euo pipefail

ROOT="${DVGC_ROOT:-/home/qy/DVGC}"
PY="/home/qy/mujoco_playground/.venv/bin/python"
BASE="${APEX_FUNNEL_ROOT:-runs/safe_state_tube_rsi_seed0_20260729/apex}"
SOURCE="$BASE/ranked_model_v2_pilot_16/candidates.pkl"
LEVEL4="$BASE/ranked_model_v2_feedback_16x4"
LEVEL8_BANK="$BASE/ranked_model_v2_funnel_4_to_8/candidates.pkl"
LEVEL8="$BASE/ranked_model_v2_feedback_8branch"
LEVEL32_BANK="$BASE/ranked_model_v2_funnel_8_to_32/candidates.pkl"
LEVEL32="$BASE/ranked_model_v2_feedback_32branch"
FINAL="$BASE/apex_entry_support_v1"
PHASE_RSI="runs/safe_state_tube_rsi_seed0_20260729/phase_balanced_tube_rsi_v1"
SUPPORT="runs/stage_next_bootstrap_seed0_20260720/support_v2/descent_proposal_support_v1.pkl"
TERMINAL="runs/stage_next_reset_v3_seed0_20260723/apex/feedback_bridge_v1/descent_terminal_proposals_current.pkl"
PI_D="runs/stage_experts/descent_tube_seed0_20260716T2330/round_3/train/policy"
PI_L="runs/landing/refinement_seed0/policy"

cd "$ROOT"
mkdir -p "$BASE"
exec 9>"$BASE/.apex_reachability_funnel.lock"
flock -n 9 || { echo "Apex reachability funnel already active" >&2; exit 0; }

wait_for_atomic_report() {
  local report="$1"
  while [[ ! -s "$report" ]]; do sleep 120; done
}

state_count() {
  "$PY" -c 'import sys; from dvgc.bank import SnapshotBank; print(len(SnapshotBank.load(sys.argv[1]).records))' "$1"
}

write_cost() {
  local bank="$1" branches="$2" output="$3" hypothesis="$4"
  if [[ ! -s "$output" ]]; then
    "$PY" -m cli.stage_cost_estimate --output "$output" \
      --unique-states "$(state_count "$bank")" --branches "$branches" --horizon 40 \
      --pilot-fraction .05 --throughput .59 --hypothesis "$hypothesis"
  fi
}

run_audit() {
  local bank="$1" branches="$2" output="$3" seed="$4"
  if [[ ! -s "$output/report.json" ]]; then
    "$PY" -u -m cli.pilot_apex_feedback_candidates \
      --candidate-bank "$bank" --support-bank "$SUPPORT" --terminal-bank "$TERMINAL" \
      --descent-policy "$PI_D" --landing-policy "$PI_L" --output-root "$output" \
      --branches "$branches" --horizon 40 --lookahead 3 --downstream-horizon 200 --seed "$seed"
  fi
}

wait_for_atomic_report "$LEVEL4/report.json"

if [[ ! -s "$LEVEL8_BANK" ]]; then
  "$PY" -m cli.select_exact_branch_survivors --bank "$SOURCE" --report "$LEVEL4/labels.json" \
    --required-branches 4 --next-branches 8 --output-bank "$LEVEL8_BANK" \
    --output-report "$BASE/ranked_model_v2_funnel_4_to_8/selection.json"
fi
mkdir -p "$LEVEL8"
write_cost "$LEVEL8_BANK" 8 "$LEVEL8/cost_estimate.json" \
  "4/4 Apex states retain local next-stage reachability under eight independent branches"
run_audit "$LEVEL8_BANK" 8 "$LEVEL8" 10830000

if [[ ! -s "$LEVEL32_BANK" ]]; then
  "$PY" -m cli.select_exact_branch_survivors --bank "$LEVEL8_BANK" --report "$LEVEL8/labels.json" \
    --required-branches 8 --next-branches 32 --output-bank "$LEVEL32_BANK" \
    --output-report "$BASE/ranked_model_v2_funnel_8_to_32/selection.json"
fi
mkdir -p "$LEVEL32"
write_cost "$LEVEL32_BANK" 32 "$LEVEL32/cost_estimate.json" \
  "8/8 Apex states remain exact local next-stage successes in independent 32-branch audit"
run_audit "$LEVEL32_BANK" 32 "$LEVEL32" 10840000

if [[ ! -s "$FINAL/apex_entry_support_v1.pkl" ]]; then
  "$PY" -m cli.build_stage_tube_from_independent_audit \
    --audit-bank "$LEVEL32_BANK" --audit-report "$LEVEL32/labels.json" \
    --output-tube "$FINAL/apex_entry_support_v1.pkl" --output-report "$FINAL/report.json" \
    --stage apex --branches 32 --evidence-scope local_next_stage \
    --require-teacher-action-evidence
fi

if [[ ! -s "$PHASE_RSI/bank.pkl" ]]; then
  "$PY" -m cli.build_phase_balanced_tube_rsi_bank \
    --takeoff-bank runs/safe_state_tube_rsi_seed0_20260729/takeoff/takeoff_entry_support_v2.pkl \
    --ascent-bank runs/safe_state_tube_rsi_seed0_20260729/ascent/ascent_entry_support_v2.pkl \
    --apex-bank "$FINAL/apex_entry_support_v1.pkl" \
    --descent-bank runs/descent_reachability_network_v3/independent_tube_extension_3x32_20260729/descent_tube_v5.pkl \
    --landing-bank artifacts/landing_tube.pkl \
    --landing-completion-analysis runs/landing/landing_completion_analysis.json \
    --output-bank "$PHASE_RSI/bank.pkl" --output-report "$PHASE_RSI/report.json"
fi

echo "Apex reachability funnel and phase-balanced Tube-RSI preparation complete: $PHASE_RSI/report.json"
