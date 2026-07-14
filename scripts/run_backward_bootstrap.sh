#!/usr/bin/env bash
set -euo pipefail

# Formal shared-Actor path:
# geometric bootstrap -> frozen-policy Tube -> Tube-guided RSI refinement ->
# fresh frozen-policy Tube -> independent audit.  Bootstrap and refinement
# split the previous per-stage PPO budget.  The intermediate certification
# branches are additional environment interactions and must be reported.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/qy/mujoco_playground/.venv/bin/python}"
CFG="${CFG:-configs/default.json}"
NUM_ENVS="${NUM_ENVS:-320}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
BATCH_SIZE="${BATCH_SIZE:-80}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
TRAIN_LAYOUT=(
  --num-envs "$NUM_ENVS"
  --num-eval-envs "$NUM_EVAL_ENVS"
  --batch-size "$BATCH_SIZE"
  --num-minibatches "$NUM_MINIBATCHES"
)

cd "$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
bash scripts/local_preflight.sh
"$PYTHON" -m cli.runtime_gate --config "$CFG" --output docs/RUNTIME_GATE.json --check-only

run_stage() {
  local phase="$1"
  local target="$2"
  local bootstrap_steps="$3"
  local refine_steps="$4"
  local downstream_bank="$5"
  local resume_policy="$6"
  local candidates="artifacts/${phase}_candidates.pkl"
  local bootstrap_run="runs/${phase}_bootstrap"
  local bootstrap_tube="artifacts/${phase}_bootstrap_tube.pkl"
  local final_run="runs/${phase}"
  local final_tube="artifacts/${phase}_tube.pkl"
  local downstream_args=()
  local resume_args=()

  if [[ -n "$downstream_bank" ]]; then
    downstream_args=(--downstream-bank "$downstream_bank")
  fi
  if [[ -n "$resume_policy" ]]; then
    resume_args=(--resume "$resume_policy")
  fi

  "$PYTHON" -m cli.build_candidates \
    --phase "$phase" --target "$target" --bank "$candidates" --config "$CFG"
  "$PYTHON" -m cli.train \
    --stage "$phase" --bank "$candidates" "${downstream_args[@]}" \
    --config "$CFG" --run "$bootstrap_run" "${resume_args[@]}" \
    --timesteps "$bootstrap_steps" "${TRAIN_LAYOUT[@]}"
  "$PYTHON" -m cli.certify \
    --phase "$phase" --policy "$bootstrap_run/policy" \
    --candidate-bank "$candidates" "${downstream_args[@]}" \
    --output-bank "$bootstrap_tube"

  "$PYTHON" -m cli.train \
    --stage "$phase" --bank "$bootstrap_tube" "${downstream_args[@]}" \
    --config "$CFG" --run "$final_run" --resume "$bootstrap_run/policy" \
    --require-final-safe-rsi --timesteps "$refine_steps" "${TRAIN_LAYOUT[@]}"
  "$PYTHON" -m cli.certify \
    --phase "$phase" --policy "$final_run/policy" \
    --candidate-bank "$bootstrap_tube" "${downstream_args[@]}" \
    --output-bank "$final_tube"
  "$PYTHON" -m cli.audit \
    --phase "$phase" --policy "$final_run/policy" --bank "$final_tube" \
    "${downstream_args[@]}" --output "$final_run/audit.json"
}

run_stage landing 96 600000 400000 "" ""
run_stage flight 160 720000 480000 \
  artifacts/landing_tube.pkl runs/landing/policy
run_stage takeoff 180 900000 600000 \
  artifacts/flight_tube.pkl runs/flight/policy
run_stage approach 160 1080000 720000 \
  artifacts/takeoff_tube.pkl runs/takeoff/policy

"$PYTHON" -m cli.evaluate \
  --stage full --policy runs/approach/policy --episodes 200 \
  --output runs/natural_start_evaluation.json
