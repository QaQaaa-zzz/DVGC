#!/usr/bin/env bash
set -euo pipefail

# Resumable seed-0 RA-L core: frozen Landing -> Flight -> Takeoff -> Approach
# -> natural-start evaluation.  Every expensive command is hash-bound to its
# inputs and refuses to overwrite stale outputs.

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/qy/mujoco_playground/.venv/bin/python}"
CFG="${CFG:-configs/default.json}"
REVISION="${PIPELINE_REVISION:-v1}"
STATE_ROOT="runs/remaining_pipeline/${REVISION}"
MARKER_ROOT="${STATE_ROOT}/markers"
LOG_ROOT="${STATE_ROOT}/logs"
CERT_CHUNK_STATES="${CERT_CHUNK_STATES:-40}"
AUDIT_CHUNK_STATES="${AUDIT_CHUNK_STATES:-40}"
PILOT_STEPS="${PILOT_STEPS:-100000}"
CANDIDATE_ATTEMPT_BUDGET="${CANDIDATE_ATTEMPT_BUDGET:-450}"
CANDIDATE_DEDUP_DISTANCE="${CANDIDATE_DEDUP_DISTANCE:-0.03}"
NUM_ENVS="${NUM_ENVS:-320}"
NUM_EVAL_ENVS="${NUM_EVAL_ENVS:-128}"
BATCH_SIZE="${BATCH_SIZE:-80}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-4}"
MAX_TIMEOUT="${MAX_TIMEOUT:-0.05}"
MIN_AUDIT_PRECISION="${MIN_AUDIT_PRECISION:-0.95}"

cd "$ROOT"
mkdir -p "$MARKER_ROOT" "$LOG_ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

if [[ ! -x "$PYTHON" ]]; then echo "Configured Python is unavailable: $PYTHON" >&2; exit 2; fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then echo "Tracked worktree changes present; commit or resolve them before pipeline execution" >&2; exit 2; fi
"$PYTHON" -m cli.runtime_gate --config "$CFG" --output docs/RUNTIME_GATE.json --check-only >/dev/null

join_by_semicolon() { local IFS=';'; echo "$*"; }

run_step() {
  local step="$1" signature="$2" input_text="$3" output_text="$4"; shift 4
  local marker="$MARKER_ROOT/${step}.json" log="$LOG_ROOT/${step}.log"
  local -a inputs=() outputs=() marker_args=()
  [[ -n "$input_text" ]] && IFS=';' read -r -a inputs <<< "$input_text"
  [[ -n "$output_text" ]] && IFS=';' read -r -a outputs <<< "$output_text"
  inputs+=("scripts/run_remaining_pipeline.sh" "cli/pipeline_marker.py" "dvgc/pipeline.py" "$CFG" "docs/RUNTIME_GATE.json")
  marker_args=(--marker "$marker" --step "$step" --token "$REVISION" --token "$signature")
  local value
  for value in "${inputs[@]}"; do marker_args+=(--input "$value"); done
  for value in "${outputs[@]}"; do marker_args+=(--output "$value"); done
  if "$PYTHON" -m cli.pipeline_marker check "${marker_args[@]}" >/dev/null 2>&1; then
    echo "[pipeline] skip $step"
    return 0
  fi
  for value in "${outputs[@]}"; do
    if [[ -e "$value" && "${RUN_STEP_ALLOW_EXISTING:-0}" != 1 ]]; then echo "[pipeline] stale/unmarked output blocks $step: $value" >&2; return 3; fi
  done
  if [[ -e "$log" ]]; then echo "[pipeline] prior log blocks $step under revision $REVISION: $log" >&2; return 3; fi
  echo "[pipeline] start $step"
  set +e
  "$@" >"$log" 2>&1
  local status=$?
  set -e
  "$PYTHON" -m cli.pipeline_marker record "${marker_args[@]}" --exit-status "$status" --log "$log" >/dev/null
  if (( status != 0 )); then
    echo "[pipeline] FAIL $step exit=$status log=$log" >&2
    tail -n 60 "$log" >&2
    return "$status"
  fi
  echo "[pipeline] complete $step"
}

certify_chunked() {
  local phase="$1" policy="$2" candidates="$3" downstream="$4" output="$5" seed="$6" namespace="$7" prefix="$8"
  local count start stop index=0 part
  count=$("$PYTHON" -m cli.pipeline_gate bank-count --bank "$candidates" --phase "$phase")
  local -a parts=()
  for ((start=0; start<count; start+=CERT_CHUNK_STATES)); do
    stop=$((start+CERT_CHUNK_STATES)); (( stop>count )) && stop=$count
    part="${output%.pkl}.part${index}.pkl"; parts+=("$part")
    local -a command=("$PYTHON" -u -m cli.certify --phase "$phase" --policy "$policy" --candidate-bank "$candidates" --output-bank "$part" --seed "$seed" --namespace "$namespace" --start-index "$start" --limit "$((stop-start))")
    [[ -n "$downstream" ]] && command+=(--downstream-bank "$downstream")
    local inputs; inputs=$(join_by_semicolon "$policy" "$candidates" ${downstream:+"$downstream"})
    run_step "${prefix}_cert_part${index}" "$phase:$seed:$namespace:$start:$stop" "$inputs" "$(join_by_semicolon "$part" "${part%.pkl}.cert.json")" "${command[@]}"
    index=$((index+1))
  done
  local merge_inputs; merge_inputs=$(join_by_semicolon "${parts[@]}" "${parts[@]/%.pkl/.cert.json}")
  run_step "${prefix}_cert_merge" "$phase:$seed:$namespace:$count" "$merge_inputs" "$(join_by_semicolon "$output" "${output%.pkl}.cert.json")" "$PYTHON" -m cli.merge_certifications --parts "${parts[@]}" --output-bank "$output"
}

audit_chunked() {
  local phase="$1" policy="$2" tube="$3" downstream="$4" run_root="$5"
  local count start stop index=0 part
  count=$("$PYTHON" -m cli.pipeline_gate bank-count --bank "$tube" --phase "$phase")
  local -a parts=()
  for ((start=0; start<count; start+=AUDIT_CHUNK_STATES)); do
    stop=$((start+AUDIT_CHUNK_STATES)); (( stop>count )) && stop=$count
    part="$run_root/audit.part${index}.json"; parts+=("$part")
    local -a command=("$PYTHON" -u -m cli.audit --phase "$phase" --policy "$policy" --bank "$tube" --output "$part" --seed 1000000 --namespace audit --branches 16 --start-index "$start" --limit "$((stop-start))")
    [[ -n "$downstream" ]] && command+=(--downstream-bank "$downstream")
    local inputs; inputs=$(join_by_semicolon "$policy" "$tube" ${downstream:+"$downstream"})
    run_step "${phase}_audit_part${index}" "$phase:1000000:audit:16:$start:$stop" "$inputs" "$part" "${command[@]}"
    index=$((index+1))
  done
  local audit="$run_root/audit.json"
  run_step "${phase}_audit_merge" "$phase:16:$count" "$(join_by_semicolon "${parts[@]}")" "$audit" "$PYTHON" -m cli.merge_audits --parts "${parts[@]}" --output "$audit"
  run_step "${phase}_audit_gate" "$phase:$MIN_AUDIT_PRECISION:$MAX_TIMEOUT" "$audit" "$run_root/audit_analysis.json" "$PYTHON" -m cli.pipeline_gate audit --report "$audit" --minimum-precision "$MIN_AUDIT_PRECISION" --maximum-timeout "$MAX_TIMEOUT" --output "$run_root/audit_analysis.json"
}

stage_value() {
  local phase="$1" flight="$2" takeoff="$3" approach="$4"
  case "$phase" in flight) echo "$flight";; takeoff) echo "$takeoff";; approach) echo "$approach";; esac
}

build_candidates_chunked() {
  local phase="$1" target="$2" bank="$3" chunk previous=-1 stagnant=0 count
  chunk=$("$PYTHON" - "$bank" <<'PY'
import sys
from dvgc.bank import SnapshotBank
print(len(SnapshotBank.load(sys.argv[1]).metadata.get("candidate_build_history",[])))
PY
)
  while true; do
    count=$("$PYTHON" -m cli.pipeline_gate bank-count --bank "$bank" --phase "$phase")
    (( count>=target )) && break
    if (( count==previous )); then stagnant=$((stagnant+1)); else stagnant=0; fi
    if (( stagnant>=2 || chunk>=30 )); then echo "Candidate construction made insufficient progress: $count/$target" >&2; return 2; fi
    previous=$count
    "$PYTHON" -u -m cli.build_candidates --phase "$phase" --target "$target" --bank "$bank" --config "$CFG" --seed "$chunk" --aux-fraction 0 --attempt-budget "$CANDIDATE_ATTEMPT_BUDGET" --dedup-distance "$CANDIDATE_DEDUP_DISTANCE" --allow-partial
    chunk=$((chunk+1))
  done
  "$PYTHON" - <<PY
import json
from pathlib import Path
p=Path("${bank%.pkl}.build.json")
r=json.loads(p.read_text())
if r.get("status")!="PASS": raise SystemExit("Final candidate chunk did not reach PASS")
print(json.dumps({"status":r["status"],"phase":r["phase"],"target":r["target"],"aggregate_attempts":r["aggregate_attempts"],"deduplication_rate":r["deduplication_rate"]}))
PY
}

run_stage() {
  local phase="$1" target="$2" total_bootstrap="$3" refine_steps="$4" downstream="$5" resume_policy="$6"
  local stage_root="runs/${phase}/pipeline_seed0_${REVISION}" candidates="artifacts/${phase}_candidates.pkl"
  local pilot="$stage_root/pilot" formal="$stage_root/bootstrap" refine="$stage_root/refinement"
  local bootstrap_tube="artifacts/${phase}_bootstrap_tube.pkl" final_tube="artifacts/${phase}_tube.pkl"
  local minimum_final learning_rate eval_seed continuation_steps
  minimum_final=$(stage_value "$phase" "${FLIGHT_MIN_FINAL:-0.50}" "${TAKEOFF_MIN_FINAL:-0.35}" "${APPROACH_MIN_FINAL:-0.20}")
  learning_rate=$(stage_value "$phase" "${FLIGHT_LEARNING_RATE:-0.0001}" "${TAKEOFF_LEARNING_RATE:-0.0001}" "${APPROACH_LEARNING_RATE:-0.0001}")
  eval_seed=$(stage_value "$phase" 2100000 2200000 2300000)
  continuation_steps=$((total_bootstrap-PILOT_STEPS))
  mkdir -p "$stage_root"

  RUN_STEP_ALLOW_EXISTING=1 run_step "${phase}_candidates" "$phase:$target:0:$CANDIDATE_ATTEMPT_BUDGET:$CANDIDATE_DEDUP_DISTANCE" "$(join_by_semicolon data/reference_jump.csv "$CFG")" "$(join_by_semicolon "$candidates" "${candidates%.pkl}.build.json")" build_candidates_chunked "$phase" "$target" "$candidates"
  run_step "${phase}_candidate_audit" "$phase:$target:25" "$candidates" "$stage_root/candidate_audit.json" "$PYTHON" -u -m cli.audit_candidates --phase "$phase" --bank "$candidates" --config "$CFG" --expected-count "$target" --horizon 25 --output "$stage_root/candidate_audit.json"
  run_step "${phase}_candidate_gate" "$phase:$target" "$(join_by_semicolon "${candidates%.pkl}.build.json" "$stage_root/candidate_audit.json")" "$stage_root/candidate_decision.json" "$PYTHON" -m cli.pipeline_gate candidate --build "${candidates%.pkl}.build.json" --audit "$stage_root/candidate_audit.json" --output "$stage_root/candidate_decision.json"

  local -a pilot_cmd=("$PYTHON" -u -m cli.train --stage "$phase" --bank "$candidates" --downstream-bank "$downstream" --config "$CFG" --run "$pilot" --resume "$resume_policy" --timesteps "$PILOT_STEPS" --seed 0 --num-envs "$NUM_ENVS" --num-eval-envs "$NUM_EVAL_ENVS" --batch-size "$BATCH_SIZE" --num-minibatches "$NUM_MINIBATCHES" --learning-rate "$learning_rate")
  run_step "${phase}_pilot" "$phase:$PILOT_STEPS:0:$learning_rate" "$(join_by_semicolon "$candidates" "$downstream" "$resume_policy")" "$(join_by_semicolon "$pilot/training_metrics.json" "$pilot/policy")" "${pilot_cmd[@]}"
  run_step "${phase}_pilot_analysis" "$phase" "$(join_by_semicolon "$pilot/training_metrics.json" "$pilot/config.json" "$LOG_ROOT/${phase}_pilot.log")" "$pilot/analysis.json" "$PYTHON" -m cli.analyze_training --run "$pilot" --console-log "$LOG_ROOT/${phase}_pilot.log"
  run_step "${phase}_pilot_eval" "$phase:$target:$eval_seed" "$(join_by_semicolon "$pilot/policy" "$candidates" "$downstream")" "$pilot/fixed_candidate_evaluation.json" "$PYTHON" -u -m cli.evaluate --stage "$phase" --policy "$pilot/policy" --bank "$candidates" --downstream-bank "$downstream" --episodes "$target" --seed "$eval_seed" --output "$pilot/fixed_candidate_evaluation.json"
  run_step "${phase}_pilot_gate" "$phase:$minimum_final:$MAX_TIMEOUT" "$(join_by_semicolon "$pilot/analysis.json" "$pilot/fixed_candidate_evaluation.json")" "$pilot/decision.json" "$PYTHON" -m cli.pipeline_gate training --analysis "$pilot/analysis.json" --evaluation "$pilot/fixed_candidate_evaluation.json" --minimum-final "$minimum_final" --maximum-timeout "$MAX_TIMEOUT" --output "$pilot/decision.json"

  local -a formal_cmd=("$PYTHON" -u -m cli.train --stage "$phase" --bank "$candidates" --downstream-bank "$downstream" --config "$CFG" --run "$formal" --resume "$pilot/policy" --timesteps "$continuation_steps" --seed 0 --num-envs "$NUM_ENVS" --num-eval-envs "$NUM_EVAL_ENVS" --batch-size "$BATCH_SIZE" --num-minibatches "$NUM_MINIBATCHES" --learning-rate "$learning_rate")
  run_step "${phase}_bootstrap" "$phase:$continuation_steps:0:$learning_rate" "$(join_by_semicolon "$candidates" "$downstream" "$pilot/policy" "$pilot/decision.json")" "$(join_by_semicolon "$formal/training_metrics.json" "$formal/policy")" "${formal_cmd[@]}"
  run_step "${phase}_bootstrap_analysis" "$phase" "$(join_by_semicolon "$formal/training_metrics.json" "$formal/config.json" "$LOG_ROOT/${phase}_bootstrap.log")" "$formal/analysis.json" "$PYTHON" -m cli.analyze_training --run "$formal" --console-log "$LOG_ROOT/${phase}_bootstrap.log"
  run_step "${phase}_bootstrap_eval" "$phase:$target:$eval_seed" "$(join_by_semicolon "$formal/policy" "$candidates" "$downstream")" "$formal/fixed_candidate_evaluation.json" "$PYTHON" -u -m cli.evaluate --stage "$phase" --policy "$formal/policy" --bank "$candidates" --downstream-bank "$downstream" --episodes "$target" --seed "$eval_seed" --output "$formal/fixed_candidate_evaluation.json"
  run_step "${phase}_bootstrap_gate" "$phase:$minimum_final:$MAX_TIMEOUT:0.05" "$(join_by_semicolon "$formal/analysis.json" "$formal/fixed_candidate_evaluation.json" "$pilot/fixed_candidate_evaluation.json")" "$formal/decision.json" "$PYTHON" -m cli.pipeline_gate training --analysis "$formal/analysis.json" --evaluation "$formal/fixed_candidate_evaluation.json" --reference-evaluation "$pilot/fixed_candidate_evaluation.json" --minimum-final "$minimum_final" --maximum-timeout "$MAX_TIMEOUT" --maximum-final-drop 0.05 --output "$formal/decision.json"

  certify_chunked "$phase" "$formal/policy" "$candidates" "$downstream" "$bootstrap_tube" 0 build "${phase}_bootstrap"
  run_step "${phase}_bootstrap_cert_gate" "$phase:4" "${bootstrap_tube%.pkl}.cert.json" "$stage_root/bootstrap_certification_analysis.json" "$PYTHON" -m cli.pipeline_gate certification --report "${bootstrap_tube%.pkl}.cert.json" --phase "$phase" --minimum-safe 4 --output "$stage_root/bootstrap_certification_analysis.json"

  local -a refine_cmd=("$PYTHON" -u -m cli.train --stage "$phase" --bank "$bootstrap_tube" --downstream-bank "$downstream" --config "$CFG" --run "$refine" --resume "$formal/policy" --require-final-safe-rsi --timesteps "$refine_steps" --seed 0 --num-envs "$NUM_ENVS" --num-eval-envs "$NUM_EVAL_ENVS" --batch-size "$BATCH_SIZE" --num-minibatches "$NUM_MINIBATCHES" --learning-rate "$learning_rate")
  run_step "${phase}_refinement" "$phase:$refine_steps:0:$learning_rate" "$(join_by_semicolon "$bootstrap_tube" "$downstream" "$formal/policy" "$stage_root/bootstrap_certification_analysis.json")" "$(join_by_semicolon "$refine/training_metrics.json" "$refine/policy")" "${refine_cmd[@]}"
  run_step "${phase}_refinement_analysis" "$phase" "$(join_by_semicolon "$refine/training_metrics.json" "$refine/config.json" "$LOG_ROOT/${phase}_refinement.log")" "$refine/analysis.json" "$PYTHON" -m cli.analyze_training --run "$refine" --console-log "$LOG_ROOT/${phase}_refinement.log"
  run_step "${phase}_refinement_eval" "$phase:$target:$eval_seed" "$(join_by_semicolon "$refine/policy" "$candidates" "$downstream")" "$refine/fixed_candidate_evaluation.json" "$PYTHON" -u -m cli.evaluate --stage "$phase" --policy "$refine/policy" --bank "$candidates" --downstream-bank "$downstream" --episodes "$target" --seed "$eval_seed" --output "$refine/fixed_candidate_evaluation.json"
  run_step "${phase}_refinement_gate" "$phase:$minimum_final:$MAX_TIMEOUT:0.05" "$(join_by_semicolon "$refine/analysis.json" "$refine/fixed_candidate_evaluation.json" "$formal/fixed_candidate_evaluation.json")" "$refine/decision.json" "$PYTHON" -m cli.pipeline_gate training --analysis "$refine/analysis.json" --evaluation "$refine/fixed_candidate_evaluation.json" --reference-evaluation "$formal/fixed_candidate_evaluation.json" --minimum-final "$minimum_final" --maximum-timeout "$MAX_TIMEOUT" --maximum-final-drop 0.05 --output "$refine/decision.json"

  certify_chunked "$phase" "$refine/policy" "$bootstrap_tube" "$downstream" "$final_tube" 2000 recert "${phase}_recert"
  run_step "${phase}_recert_gate" "$phase:4" "${final_tube%.pkl}.cert.json" "$refine/final_recertification_analysis.json" "$PYTHON" -m cli.pipeline_gate certification --report "${final_tube%.pkl}.cert.json" --phase "$phase" --minimum-safe 4 --output "$refine/final_recertification_analysis.json"
  audit_chunked "$phase" "$refine/policy" "$final_tube" "$downstream" "$refine"
}

LANDING_POLICY="runs/landing/refinement_seed0/policy"
LANDING_TUBE="artifacts/landing_tube.pkl"
[[ -d "$LANDING_POLICY" && -f "$LANDING_TUBE" ]] || { echo "Frozen Landing inputs are missing" >&2; exit 2; }

run_stage flight 160 720000 480000 "$LANDING_TUBE" "$LANDING_POLICY"
run_stage takeoff 180 900000 600000 artifacts/flight_tube.pkl "runs/flight/pipeline_seed0_${REVISION}/refinement/policy"
run_stage approach 160 1080000 720000 artifacts/takeoff_tube.pkl "runs/takeoff/pipeline_seed0_${REVISION}/refinement/policy"

APPROACH_POLICY="runs/approach/pipeline_seed0_${REVISION}/refinement/policy"
NATURAL_REPORT="$STATE_ROOT/natural_start_seed0.json"
run_step natural_start_seed0 "full:200:2400000" "$APPROACH_POLICY" "$NATURAL_REPORT" "$PYTHON" -u -m cli.evaluate --stage full --policy "$APPROACH_POLICY" --episodes 200 --seed 2400000 --output "$NATURAL_REPORT"
run_step natural_start_seed0_gate "full:0.10:0.10" "$NATURAL_REPORT" "$STATE_ROOT/natural_start_seed0_analysis.json" "$PYTHON" -m cli.pipeline_gate evaluation --report "$NATURAL_REPORT" --minimum-final 0.10 --maximum-timeout 0.10 --output "$STATE_ROOT/natural_start_seed0_analysis.json"
echo "[pipeline] seed-0 backward chain and natural-start evaluation complete"
