#!/usr/bin/env bash
set -euo pipefail

ROOT="${EXPERT_ROOT:-runs/stage_experts/flight_seed0_20260715T2045}"
PYTHON="${PYTHON:-/home/qy/mujoco_playground/.venv/bin/python}"
CFG="${CFG:-configs/default.json}"
BANK="${FLIGHT_BANK:-artifacts/flight_candidates_augmented_v1.pkl}"
ENTRY="${LANDING_ENTRY_SET:-artifacts/landing_entry_tube_v2.pkl}"
REGISTRY="${EXPERT_REGISTRY:-$ROOT/expert_registry_runtime_gate.json}"
POLICY="$ROOT/pi_f_init"
INITIAL_COMPOSITE="$ROOT/initial_composite_evaluation.json"
LANDING_BASELINE="$ROOT/frozen_landing_baseline.json"
mkdir -p "$ROOT/logs"

[[ -x "$PYTHON" && -f "$REGISTRY" && -d "$POLICY" ]] || { echo "Missing expert baseline inputs" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "Tracked worktree is dirty" >&2; exit 2; }
"$PYTHON" -m cli.runtime_gate --config "$CFG" --output docs/RUNTIME_GATE.json --check-only >/dev/null

for curriculum in late_descent descent apex ascent full; do
  run="$ROOT/curriculum/$curriculum"; log="$ROOT/logs/$curriculum.log"
  if [[ -f "$run/training_metrics.json" ]] && "$PYTHON" -c "import json; raise SystemExit(0 if json.load(open('$run/training_metrics.json'))['status']=='gate_pass' else 1)"; then
    echo "[expert-pipeline] skip passed $curriculum"
  else
    [[ ! -e "$run" && ! -e "$log" ]] || { echo "Existing non-passed run blocks $curriculum: $run" >&2; exit 3; }
    echo "[expert-pipeline] start $curriculum"
    set +e
    extra=()
    [[ "$curriculum" == late_descent ]] && extra=(--initial-composite-evaluation "$INITIAL_COMPOSITE" --landing-baseline "$LANDING_BASELINE")
    "$PYTHON" -u -m cli.train_expert --stage flight --curriculum "$curriculum" --bank "$BANK" --entry-set "$ENTRY" --registry "$REGISTRY" --resume "$POLICY" --run "$run" --config "$CFG" --seed 0 "${extra[@]}" >"$log" 2>&1
    status=$?
    set -e
    if (( status != 0 )); then echo "[expert-pipeline] FAIL $curriculum exit=$status" >&2; tail -n 40 "$log" >&2; exit "$status"; fi
  fi
  passed_report=$("$PYTHON" -c "import json,pathlib; xs=[p for p in pathlib.Path('$run/blocks').glob('block_*/report.json') if json.load(open(p))['status']=='PASS']; assert len(xs)==1; print(xs[0])")
  block_dir="$(dirname "$passed_report")"; POLICY="$block_dir/policy"; REGISTRY="$block_dir/expert_registry.json"
  echo "[expert-pipeline] passed $curriculum policy=$POLICY"
done

"$PYTHON" - <<PY
import json
from pathlib import Path
out=Path('$ROOT/frozen_flight_expert.json')
if out.exists(): raise SystemExit('Frozen expert manifest already exists')
out.write_text(json.dumps({'status':'PASS','policy':str(Path('$POLICY').resolve()),'registry':str(Path('$REGISTRY').resolve()),'entry_set':str(Path('$ENTRY').resolve())},indent=2))
print(out)
PY
