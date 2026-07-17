#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${DESCENT_TUBE_RUN:-runs/stage_experts/descent_tube_seed0_20260716T2330}"
UNIT="dvgc-descent-tube-controller.service"
cd "$ROOT"

while true; do
  active="$(systemctl --user is-active "$UNIT" 2>/dev/null || true)"
  decision="$(/home/qy/mujoco_playground/.venv/bin/python - "$RUN/controller_state.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("wait")
else:
    state = json.loads(path.read_text(encoding="utf-8"))
    reason = str(state.get("stop_reason", ""))
    if state.get("current_stage") == "gate_pause" and "Round-2 exact pointwise Tube precision" in reason:
        print("resume_support_repair")
    elif state.get("current_stage") in {"pipeline_complete", "authorized_stop"}:
        print("terminal")
    else:
        print("wait")
PY
)"
  if [[ "$decision" == "resume_support_repair" && "$active" != "active" && "$active" != "activating" ]]; then
    systemctl --user start "$UNIT"
    exit 0
  fi
  if [[ "$decision" == "terminal" ]]; then
    exit 0
  fi
  if [[ "$active" == "inactive" || "$active" == "failed" ]]; then
    # Non-target exits are left to the controller's declared Restart policy or
    # to a structured terminal decision; the watcher never broadens authority.
    sleep 10
  else
    sleep 300
  fi
done
