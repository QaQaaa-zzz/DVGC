#!/usr/bin/env bash
set -euo pipefail

JIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${JIT_SCRIPT_DIR}/../.." && pwd)"
PY="${JIT_PYTHON:-/home/qy/mujoco_playground/.venv/bin/python}"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/JIT/src"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"

"${PY}" -m compileall -q JIT/src JIT/cli

"${PY}" - <<'PY'
from pathlib import Path
import json

import jit_dvgc.acquisition as acquisition
import jit_dvgc.analysis as analysis
import jit_dvgc.continuation as continuation
import jit_dvgc.snapshots as snapshots
import jit_dvgc.training as training
import jit_dvgc.tube as tube
import jit_dvgc.workflow as workflow

formal_config_count = 0
replay_contract_count = 0
for path in sorted(Path("JIT/configs").glob("*.json")):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != training.FORMAL_SCHEMA:
        continue
    training.load_unified_formal_config(path)
    formal_config_count += 1
    if "tube_sampling" in raw:
        tube.normalize_core_replay_contract(raw["tube_sampling"])
        replay_contract_count += 1

if formal_config_count < 1:
    raise AssertionError("preflight requires at least one unified formal config")
if replay_contract_count < 1:
    raise AssertionError("preflight requires at least one replay contract")

for required in (
    acquisition.collect_unified_boundary_candidates,
    analysis.run_unified_fixed_panel,
    continuation.label_unified_continuations,
    snapshots.load_unified_envelope_snapshot,
    training.run_unified_formal,
    tube.build_core_retaining_tube,
    workflow.run_workflow,
):
    assert callable(required)

print("JIT PACKAGE/API PREFLIGHT = PASS")
print(f"UNIFIED FORMAL CONFIGS = {formal_config_count}")
print(f"REPLAY CONTRACTS = {replay_contract_count}")
PY

"${PY}" -m pytest JIT/tests -q -m "not gpu"

if [[ "${JIT_RUN_GPU_TESTS:-0}" == "1" ]]; then
  "${PY}" -m pytest \
    JIT/tests/test_env_gpu.py \
    JIT/tests/test_tube_rsi_mixed_snapshot.py \
    JIT/tests/test_unified_reset_mixture_gpu.py \
    JIT/tests/test_unified_continuation_labels_gpu.py \
    -q -m gpu
fi
