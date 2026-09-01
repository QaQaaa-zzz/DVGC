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

import jit_dvgc.acquisition as acquisition
import jit_dvgc.analysis as analysis
import jit_dvgc.continuation as continuation
import jit_dvgc.snapshots as snapshots
import jit_dvgc.training as training
import jit_dvgc.tube as tube
import jit_dvgc.workflow as workflow

pi1 = training.load_unified_formal_config(
    Path("JIT/configs/pi_unified_iter1_tube1_natural10_retry01.json")
)
assert pi1.ppo.requested_transitions == 10_009_600
assert pi1.ppo.seed == 821101
assert pi1.runtime_naccdmax == 1024
assert pi1.reset_mixture.natural_reset_probability == 0.1
assert pi1.reset_mixture.soft_tube_probability == 0.9
assert pi1.formal.resume_semantics == "fresh_only"
assert pi1.raw["claim_boundary"]["test_data_used"] is False
assert pi1.raw["claim_boundary"]["validation_data_used"] is False

tube1 = tube.load_core_retaining_tube_config(
    Path("JIT/configs/envelope_iter0_tube1_core_retaining.json")
)
assert tube1["protocol"]["iteration"] == 1
assert tube1["protocol"]["source_iteration"] == 0
assert tube1["protocol"]["policy_name"] == "pi_0"

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
print("PI1 FORMAL CONTRACT = PASS")
print("TUBE1 LOCKED CONFIG = PASS")
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
