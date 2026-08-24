#!/usr/bin/env bash
set -euo pipefail

JIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JIT_ROOT="$(cd "${JIT_SCRIPT_DIR}/../.." && pwd)"
JIT_PYTHON="/home/qy/mujoco_playground/.venv/bin/python"
JIT_SUCCESS_RUN="${JIT_ROOT}/JIT/runs/phase_u/phase_u_1024_one_block_20260824_seed820001_retry2"

cd "${JIT_ROOT}"
export PYTHONPATH="${JIT_ROOT}/JIT/src"

"${JIT_PYTHON}" -m compileall -q JIT/src JIT/cli
"${JIT_PYTHON}" - <<'PY'
import ast
from pathlib import Path

source_root = Path("JIT/src/jit_dvgc")
for path in source_root.glob("*.py"):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        if any(name == "dvgc" or name.startswith("dvgc.") for name in names):
            raise SystemExit(f"forbidden legacy import in {path}")
PY
"${JIT_PYTHON}" -m pytest JIT/tests -q -m "not gpu"
"${JIT_PYTHON}" -m pytest JIT/tests/test_env_gpu.py -q -m gpu
"${JIT_PYTHON}" -m jit_dvgc.reference_analysis \
  --input data/reference_jump.csv \
  --output JIT/runs/reference_analysis.json
"${JIT_PYTHON}" -m jit_dvgc.provenance verify-run "${JIT_SUCCESS_RUN}"
