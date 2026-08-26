#!/usr/bin/env bash
set -euo pipefail

JIT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JIT_ROOT="$(cd "${JIT_SCRIPT_DIR}/../.." && pwd)"
JIT_PYTHON="/home/qy/mujoco_playground/.venv/bin/python"

cd "${JIT_ROOT}"
export PYTHONPATH="${JIT_ROOT}/JIT/src"

"${JIT_PYTHON}" -m compileall -q JIT/src JIT/cli
"${JIT_PYTHON}" - <<'PY'
from pathlib import Path

from jit_dvgc.config import load_config

legacy = load_config(Path("JIT/configs/phase_u_formal.json"))
active = load_config(Path("JIT/configs/phase_u_absolute_5m.json"))
smoke = load_config(Path("JIT/configs/phase_u_absolute_smoke.json"))
continuation = load_config(Path("JIT/configs/phase_u_continuation_10m.json"))
continuation_smoke = load_config(Path("JIT/configs/phase_u_continuation_smoke.json"))
if legacy.formal is None or legacy.formal.formal_blocks != 39:
    raise SystemExit("retained v2 formal configuration contract is invalid")
if active.formal is None or active.formal.formal_blocks != 203:
    raise SystemExit("active v3 formal configuration contract is invalid")
if active.ppo.requested_transitions != 4_988_928:
    raise SystemExit("active v3 target is invalid")
if smoke.ppo.requested_transitions != smoke.ppo.block_transitions:
    raise SystemExit("active v3 smoke is not one exact PPO block")
if continuation.formal is None or continuation.formal.formal_blocks != 814:
    raise SystemExit("active v4 formal configuration contract is invalid")
if continuation.formal.resume_semantics != "fresh_only":
    raise SystemExit("active v4 formal configuration must be fresh-only")
if continuation.ppo.requested_transitions != 20_004_864:
    raise SystemExit("active v4 target is invalid")
if continuation.ppo.num_evals != 815:
    raise SystemExit("active v4 PPO evaluation schedule is invalid")
if continuation.formal.checkpoint_transitions != (
    0,
    983_040,
    3_981_312,
    9_977_856,
    15_998_976,
    20_004_864,
):
    raise SystemExit("active v4 checkpoint schedule is invalid")
if continuation.formal.fixed_evaluation_transitions != (
    983_040,
    3_981_312,
    9_977_856,
    15_998_976,
    20_004_864,
):
    raise SystemExit("active v4 evaluation schedule is invalid")
if continuation.ppo.seed != 820501:
    raise SystemExit("active v4 training seed is invalid")
if continuation.ppo.held_out_seeds != tuple(range(960001, 960009)):
    raise SystemExit("active v4 held-out namespace is invalid")
if continuation_smoke.ppo.requested_transitions != continuation_smoke.ppo.block_transitions:
    raise SystemExit("active v4 smoke is not one exact PPO block")
PY
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
