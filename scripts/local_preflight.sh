#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/qy/mujoco_playground/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Configured Python is not executable: %s\n' "$PYTHON" >&2
  printf 'Set PYTHON to the existing MuJoCo Playground interpreter.\n' >&2
  exit 2
fi

cd "$ROOT"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

"$PYTHON" - <<'PY'
import sys

import jax
import mujoco
import mujoco_playground

print(f"python={sys.executable}")
print(f"jax_backend={jax.default_backend()}")
print(f"jax_devices={jax.devices()}")
print(f"mujoco={mujoco.__version__}")
print(f"mujoco_playground={mujoco_playground.__file__}")
if jax.default_backend() != "gpu":
    raise SystemExit("JAX did not select the GPU backend")
PY

"$PYTHON" -m cli.prepare_project \
  --xml assets/orange_bike_4kg_horizontal.xml \
  --reference data/reference_jump.csv
"$PYTHON" -m pytest -q

printf 'DVGC local preflight passed.\n'
