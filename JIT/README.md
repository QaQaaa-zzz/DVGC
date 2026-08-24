# JIT Phase U engineering stack

`JIT` is an independent implementation of the first Propulsion-Ascent
engineering delivery described in the repository rebuild guide. It does not
import the existing `dvgc` package and does not copy the authoritative XML.

The current scope is limited to environment/runtime integrity, observable
Phase U semantics and reward, deterministic evaluation/video contracts, and
one aligned 25,600-transition PPO smoke. It does not claim a trained expert,
learnability, a feasibility Tube, safety, or end-to-end two-phase capability.

Use the retained interpreter directly:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests -q
```

Generated run evidence belongs under `JIT/runs/` and is ignored by Git.

Run the complete local verification without launching training:

```bash
bash JIT/scripts/local_preflight.sh
```

The only implemented training entry is the explicitly bounded smoke command:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_smoke.json \
  --run-id <unique-run-id> \
  --smoke
```
