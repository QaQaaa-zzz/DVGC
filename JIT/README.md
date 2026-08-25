# JIT Phase U training stack

`JIT` is an independent implementation of the first Propulsion-Ascent
engineering delivery described in the repository rebuild guide. It does not
import the existing `dvgc` package and does not copy the authoritative XML.

The active v2 scope now includes the target-free reference reward, a one-shot
root-x jump signal, 5% bounded airborne RSI for training resets, natural-only
held-out evaluation, height/descent Apex termination, and synchronized
numeric/PNG/video diagnostics. It also retains environment/runtime integrity,
the aligned 25,600-transition PPO engineering-smoke entrypoint, and an
auditable formal-only Phase U runner. It does not implement Phase D, continuation labels,
`V_up`/`V_down`, learned soft Tubes, unified PPO, or JCE/JEL certification.

The v2 Actor input is `3 x 25 + 1 = 76`: three real sensor-history frames plus
one current `jump_signal`. The critic receives that complete 76-value input
plus 30 privileged values, for 106 total. The signal is therefore available to
both networks and is not repeated in the history. v1 checkpoints (`81/114`)
are deliberately incompatible and must not be resumed into v2.

No PPO training has been run with the v2 contract yet. The retained 2026-08-24
smoke/formal artifacts are historical v1 evidence only; they remain
provenance-verifiable but are not candidate v2 models.

Use the retained interpreter directly:

```bash
PYTHONPATH=JIT/src /home/qy/mujoco_playground/.venv/bin/python -m pytest JIT/tests -q
```

Generated run evidence belongs under `JIT/runs/` and is ignored by Git.
Every new representative v2 video also produces a full-trajectory diagnostic
PNG, an aligned compressed NPZ, and SHA-256 fields in its JSON report.

Run the complete local verification without launching training:

```bash
bash JIT/scripts/local_preflight.sh
```

The explicitly bounded smoke command remains available:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_smoke.json \
  --run-id <unique-run-id> \
  --smoke
```

## Formal Phase U training

Formal mode is exactly 39 aligned blocks, or 998,400 training transitions,
with seed `820101`. Identity-bound checkpoints are written at transitions 0,
102,400, 256,000, 512,000, 742,400, and 998,400. The five nonzero milestones
each run deterministic evaluation on held-out seeds 920001 through 920008.
Brax evaluation is disabled and fixed evaluation is accounted separately.

Source verification, a focused JIT-only commit, and its GitHub push are hard
predecessors of this persistent launch:

```bash
mkdir -p JIT/runs/phase_u
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_formal.json \
  --run-id phase_u_formal_998400_seed820101_20260824_retry1 \
  --formal \
  > JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824_retry1.launch.log 2>&1 \
  < /dev/null &
JIT_FORMAL_PID=$!
printf '%s\n' "${JIT_FORMAL_PID}" \
  > JIT/runs/phase_u/phase_u_formal_998400_seed820101_20260824_retry1.pid
```

Inspect startup once, then only the declared milestones, completion, or an
abnormal exit. A high but finite KL is evidence to inspect, not permission to
change rewards or PPO hyperparameters during the run.

If an abnormal exit requires recovery, `--restore-checkpoint PATH` starts a
new run segment from the saved observation normalizer, Actor, and critic.
Brax resets optimizer state and PPO RNG, so this is explicitly a parameter
warm start and never a bit-exact continuation.

Finishing the transition budget does not by itself establish a trained expert.
Promotion requires multiple legal, low-rotation Apex successes across the
frozen held-out seeds. It never establishes a safe Tube, JCE, or JEL.
