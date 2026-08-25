# JIT Phase U training stack

## Completed v3 absolute-joint 5M experiment

The completed experiment used `phase_u_absolute_5m.json` and started from newly
initialized PPO parameters. It never restored a v1/v2 checkpoint. Hip and knee
share one keyframe-centered absolute-target rule: action zero commands the
XML keyframe angle, while negative/positive actions interpolate to that joint's
lower/upper limit. Consequently hip maps `[-1, 0, 1]` to
`[-1.3, -1.2, 0.5]` radians and knee maps it to `[-1.5, 2.5, 2.5]` radians.
Steering and rear-wheel mappings are unchanged.

The exact target is 4,988,928 transitions: 203 aligned PPO blocks with 384
parallel environments, 64-step unrolls, 16 chunks per minibatch, 24
minibatches, and 8 optimizer passes. The fixed learning rate is `1e-4`;
entropy is `0.01`; gamma/GAE/clip/max-gradient are `0.99/0.95/0.2/0.5`.
The first-block KL includes observation-normalizer warm-up and is not treated
as a pure policy-shift metric; later blocks use an established normalizer.

At each declared milestone, natural-reset held-out evaluation and forced
airborne-RSI diagnostics are stored separately. Only natural-reset panels can
support promotion. Both routes save complete NPZ traces; their final
representatives also save MP4 and aligned reward/state PNG diagnostics. RSI
interactions are recorded only in the diagnostic ledger.

The retained launch command had deliberately no `--restore-checkpoint`:

```bash
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_absolute_5m.json \
  --run-id phase_u_absolute_4988928_seed820201_20260825 \
  --formal \
  > JIT/runs/phase_u/phase_u_absolute_4988928_seed820201_20260825.launch.log 2>&1 \
  < /dev/null &
```

The run completed 4,988,928 training transitions and passed strict provenance,
but it is `NO_PROMOTION`. Every natural-start panel had 0/8 Apex and 8/8
physical failures; the final policy caused illegal wheel contact after two
control steps. Every forced-airborne RSI panel had 8/8 Apex, but those resets
already supplied height and upward velocity and do not count as natural jump
success. Do not resume or promote the final checkpoint. See the complete
analysis in
`docs/experiments/phase_u_absolute_4988928_seed820201_20260825/REPORT.md`.

Everything below documents retained v1/v2 behavior and historical evidence;
those checkpoints are audit artifacts, not inputs to the active v3 run.

`JIT` is an independent implementation of the first Propulsion-Ascent
engineering delivery described in the repository rebuild guide. It does not
import the existing `dvgc` package and does not copy the authoritative XML.

The retained v2 scope introduced the target-free reference reward, a one-shot
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

One fresh v2 formal run completed on 2026-08-25 with 998,400 training
transitions. All five frozen natural-reset panels had zero Apex/height/ascent
events and 100% roll-limit failures, so the result is `NO_PROMOTION` and no
checkpoint is a trained expert. The retained 2026-08-24 smoke/formal artifacts
remain historical v1 evidence only.

The full v2 experiment analysis is in
`docs/experiments/phase_u_reward_rsi_diagnostics_v2_20260825/REPORT.md`. The
ignored final evidence is under
`JIT/runs/phase_u/phase_u_v2_formal_998400_seed820101_20260825/`; do not reuse
that run ID or extend its final checkpoint.

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
