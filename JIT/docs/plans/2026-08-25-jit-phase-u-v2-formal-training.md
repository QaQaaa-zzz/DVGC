# JIT Phase U v2 Formal Training Declaration

## Purpose

Run one fresh Propulsion-Ascent v2 learnability experiment after the complete
reward/RSI/diagnostics source is committed and the exact commit is present on
GitHub. This is not a promotion, Tube, JCE, or JEL run.

## Frozen inputs

- config: `JIT/configs/phase_u_formal.json`
- resolved config SHA-256:
  `df565a03c0c8f40531a5ac57bd6c2c2674d9249ca52c31df143484f4ad484112`
- XML SHA-256:
  `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192`
- reference CSV SHA-256:
  `612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f`
- training seed: `820101`
- held-out seeds: `920001..920008`, forced natural reset only
- network contract: Actor 76, critic 106, action order
  `[steer, rear-wheel drive, hip, knee]`

The run must start fresh. No v1 or v2 checkpoint is an authorized parent.

## Interaction budget and stopping

- exact training budget: 998,400 transitions, 39 aligned blocks of 25,600;
- Brax evaluation transitions: zero;
- five frozen panels at 102,400, 256,000, 512,000, 742,400, and 998,400;
- each panel contains eight natural-start rollouts, each capped at 200 ticks;
- stop normally at 998,400 training transitions;
- close as engineering error on nonfinite metrics, source/config/checkpoint
  identity mismatch, callback-order violation, renderer/provenance failure, or
  an unhandled runtime exception.

Training resets use the frozen 95% natural / 5% airborne-RSI mixture. RSI and
natural results must remain distinguishable in training metrics. Only natural
held-out panels can inform expert selection.

## Output and persistence

- run ID: `phase_u_v2_formal_998400_seed820101_20260825`
- output directory:
  `JIT/runs/phase_u/phase_u_v2_formal_998400_seed820101_20260825/`
- launch log:
  `JIT/runs/phase_u/phase_u_v2_formal_998400_seed820101_20260825.launch.log`
- PID file:
  `JIT/runs/phase_u/phase_u_v2_formal_998400_seed820101_20260825.pid`

All outputs are ignored runtime evidence and remain under `JIT/`.

## Launch gate

Before launch, all of the following must be true:

1. final JIT preflight passes;
2. complete repository compatibility passes except the exact pre-existing
   user dirty-path test;
3. independent review has no Critical or Important finding;
4. one JIT-only commit is created and pushed without force;
5. the remote branch ref equals the local source commit;
6. the output directory does not already exist.

## Persistent command

```bash
nohup setsid env XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  PYTHONUNBUFFERED=1 PYTHONPATH=JIT/src \
  /home/qy/mujoco_playground/.venv/bin/python JIT/cli/train_phase_expert.py \
  --phase propulsion_ascent \
  --config JIT/configs/phase_u_formal.json \
  --run-id phase_u_v2_formal_998400_seed820101_20260825 \
  --formal \
  > JIT/runs/phase_u/phase_u_v2_formal_998400_seed820101_20260825.launch.log \
  2>&1 < /dev/null &
```

Inspect startup once to confirm GPU backend, running status, and transition-0
checkpoint. After that, inspect only declared milestones, completion, or an
abnormal exit; do not poll full logs repeatedly.
