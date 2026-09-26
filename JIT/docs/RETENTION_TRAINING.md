# JIT old-support action retention experiment

User approved implementation and direct bounded training on 2026-09-14.

Purpose: measure old-state retention and pending-state learning separately. Keep
all old policies and historical runs immutable. No final TEST, physics, reward,
endpoint, or observation change. This is development evidence, not a guarantee.

Implementation plan:
- Add an opt-in frozen-source deterministic-action MSE on a locked bank of
  source-policy witnessed TRAIN snapshot observations. Sample with equal phase
  mass and existing within-phase weights. Teacher uses its frozen normalizer;
  student uses the current normalizer. The bank is supervised data, never PPO
  trajectory replay. Do not average different teachers.
- Keep the existing PPO trainer and independent fresh critic/optimizer. Record
  base PPO loss, action MSE, weighted penalty, total loss and approximate KL.
- Compare coefficient 0 versus 1, same seed, source, reset support and 128,000
  transitions per arm; save checkpoints at 32k/64k/128k. Use 128 environments and
  the established aligned PPO recipe. Anchor batch 64; no sweep or auto retry.
- Preserve all checkpoint panel trajectories, rewards and optimizer logs; export
  plotting data and PNG/PDF/SVG. Evaluate the same old and pending states with
  each policy, distinguishing retention/loss/gain/unknown by complete context.
- A CPU supervisor launches only when the GPU has no compute processes; it
  waits at most 24 hours, never signals external jobs, and stops on child error.

Validation: behavioral tests for penalty gradients, frozen teacher, zero weight,
balanced source-only sampling, rejected malformed contracts and idle gate;
existing candidate/formal/queue tests; bounded production training starts after
these checks. Record actual launch status instead of calling queued work trained.
