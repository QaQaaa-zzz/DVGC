# JIT Phase U v4 10M Training Design

## Scope

This round updates the active Phase U v4 method in place. It does not preserve
verification compatibility for the previous v4 runs, because those results are
not retained as method evidence. Older v1-v3 schemas and their artifacts remain
unchanged.

All source, tests, configuration, documentation, launch records, automated
analysis output, and run output belong under `JIT/`. The authoritative XML,
geometry, control timing, observation layout, action order, absolute hip/knee
mapping, termination limits, and PPO network architecture remain unchanged.

## Approved method changes

The active v4 contract uses these exact values:

- jump window: `x_min = 2.5 m`, `x_max = 3.4 m`;
- airborne RSI probability: `0.08`;
- height reward coefficient: `40.0`;
- height reward remains multiplied by the one-shot jump signal, so it is zero
  before entry and after the signal is consumed;
- Warp aggregate contact capacity: `naconmax = 4096`;
- Warp aggregate CCD capacity: `naccdmax = 256`;
- Warp constraint capacity: `njmax = 256`.

The jump signal still turns on only after the natural trajectory enters the
window. It turns off after leaving the extended window and cannot turn on a
second time within the same episode. An airborne-RSI reset starts with the
signal enabled and follows the same one-shot consumption rule.

## CCD capacity handling

`naccdmax` becomes an explicit model/config/provenance field and is passed to
every Warp `make_data` construction path. Tests inspect the resulting Warp data
capacity rather than merely checking the JSON value.

This round does not turn every collision-capacity warning into an immediate
training abort. The bounded GPU smoke checks whether increasing the capacity
from the implicit default of 48 to 256 removes the prior large warning stream.
The final automatic analysis reports any remaining CCD overflow count so it is
visible without overstating occasional warnings as a formal safety failure.

## Fresh 10M PPO run

The run starts from newly initialized PPO parameters and optimizer state. It
must not accept a parent checkpoint or restore option. With the retained v4 PPO
block size of 24,576 transitions, the effective target is 9,977,856 training
transitions, the closest complete-block total not exceeding 10,000,000.

The formal checkpoints and fixed held-out panels are:

- 0;
- 491,520;
- 1,990,656;
- 4,988,928;
- 7,987,200;
- 9,977,856 transitions.

Every nonzero checkpoint receives both the natural-start held-out panel and the
separate airborne-RSI diagnostic panel. The final checkpoint retains aligned
MP4, diagnostic PNG/NPZ, and pre-/post-Apex traces for both panels. Training
curves retain mean episode reward, mean episode length, RSI fraction, KL,
policy/value/total loss, policy standard deviation, and throughput.

## Event-triggered Codex analysis

A generalized CLI under `JIT/cli/` waits locally for one declared run.
Polling reads the PID and `status.json` only and does not call a model. Once the
training process reaches a terminal state, the watcher invokes `codex exec`
exactly once in read-only mode with a narrow prompt bound to the exact run
directory.

The spawned Codex task performs read-only analysis:

- inspect terminal status and interaction accounting;
- run the strict provenance verifier when the run completed;
- summarize training curves and PPO anomalies;
- compare natural and airborne-RSI panels at every checkpoint;
- inspect final termination causes, Apex/post-Apex behavior, and diagnostic
  artifacts;
- count remaining CCD overflow warnings;
- write its final response to `AUTO_ANALYSIS.md` inside the run directory.

The watcher uses run-local started/completed markers so a restart cannot trigger
a second paid analysis. Failure to start Codex is recorded under the run
directory and does not alter the already closed training status. The watcher
does not edit source, create commits, resume training, or load a checkpoint.

## Verification and launch gate

Implementation follows red-green TDD for configuration drift, Warp capacity,
window/signal semantics, reward gating, RSI probability, transition alignment,
fresh-start semantics, and watcher single-shot behavior.

Before launch:

1. static compilation passes;
2. all JIT non-GPU tests pass;
3. all JIT GPU tests pass;
4. `JIT/scripts/local_preflight.sh` passes;
5. the source/config/docs/tests/scripts are staged explicitly and committed in
   one validated JIT commit;
6. the committed revision is pushed and remote equality is confirmed;
7. only then, a bounded GPU smoke completes with finite PPO metrics and without
   the prior large CCD overflow stream.

No PPO interaction, including the bounded smoke, may occur before the validated
commit and push. The fresh formal training and its watcher are launched only
after that gate and the closed smoke.
The current interactive Codex session reports the run ID, training PID, watcher
PID, GPU, transition-zero checkpoint identity, and an ETA, then exits without
supervising the run.
