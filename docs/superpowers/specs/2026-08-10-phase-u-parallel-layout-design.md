# Phase U Parallel Layout Design

## Decision

The formal Phase U rerun uses 512 parallel MJX environments. The requested
1,024-environment layout is rejected because its one-block benchmark emitted
MJX Warp broadphase-overflow warnings: the authoritative runtime configured
`naconmax=1024`, while representative batches required up to 1,443 candidate
contacts. A run that can truncate collision candidates is not valid physical
training evidence.

The 512-environment benchmark completed one rollout block without broadphase,
NaN, Inf, OOM, traceback, or contract warnings. It is therefore the highest
tested valid parallelism below the failing 1,024 layout. The 64-environment
layout remains valid but is slower.

## Stable Training Layout

Only the batching layout and transition alignment change:

- `num_parallel_envs = 512`
- `unroll_length = 25`
- `batch_size = 16`
- `num_minibatches = 32`
- PPO rollout block = `25 * 16 * 32 = 12,800` transitions
- maximum aligned training budget = 998,400 transitions
- effective checkpoint schedule = 0, 102,400, 256,000, 512,000, 755,200,
  and 998,400 transitions

The final budget is the largest complete 12,800-transition block below the
authorized 1,000,000-transition ceiling. The reward, reset, optimizer,
network, episode horizon, XML, collision capacity, actuator limits, action
mapping, and fixed evaluation seeds remain unchanged.

## Safety and Stop Contract

The formal run must pass configuration preflight and repository verification
before launch. Its run-bound authorization records source/config/XML hashes,
purpose, interaction ceilings, stopping conditions, and output directory.

The existing Gate Pause rules remain active, including numerical or contract
failure, source/config/XML mismatch, severe action saturation or reward
hacking, repeated held-out degradation, and three-window held-out physical
plateau. Any broadphase-overflow warning at startup invalidates the run and
causes an immediate pause.

## Execution and Observation

The long run is launched as a detached, resumable process. The launcher gets
one bounded startup health check covering PID liveness, `status.json`, the
transition-0 checkpoint, and fatal-warning patterns in the persistent log.
After that check, this interaction stops; the persistent goal resumes Codex on
completion or Gate Pause. There is no periodic chat-side log polling.

Formal outputs remain under `runs/two_phase/` and are not committed. The
repository records only stable configuration, tests, method state, and the
commands needed to resume or audit the run.

