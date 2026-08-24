# JIT Phase U Formal Training Design

## Status and approval

This design implements the user's approved next step after the independent JIT
engineering smoke: one formal 998,400-transition Propulsion-Ascent PPO run.
The user selected the concise persistent-process approach and required every
validated JIT source round to be pushed to GitHub before training starts.

## Scope and claim boundary

This round adds and launches formal Phase U training only. It does not add
Phase D, continuation labels, `V_up`/`V_down`, learned soft Tubes, unified PPO,
or JCE/JEL certification. Completing 998,400 transitions does not by itself
make the result a trained expert. Promotion requires deterministic held-out
evaluation showing real, legal, low-rotation Apex successes across independent
seeds.

The authoritative XML, physics, action mapping, Actor/critic observations,
reward, terminal conditions, network widths, and PPO hyperparameters remain
unchanged from the verified smoke except for the declared training seed,
number of aligned blocks, and evaluation/checkpoint schedule.

## Training layout

The formal config uses:

```text
num_parallel_envs = 1024
episode_horizon = 200
unroll_length = 25
batch_size = 128
num_minibatches = 8
num_updates_per_batch = 1
block_transitions = 25,600
formal_blocks = 39
requested_training_transitions = 998,400
```

The new formal training seed is `820101`. Held-out deterministic evaluation
continues to use the frozen disjoint seeds `920001` through `920008`.
The initial run id is
`phase_u_formal_998400_seed820101_20260824`; a recovery segment, if ever
needed, must use a new run id and identify its parent checkpoint.

## Entrypoint separation

`train_phase_expert.py` retains one stable CLI and accepts exactly one of
`--smoke` or `--formal`. Smoke continues to require one exact block and cannot
load the formal config. Formal mode requires the formal schema and refuses any
budget other than 39 aligned blocks.

The formal implementation lives in `formal_training.py`; `ppo.py` keeps the
shared asymmetric network factory and engineering smoke. This avoids growing
the already-audited smoke runner into one mixed orchestration function.

## Checkpoints and warm resume

Absolute checkpoint transitions are:

```text
0
102,400
256,000
512,000
742,400
998,400
```

Each checkpoint stores the observation normalizer, Actor parameters, critic
parameters, absolute training transition, config/XML hashes, Actor field
order, and action order. The final checkpoint is immediately loaded and used
for inference before the run can close as completed.

The installed Brax public training API does not expose optimizer state, PPO
minibatch RNG, or its complete internal TrainingState. Therefore resume is
truthfully defined as a parameter-level warm resume into a newly predeclared
run segment. The restored normalizer, Actor, and critic initialize the new
segment; Brax reinitializes optimizer and runtime RNG. The manifest records the
parent checkpoint, absolute starting transition, derived segment seed, and
`resume_semantics=parameter_warm_start_optimizer_reset`. No output may call
this bit-exact or optimizer-exact continuation.

The initial formal launch is one uninterrupted persistent process, so warm
resume is only an abnormal-exit recovery route.

## Fixed deterministic evaluation

Fixed evaluation runs at each nonzero checkpoint transition. Every panel uses
all eight held-out seeds, the frozen training environment `done` semantics,
and deterministic Actor actions. Collection stops immediately at terminated
or truncated.

Each panel saves:

- an aggregate Phase U summary;
- one state-trace NPZ per seed containing qpos, qvel, ctrl, action, reward,
  reward components, terminal flags, and end code;
- exact fixed-evaluation interaction counts;
- a representative MP4 only at the final checkpoint, selected as the first
  Apex success if one exists and otherwise the first held-out trace.

Rendering replays saved Host states and consumes zero environment transitions.
Video never determines pass/fail.

## Progress, accounting, and stopping

Brax is configured for one callback after every aligned block. The callback
records absolute training transitions and finite scalar metrics. Only declared
milestones trigger checkpoint and fixed evaluation work.

The run is predeclared before environment interaction and then marked running
with process id, UTC start time, seed, target, and resume semantics. Training,
Brax evaluation, fixed evaluation, and diagnostic transitions remain separate.
Brax evaluation stays disabled because the frozen eight-seed panel is the
authoritative evaluation path.

The process stops and closes `engineering_error` on nonfinite metric, CUDA/OOM,
checkpoint identity or restore failure, trace-save failure, or abnormal
exception. It does not change reward or hyperparameters in response. A high
but finite KL is recorded; at the 102,400 milestone it is inspected together
with terminal causes before any later research change.

## Persistence and monitoring

After source verification, focused Git commit, and GitHub push, the formal
command is launched with `nohup` and a new process session. Its launcher log
and pid record live under ignored `JIT/runs/phase_u/`. The child process creates
the immutable run directory itself before interaction.

Monitoring is sparse:

1. once after startup for GPU, `status=running`, transition-0 checkpoint, and
   absence of NaN/OOM;
2. at declared checkpoints;
3. on completion or abnormal exit.

No full-log polling loop is used.

## Verification before launch

Tests must prove formal config alignment, exact schedules, mode separation,
warm-resume identity and offset rules, running-status transition, trace
serialization, fixed evaluation accounting, final checkpoint restore, and
formal provenance verification. A reduced injected trainer exercises the
complete formal orchestration without consuming a real 998,400-transition
budget. The existing GPU environment suite and one-block smoke evidence must
remain valid.

The implementation is committed and pushed to
`origin/agent/two-phase-soft-tube` before the formal process starts.
