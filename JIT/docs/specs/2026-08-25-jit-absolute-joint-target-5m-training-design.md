# JIT Absolute Joint Targets and Fresh 5M Phase U Training Design

## Decision

The user approved one joint-control rule for both articulated joints:

```text
action = 0  -> XML keyframe position
action < 0  -> piecewise absolute target toward the XML lower limit
action > 0  -> piecewise absolute target toward the XML upper limit
```

Hip remains `-1 -> -1.3`, `0 -> -1.2`, `+1 -> 0.5` radians. Knee changes
from an incremental target to `-1 -> -1.5`, `0 -> 2.5`, `+1 -> 2.5`
radians. The positive knee branch is constant because the authoritative XML
keyframe knee position already equals its upper limit.

Steering and rear-wheel mappings remain unchanged. The action order remains
`[steer, rear-wheel drive, hip, knee]` and actions remain clipped to
`[-1, 1]^4`.

## Compatibility boundary

The absolute-knee contract receives a new configuration schema and an explicit
action-semantics token. Existing v1/v2 configuration parsing and historical
run verification retain their old incremental-knee meaning. The new training
configuration has a different canonical configuration hash, so an old
checkpoint cannot satisfy its identity check.

The requested training starts at transition zero with random network and
optimizer initialization. Its manifest must record `parent_checkpoint=null`,
`starting_training_transition=0`, and `resume_semantics=fresh`. Recovery after
an abnormal exit may use only a checkpoint produced by the same new config;
that remains a parameter warm start with optimizer reset and is not part of the
initial launch.

## PPO translation

The new run translates the user's successful low-parallelism SB3 sampling and
update density into the installed Brax runner without copying SB3's per-env
`n_steps` literally:

```text
num_parallel_envs       = 384
episode_horizon         = 200
unroll_length           = 64
batch_size              = 16
num_minibatches         = 24
num_updates_per_batch   = 8
block_transitions       = 24,576
learning_rate           = 0.0001
entropy_cost            = 0.01
reward_scaling          = 0.1
discounting             = 0.99
gae_lambda              = 0.95
clipping_epsilon        = 0.2
max_grad_norm            = 0.5
```

Each minibatch contains `16 * 64 = 1,024` transitions, each rollout block
contains `384 * 64 = 24,576` transitions, and every block performs
`24 * 8 = 192` optimizer steps. The learning rate stays at `1e-4` instead of
immediately increasing to `3e-4`. The first-block KL includes the fixed-rate
path's observation-normalizer warm-up and is not interpreted as pure policy
movement; subsequent blocks use an established normalizer. An adaptive-KL
smoke was rejected because delaying that warm-up caused policy-output explosion.

The aligned training target is `4,988,928` transitions, the closest lower
whole-block value to five million (`203 * 24,576`). The training seed is
`820201`; fixed evaluation policy keys are `930001` through `930008` and are
disjoint from the training seed.

## Checkpoints and evaluation

Identity-bound checkpoints and fixed panels run at:

```text
0
245,760
983,040
2,506,752
3,981,312
4,988,928
```

Every nonzero milestone gets eight deterministic natural-reset rollouts.
Natural panels alone determine expert promotion. The final natural panel emits
the synchronized MP4, PNG, NPZ, and JSON report already required by JIT.

No old checkpoint is opened for initialization or diagnostics. At every
nonzero milestone, the checkpoint produced by this new run receives a separate
forced-airborne-RSI diagnostic panel in addition to the natural panel. RSI
results are never mixed into the natural promotion rate and their interactions
are accounted as diagnostics rather than fixed evaluation.

## Runtime and stopping rules

The source, config, tests, design, and plan must pass JIT-local static, host,
GPU, provenance, and preflight checks. Only JIT paths are staged. The complete
validated JIT change is committed and pushed before environment interaction.

The new run is predeclared and launched persistently. It stops on the aligned
target, nonfinite metrics, CUDA/OOM failure, callback/order failure,
checkpoint-identity failure, or trace-persistence failure. Monitoring is
sparse: startup, declared milestones, completion, or abnormal exit.

Completing five million transitions does not itself create a trained expert.
The final report must separately state natural-start Apex success, physical
failure, height/ascent evidence, policy KL/std trends, terminal causes, and the
resulting promotion decision.
