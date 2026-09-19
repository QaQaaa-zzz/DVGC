# Phase U 256-Environment Safety Layout Design

## Trigger and scope

The workstation has suffered two system crashes while 512-environment Phase U
runs were in use. The user explicitly requested reducing the parallel
environment count. No further 512-environment PPO run is permitted by this
design.

This is an operational-safety layout revision, not another scientific reward
hypothesis. It resets the comparable runtime baseline for the pending hip-std
0.10 experiment. Existing 512-environment runs remain immutable provenance and
must not be resumed or presented as directly controlled comparisons with the
new layout.

## Selected layout

Use:

```text
num_parallel_envs = 256
unroll_length = 25
batch_size = 16
num_minibatches = 16
num_updates_per_batch = 1
rollout_block_size = 6,400 total environment transitions
training_seed_count = 256
```

Reducing `num_minibatches` from 32 to 16 together with environments from 512
to 256 preserves 400 samples per minibatch and the number of optimizer passes
per collected transition. Keeping 32 minibatches would halve the minibatch
sample count and silently change optimization noise. Reducing to 128/8 is
reserved as a fallback only if the 256 layout fails a bounded smoke or causes
another host stability event.

The 192,000-transition stable formal config therefore has 30 blocks instead
of 15. Its aligned checkpoints remain exactly 0/102,400/192,000. Candidate and
continuation ceilings per eligible checkpoint each become 6,400 transitions,
matching one layout-native block.

For the fresh maximum-1M authorization, 998,400 remains exactly aligned:

```text
998,400 / 6,400 = 156 blocks
```

The requested checkpoint schedule still aligns to
0/102,400/256,000/499,200/748,800/998,400.

## Preserved contracts

- Authoritative XML remains the historical-name 2 kg payload model.
- Hip/knee limits remain +/-50 N m and action mapping is unchanged.
- Reward remains cap 4, Apex approach 8, liftoff 8, stable-airborne 16.
- The pending scientific variable remains hip initial action standard
  deviation 0.10; other action-channel standard deviations remain 0.05.
- Natural reset, observation/history, network, optimizer hyperparameters,
  horizon, safety terminations, fixed evaluation seeds, Gate Pause rules, and
  snapshot/continuation admission rules remain unchanged.
- No 512-environment formal process may be launched or resumed.

## Qualification

Red-green tests bind the stable formal config to 256/16, block size 6,400,
the resulting interaction ceilings, and hip std 0.10. After full static
validation, the GPU runtime gate must pass outside any environment that hides
CUDA. Then a single run-bound 256-environment smoke qualifies the layout. Only
after a clean smoke may a fresh 256-environment formal authorization be issued.

