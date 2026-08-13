# Phase U Airborne Progress Gate Design

## Evidence and diagnosis

The 2 kg, 256-environment, hip-std-0.10 formal run stopped at 256,000 total
training transitions with `held_out_physical_performance_plateau`. Fixed
held-out evaluations at 0, 102,400, and 256,000 transitions all reached the
legal jump window in 8/8 rollouts but achieved 0/8 liftoff and 0/8 Apex
success. Despite remaining wheel-supported, each rollout accumulated roughly
6.0--6.7 clearance-progress reward. The current implementation gates ascent
and clearance only on `jump_window_entered`, so a ground-driving policy can
collect nominally airborne shaping without performing legal liftoff.

## Single hypothesis

The reward shortcut prevents PPO from preferring a real jump. Closing only
this shortcut should improve legal-liftoff discovery without changing the
physical task, safety limits, or exploration distribution.

## Contract

Define the pure-JAX reward gate:

```text
airborne_progress_enabled = jump_window_entered AND liftoff_seen
```

The following terms require this gate:

```text
ascent_progress
clearance_progress
apex_approach
```

Forward propulsion, jump-window entry reward, legal-liftoff and stable-airborne
one-shot bonuses, and all penalties retain their current meaning. Early
airborne remains nonterminal, unpenalized, and insufficient for success. If
airborne occurs before window entry, the runtime event contract does not mark
it as legal liftoff; airborne shaping remains disabled until a post-window
legal-liftoff event occurs.

The adapter continues to derive all state from its external pure-JAX
`TwoPhaseEventState`. No event latch is added to `env.step/info`; no
environment, observation, reset, XML, action mapping, threshold, PPO, network,
optimizer, horizon, or reward weight changes are allowed in this iteration.

## Verification and experiment boundary

Red-green tests must prove:

- window entered while wheel-supported gives zero ascent, clearance, and Apex
  approach reward;
- a post-window legal-liftoff latch enables those terms;
- early airborne followed by window entry, without post-window legal liftoff,
  leaves those terms zero;
- pre-window progress remains zero and all one-shot bonuses and penalties keep
  their existing contracts;
- JIT and VMAP behavior remains valid.

After targeted/full/static/preflight validation, run one bounded PPO smoke.
Only if it passes may a fresh run-bound formal authorization be created. Do
not resume the paused hip-std-0.10 run, change another hypothesis, or declare
`pi_up_star`, formal `V_up`, or a Soft Tube.
