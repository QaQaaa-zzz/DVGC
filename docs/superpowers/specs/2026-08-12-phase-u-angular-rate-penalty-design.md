# Phase U Angular-Rate Penalty Single-Hypothesis Design

## Objective

Test one falsifiable explanation for the 2 kg Phase U Gate Pause: the bounded
angular-rate penalty is too small relative to legal-window ascent/clearance
shaping, so PPO can improve sampled return by producing high-angular-rate
launches that never enter the Apex band.

## Evidence

The fresh 2 kg run stopped at 755,200 transitions after three consecutive
held-out windows with zero window reach, liftoff, clearance, or Apex and 8/8
`pitch_limit` failures. Its deterministic policy makes a large hip move in the
first two control ticks while keeping the knee near its initial target. Failure
traces reach roughly -6 to +10 rad/s pitch rate and terminate in 10--15 ticks.

There is no pre-window task-progress leak: all window/ascent/clearance/Apex
components are exactly zero in the held-out pitch failures. A fixed 36-case
hip/knee coordination diagnostic rejected higher knee exploration, and a fixed
18-case one-tick hip diagnostic showed low-pitch liftoff impulses exist but do
not reach stable airborne. The current angular-rate term is capped at -0.25 per
tick, while ascent and clearance can contribute +4 and +2 per tick.

## Alternatives

1. **Increase only angular-rate penalty (selected).** Directly changes the
   failure variable, preserves positive task progress, and keeps the experiment
   one-dimensional.
2. Reduce ascent/clearance shaping. Rejected because it weakens both unsafe and
   potentially valid upward progress.
3. Change hip or knee exploration. Rejected: low hip impulses lack height and
   higher knee exploration did not remove pitch failure.
4. Change action mapping or add a timing latch. Rejected because existing actor
   observations already contain deployable distance history, and those changes
   exceed the current single-hypothesis scope.

## Change

Set only:

```text
phase_u_reward.angular_rate_penalty_weight: 0.25 -> 1.0
```

in both stable Phase U training configurations. The term remains observable,
bounded, future-free, label-free, reference-time-free, and active before and
after the legal window. At the cap it is comparable to, but does not erase,
the +4 ascent signal. The total reward remains clipped to [-50, 50].

No other reward coefficient, action standard deviation, reset, observation,
threshold, deadline, model/XML, force limit, action mapping, PPO hyperparameter,
network, horizon, or seed protocol changes.

## Red-Green Contract

Tests first require both stable Phase U configs to resolve weight 1.0, while
the reward function remains linear and capped: a signal at the angular-speed
threshold contributes exactly -1.0 with the new config, and non-angular terms
are unchanged. The RED state must fail because the configs still contain 0.25.

The minimal GREEN change edits only the two config values. Then run targeted
reward/config tests, static compilation, full pytest, local preflight, and a
fresh runtime gate because the reward source/config fingerprint changes.

## Experimental Gates

After validation, issue a fresh smoke-only authorization for one 512-env
12,800-transition PPO block with fixed evaluation and failure videos. Smoke is
an engineering gate, not a learnability claim. It must have finite updates,
closed transition/outcome accounting, valid checkpoint identity, and no
collision/runtime/provenance fault.

Only after clean smoke may a separate formal authorization be issued. The
formal run starts from a fresh natural reset and fresh policy; it does not
resume any 0.25-reward checkpoint. Checkpoint evaluations remain fixed and the
existing plateau/degradation/reward-hacking gates remain active. Candidate
snapshot acquisition still requires real held-out Apex successes and at least
eight independent successful parents.

## Falsification and Stop Rules

The hypothesis is not validated by lower return alone. It is supported only if
held-out behavior preserves window reach and begins producing liftoff/stable
airborne with lower pitch/rate failure, or eventually Apex success. It is
falsified if fixed evaluations again progress from missed-liftoff to early
pitch-limit failure without Apex, or if training return improves while physical
metrics deteriorate. NaN/Inf, contract/hash failure, severe action saturation,
collision overflow, or three-window plateau triggers Gate Pause.

No `pi_up_star`, formal `V_up`, Soft Tube, Phase D expert, unified PPO, or
JCE/JEL claim follows from this experiment alone.
