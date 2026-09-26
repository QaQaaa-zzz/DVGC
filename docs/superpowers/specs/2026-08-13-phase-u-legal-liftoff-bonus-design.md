# Phase U Legal-Liftoff Event Bonus Design

## Evidence

The isolated angular-rate cap-8 run
`phase_u_2kg_angrate_cap8_env512_998400_20260812_seed720202` entered a closed
Gate Pause at 256,000 training transitions. Unlike the prior run, all 24 fixed
evaluations were free of physical failure and pitch violation. Peak angular
speed at 256k was 0.45 rad/s. The cap therefore corrected the high-rate failure
mechanism, but all fixed rollouts safely reached the legal window and then
ended at `takeoff_missed_liftoff_deadline` without liftoff.

Training-time stochastic episodes still accumulated 4--8 units of ascent
shaping per batch, while physical failure fell from 100% to 20%, but no Apex
success occurred. The deterministic 256k trace keeps the hip target near
-1.23 rad and reaches only 0.062 m/s vertical velocity. This is a conservative
policy collapse, not a runtime or safety failure.

The retained fixed hip-impulse diagnostic establishes that legal-window
liftoff is physically available at low impulse. One-tick hip actions 0.10 and
0.15 produce `liftoff_seen` after legal window entry with peak pitch about
0.07--0.19 rad and no physical failure. Stronger impulses are already
distinguished by the cap-8 angular-rate cost and unchanged pitch termination.

## Single hypothesis

Add one bounded, one-shot reward component and one weight:

```text
legal_liftoff_bonus = liftoff_bonus_weight
  iff not previously_liftoff_seen
  and current_liftoff_seen
  and legal jump window has already been entered
  and no physical failure
```

The stable Phase U configs set `liftoff_bonus_weight` to 8.0. The default is
0.0 for compatibility. The bonus is a bridge signal from legal-window approach
to the existing ascent shaping. It is not success and does not terminate the
episode. Phase U success remains the complete Apex transition-band contract.

## Boundary contracts

- Early airborne before legal window entry receives zero liftoff bonus, is not
  terminated by that fact, and is not success.
- Window entry alone receives no liftoff bonus.
- The bonus occurs at most once because the event latch is monotonic.
- The bonus uses only current/previous observable event state, not future
  information, reference time/index, or labels.
- The angular-rate cap remains 8.0; roll/pitch/contact/nonfinite failures and
  all task deadlines remain unchanged.
- XML, 2 kg payload, force limits, action mapping, reset, observation,
  thresholds, PPO layout, optimizer, exploration, and all other reward weights
  remain unchanged.
- The reward remains clipped to [-50, 50], and the new weight is finite and
  non-negative and participates in the reward-contract hash.

## Rejected alternatives

- Rewarding `stable_airborne` first is rejected because the existing physical
  diagnostic reaches it mainly with larger impulses that later hit pitch
  limit; low-pitch liftoff is the safer bridge event.
- Lowering the angular-rate cap or weight is rejected because cap 8 is what
  removed the observed pitch failures.
- Increasing hip exploration, changing action mapping, or loosening safety
  limits is rejected by prior diagnostics or project scope.

## Validation and experiment

Red-green tests must prove: no pre-window bonus, exactly one post-window
liftoff bonus, no bonus from window entry alone, no Apex success implication,
stable config value 8.0, invalid weight rejection, hash binding, JIT behavior,
and unchanged progress gating. Then run targeted/full tests, preflight, a fresh
runtime gate, one 512-env 12,800-transition smoke, and only after clean smoke a
fresh formal run-bound authorization. The existing fixed evaluation and Gate
Pause protocol remains unchanged.

