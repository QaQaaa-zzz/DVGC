# Phase U Stable-Airborne Bridge Bonus Design

## Evidence

The legal-liftoff +8 run
`phase_u_2kg_liftoff8_cap8_env512_998400_20260813_seed720302` entered a closed
Gate Pause at 256,000 training transitions. All 24 deterministic evaluations
reached the legal window but had zero liftoff, zero Apex, and zero physical
failure. The 256k trace remained conservative with 0.054 m/s peak vertical
speed and 0.51 rad/s peak angular speed.

The training distribution did not lack liftoff samples. Its mean
`legal_liftoff_bonus` was 4.48--6.72 after the first block, corresponding to
roughly 56--84% of stochastic episodes receiving the one-shot +8. At 256k the
mean was +6.0, but those episodes also averaged -41.16 angular-rate cost, -12
illegal-contact cost, 26% physical failure, and zero Apex success. Increasing
the liftoff bonus would therefore reward many low-quality or unsafe departures.

The next observable milestone is the existing monotonic `stable_airborne`
event: it requires a prior legal-window liftoff and the runtime adapter's
full-structure airborne signal on a later tick, without physical failure. It
is stricter than momentary wheel unloading and is still strictly weaker than
ascending or Apex success.

## Single hypothesis

Add one bounded, one-shot reward component:

```text
stable_airborne_bonus = stable_airborne_bonus_weight
  iff not previously_stable_airborne
  and current_stable_airborne
  and legal jump window has already been entered
  and no physical failure
```

The stable Phase U configs select `stable_airborne_bonus_weight = 16.0`; the
default is 0.0 for compatibility. The existing liftoff bonus remains 8.0 and
angular-rate cap remains 8.0. A +16 stable-airborne bridge is larger than the
liftoff bridge but below the +30 Apex success bonus. It is a one-time event,
not a per-tick survival reward.

## Preserved boundaries

- Early airborne before window entry receives no liftoff or stable-airborne
  bonus, is not terminal for that fact, and is not success.
- Liftoff without stable full-structure airborne receives no stable bonus.
- Stable-airborne transition receives the bonus once; the monotonic latch
  prevents repeated collection.
- Stable airborne does not imply ascending, Apex membership, success, or done.
- All inputs are current/previous online event state; no future, reference,
  metadata, or label input is used.
- XML, 2 kg payload, force/action mapping, reset, observation, thresholds,
  deadlines, PPO layout, optimizer, exploration, cap 8, and all other reward
  terms remain fixed.
- Roll/pitch/contact/nonfinite termination remains unchanged. Total reward
  remains clipped to [-50, 50]. The weight is finite/non-negative and hashed.

## Validation and execution

Red-green tests prove pre-window zero, liftoff-only zero, exactly one legal
stable-airborne +16 event, repeat zero, no success implication, stable config
value, invalid value rejection, hash binding, JIT, and metric-key stability.
Then run full tests, preflight, a fresh runtime gate, one 512-env smoke, and—if
clean—a fresh 998,400-transition formal authorization with the unchanged fixed
evaluation/Gate Pause protocol.

