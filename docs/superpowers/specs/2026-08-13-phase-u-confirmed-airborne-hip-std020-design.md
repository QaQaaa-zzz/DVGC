# Phase U Confirmed-Airborne Hip Exploration 0.20 Design

## Evidence

The completed run
`phase_u_2kg_env256_confirmed_airborne_998400_20260813_seed720902`
consumed 998,400 PPO-training transitions and 1,296 fixed-evaluation
transitions. All six held-out checkpoints reached the legal jump window in
8/8 rollouts, but none achieved confirmed liftoff, stable airborne, ascent,
clearance, or Apex. Every rollout ended at
`takeoff_missed_liftoff_deadline`; physical failures, illegal contact,
roll/pitch violations, saturation, numerical faults, and provenance faults
were zero. The 157 checkpoint sidecars validate, all six outcome accounts
close, and all 48 MP4 plus 48 NPZ files exist with matching hashes.

The training metrics are stronger evidence than the held-out mean alone:
`legal_liftoff_bonus` was exactly zero in every PPO rollout block. The actor
therefore never observed the corrected confirmed-airborne bridge event. The
final deterministic trace drives hip slightly in the compression direction
(`-1.20` to approximately `-1.23` radians), holds knee at `2.5`, and stays on
the ground.

The older one-tick impulse evidence was produced before confirmed-airborne
semantics and over-counted one-wheel support loss as liftoff. A fixed 18-case
re-audit under current code consumed 506 diagnostic transitions and found:

- hip actions `+0.10`, `+0.15`, and `+0.20` do not produce confirmed liftoff;
- early `+0.25`, `+0.35`, and `+0.50` pulses produce confirmed liftoff,
  stable airborne, and ascent, but later end at pitch limit;
- pulses at tick 19 are too late to produce confirmed liftoff before retained
  takeoff safety failures;
- no case reaches Apex, so this diagnostic proves local action authority only.

With hip initial standard deviation `0.10`, the smallest observed confirmed-
liftoff action (`+0.25`) is a 2.5-sigma tail event and also requires useful
timing. This explains why a sparse legal-liftoff bonus can remain unseen even
over 998,400 transitions.

## Considered approaches

1. **Hip standard deviation 0.20 (selected).** This makes the smallest
   confirmed-liftoff action a 1.25-sigma event while remaining narrower than
   the previously destructive 0.25 prior.
2. **Hip standard deviation 0.15.** Safer tails, but `+0.25` remains a
   1.67-sigma event and may repeat insufficient event coverage.
3. **Return to hip standard deviation 0.25.** Best event discovery, but prior
   evidence already showed destructive stochastic tails and high physical
   failure; rejected.

Increasing reward weights is rejected because a reward cannot train an event
that never occurs. Changing deadlines, safety limits, reset, XML, network, or
optimizer would confound the exploration hypothesis.

## Single hypothesis

Change only the stable Phase U exploration prior in action order
`[steer, drive, hip, knee]`:

```text
[0.05, 0.05, 0.10, 0.05]
->
[0.05, 0.05, 0.20, 0.05]
```

The expected first effect is nonzero stochastic confirmed-liftoff coverage.
It is not required that a one-block smoke learn Apex; the smoke validates
compile/update/checkpoint/evaluation/accounting under the new prior. A fresh
formal run is permitted only after smoke integrity passes.

## Preserved contracts

- Authoritative XML remains the 2 kg payload model with unchanged geometry,
  +/-50 N m hip/knee limits, and action mapping.
- Reward weights and meanings, natural reset, threshold/deadline semantics,
  observation/history, PPO layout, optimizer, horizon, and fixed evaluation
  remain unchanged.
- Early airborne remains nonterminal and is neither success nor progress.
- Confirmed liftoff still requires stable dual-wheel airborne evidence after
  legal jump-window entry.
- Roll/pitch/contact/nonfinite failures remain active.
- Existing run-bound configs and artifacts remain immutable provenance; only
  the two stable source configs are updated.
- This is not guideline replay, behavior cloning, reference reset, a trained
  expert claim, or a Tube claim.

## Validation and execution

Use red-green TDD to change the stable-config expectation from hip `0.10` to
`0.20`, then update exactly `configs/phase_expert_smoke.json` and
`configs/phase_expert_phase_u.json`. Run focused tests, compileall, full
pytest, local preflight, and a fresh managed runtime gate because the stable
runtime fingerprint changes. Then run one fresh 256-environment engineering
smoke. If all integrity contracts pass, issue a new run-bound authorization
for one fresh 998,400-transition Phase U run and monitor only sparse fixed
checkpoints or terminal state.

