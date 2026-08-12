# Phase U Angular-Rate Cap-4 Rebalance Design

## Evidence

The isolated Apex-approach-weight-8 run
`phase_u_2kg_apex8_stable16_liftoff8_cap8_env512_998400_20260813_seed720502`
entered a closed Gate Pause at 256,000 training transitions. All 24 held-out
rollouts reached the legal window and ended at the unchanged missed-liftoff
deadline, with zero liftoff, zero Apex, and zero physical failure. All 21
checkpoint sidecars, 24 MP4/NPZ pairs, and 640 fixed-evaluation transitions
validate.

The isolated change behaved mechanically as intended but did not change the
policy outcome. At 256k the stochastic mean `apex_approach` contribution was
1.258 versus 0.283 in the prior weight-2 run, while Apex success remained zero.
The same batch had mean liftoff +4.24, stable-airborne +1.28, angular-rate cost
-34.03, illegal-contact cost -6.6, and 22% physical failure. Its policy location
minimum moved to -0.434 while the deterministic trace kept hip control between
-1.234 and -1.200 rad and never lifted off. The dominant angular-rate cost is
therefore selecting a conservative no-jump mean even though stochastic
liftoff/stable-ascent samples remain present.

The earlier cap-1 run is the opposite failure mode: it learned high-rate hip
motion and produced deterministic pitch-limit failures. Cap 8 removed that
failure but remained overly conservative across the legal-liftoff,
stable-airborne, and Apex-approach bridge iterations.

## Single hypothesis

Change only the stable Phase U configuration value:

```text
angular_rate_penalty_cap_ratio: 8.0 -> 4.0
```

The angular-rate penalty still has weight 1.0 and remains linear in threshold
ratio up to four times the Apex angular-speed limit. Thus rates substantially
above the physical Apex band remain more costly than at the disproven cap-1
setting, while the maximum per-tick cost is halved from the conservative cap-8
setting. This is a bracketed balance test, not a relaxation of termination:
roll/pitch/contact/nonfinite failure rules are unchanged.

## Preserved boundaries

- Apex approach remains weight 8 and is gated by legal-window entry, stable
  full-structure airborne, and positive vertical velocity.
- Liftoff +8 and stable-airborne +16 remain one-shot bridges and do not imply
  success or termination.
- Early airborne remains nonterminal telemetry with zero post-window progress
  before legal-window entry.
- XML/payload, force limits, action mapping, reset, observations, thresholds,
  deadlines, optimizer, network, exploration, horizon, all other reward terms,
  and fixed evaluation remain unchanged.
- Reward remains finite and clipped to [-50, 50]; the cap is positive, finite,
  validated, and reward-hash bound.

## Validation and execution

Red-green tests bind the two stable configs to cap 4 and prove reward-hash
drift. Then run focused tests, compileall, full pytest, local preflight, and a
fresh 96-transition runtime gate. A clean 512-environment smoke may authorize
one fresh 998,400-transition formal run. The same automatic checkpoint Gate
Pause and evidence-gated candidate/continuation protocols remain active.
