# Phase U Hip Exploration Standard Deviation 0.10 Design

## Evidence

The cap-4 run
`phase_u_2kg_apex8_stable16_liftoff8_cap4_env512_998400_20260813_seed720602`
entered a closed Gate Pause at 256,000 training transitions. All 21 checkpoint
sidecars, 24 held-out MP4/NPZ pairs, and 632 fixed-evaluation transitions
validate. Every held-out rollout reached the legal window but had zero
liftoff, zero Apex, and zero physical failure.

Cap 4 did not materially change the cap-8 outcome. At 256k, stochastic episodes
averaged liftoff +5.68, stable-airborne +1.60, Apex approach +1.467, angular-
rate cost -32.53, illegal-contact cost -9.6, 20% physical failure, and zero
success. The learned maximum policy standard deviation remained 0.2387, close
to the initial hip value 0.25, while policy location minimum moved to -0.422.
The deterministic trace held hip control between -1.232 and -1.200 rad and did
not lift off. Cap 1 produced high-rate deterministic pitch failures; cap 4 and
cap 8 both produced conservative no-liftoff means. The reward-cap bracket is
therefore closed rather than tuned further.

The deployment actor already observes three-frame joint/IMU/action history and
a deployable distance-to-obstacle-front signal. The failure is not absence of
window position information. Existing one-tick physical diagnostics found
that hip actions +0.10--+0.15 can produce low-pitch liftoff. With standard
deviation 0.25 these useful actions coexist with broad destructive tails, so
the optimizer can retain a negative mean while occasional positive samples
collect bridge rewards.

## Single hypothesis

Change only the hip element of the stable Phase U initial action standard
deviation:

```text
[steer, drive, hip, knee]
[0.05, 0.05, 0.25, 0.05]
->
[0.05, 0.05, 0.10, 0.05]
```

Across 256 environments, a zero-mean 0.10 prior still samples +0.10 with about
one-sigma frequency and +0.15 with finite coverage. It sharply reduces the
large-action tail and requires useful low-pitch departures to be represented
by movement of the learned mean instead of persistent high variance.

## Preserved boundaries

- Reward contract remains exactly cap 4, Apex approach 8, liftoff 8, and
  stable-airborne 16; no reward/reset/threshold/deadline changes.
- Natural stable reset, network, optimizer, horizon, fixed evaluation, and
  checkpoint Gate Pause are unchanged.
- XML remains the authoritative 2 kg payload model with +/-50 N m hip/knee
  limits and unchanged action mapping.
- Early airborne remains nonterminal and receives no pre-window task progress.
- Roll/pitch/contact/nonfinite termination remains unchanged.
- This is not reference action replay, behavior cloning, or reference reset.

## Validation and execution

Red-green tests bind both stable configs to the ordered standard deviation
vector and keep invalid-vector/hash/manifest checks active. Then run focused
tests, compileall, full pytest, local preflight, and a fresh runtime gate. A
clean 256-environment smoke may authorize one fresh 998,400-transition formal
run with unchanged sparse checkpoint monitoring and evidence-gated snapshot /
continuation acquisition.
