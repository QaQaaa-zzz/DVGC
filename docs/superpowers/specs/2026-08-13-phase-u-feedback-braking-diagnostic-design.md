# Phase U Feedback-Braking Launch Diagnostic Design

## Objective

Determine whether the authoritative 2 kg, +/-50 N m model admits a
natural-start launch that combines useful vertical motion with bounded angular
rate. This is a fixed-budget physical diagnostic after the completed v8 PPO
run; it is not another PPO experiment and does not alter the environment,
model, thresholds, reward, reset, action mapping, or observation contract.

The single hypothesis is:

```text
the launch impulse is physically available, but a short deployable
pitch/pitch-rate feedback brake is required to decouple upward propulsion
from the high-rotation ejection mode seen during stochastic training
```

## Evidence and Rejected Alternatives

The v8 formal run completed 998,400 training transitions with zero fixed Apex
success. Coordinated-joint credit was not disconnected: it correlated 0.944
with stochastic liftoff, but also 0.769 with physical failure and tightly with
large angular-rate cost. Increasing its weight would reward both desired and
unsafe launch behavior. Lowering its 0.15-rad/s deadband would pay the final
grounded fixed policy, whose maximum synchronized joint velocity was 0.1383
rad/s. Both changes are rejected before further evidence.

Earlier fixed impulse grids establish that launch authority exists, but their
constant pulses ended in positive-pitch, wheelie, or pitch-limit failures and
did not test feedback braking. The new diagnostic therefore changes no
training contract. It tests a bounded family of physical feedback controllers
from the unchanged natural reset.

## Controller Family

Before the existing monotonic jump-window latch, the action is neutral. Once
the latch is active, the controller uses only current deployable physical
signals:

```text
hip = clip(launch_bias
           - pitch_gain * pitch
           - pitch_rate_gain * pitch_rate,
           -action_limit,
           +action_limit)

knee = clip(knee_ratio * max(hip, 0), 0, action_limit)
```

After a fixed active duration, action returns to neutral. Steering and drive
remain neutral, which preserves the existing nominal forward-speed command.
The controller does not read reference index/time, outcome, success, end code,
future state, teacher identity, or metadata. The kinematic guideline may only
remain threshold provenance and is not replayed.

The exact frozen grid is:

```text
launch_bias:       0.20, 0.30, 0.40, 0.50
knee_ratio:        0.0, 0.5, 1.0, 1.5
pitch_gain:        0.0, 0.5, 1.0
pitch_rate_gain:   0.00, 0.03, 0.06, 0.10
active_ticks:      4, 7
action_limit:      0.80
```

This is 384 branches. Each branch uses one fixed natural-reset seed and a
maximum horizon of 80 real control ticks, for at most 30,720 diagnostic
environment transitions. Branches are evaluated exactly once; no parameter is
moved after seeing outcomes.

## Runtime and Accounting

The diagnostic uses `PhaseExpertEnvAdapter` and the formal pure-JAX two-phase
runtime, not host MuJoCo as its transition authority. It runs one environment
at a time or in a small bounded batch to avoid host-memory pressure. It records
separately:

- diagnostic environment transitions;
- PPO training transitions, always zero;
- fixed source/model/config/threshold/reward hashes;
- exact controller parameters and seed;
- event ticks, end code, terminal reason, and closed outcome category;
- maximum height, vertical velocity, clearance, roll/pitch, angular speed;
- minimum forward velocity and minimum Apex-contract residual;
- full qpos/qvel/ctrl/action traces for representative branches.

Outcome categories are `success`, `physical_failure`, `timeout`, and
`other_failure`; their counts must sum to the number of branches. The report
must distinguish full Apex membership from partial launch progress.

## Ranking and Decision Rule

Results are ranked without rewriting the search grid:

1. valid Apex transition-band success;
2. stable-airborne and ascending progress with no physical failure;
3. smaller minimum normalized Apex-contract residual;
4. lower peak angular speed and attitude excursion;
5. lower action energy.

If any branch reaches valid Apex, it is only a
`physical_launch_diagnostic_candidate`. It does not become an expert snapshot,
reachable state, safe state, Tube, or training reset. The next PPO hypothesis
may use the distinguishing deployable physical quantity, but cannot track this
controller or trajectory pointwise.

If no branch reaches Apex but low-rate branches materially improve stable
airborne/ascent and Apex residual, the next single reward hypothesis may target
that measured distinction. If all useful vertical motion still requires
unsafe rotation, the result is a physical/control-design blocker that must be
documented before another reward iteration.

## Media and Failure Evidence

Render and hash at least one representative branch for each observed terminal
reason, plus the best nonterminal-progress branch and every Apex-success
branch up to eight total videos. Rendering has zero environment-transition
cost and cannot decide pass/fail. Every video has a timing-aligned NPZ trace.

## Test Contract

Red-green tests must prove:

- exact 384-spec deterministic grid and no duplicate specs;
- pre-window action is neutral;
- feedback uses pitch and pitch rate with the declared signs and clipping;
- knee command is nonnegative and follows only positive hip launch effort;
- active duration is monotonic and then returns to neutral;
- closed outcome accounting rejects inconsistent totals;
- ranking prefers Apex, then safe stable progress, then residual/rate/energy;
- manifest declares zero PPO transitions and forbids expert/Tube/safety claims;
- representative media selection is outcome-driven, not renderer-driven.

## Stop Boundary

After one frozen diagnostic run, audit all counts, hashes, traces, and videos.
Do not start PPO from the diagnostic result automatically. First write one
evidence-backed reward or exploration hypothesis, then repeat red-green,
static/runtime qualification, a fresh smoke, and a new run-bound formal
authorization.
