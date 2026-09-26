# Phase U Window-Active Ascent Credit Design

## Evidence

The completed run
`phase_u_2kg_env256_confirmed_airborne_hipstd020_998400_20260813_seed721002`
used 998,400 PPO-training transitions and 1,288 fixed-evaluation transitions.
All six held-out checkpoints reached the legal jump window in 8/8 rollouts,
but none achieved confirmed liftoff, stable airborne, ascent, clearance, or
Apex. Every held-out rollout ended at `takeoff_missed_liftoff_deadline` with
zero physical failure, roll/pitch violation, illegal contact, action
saturation, timeout, or numerical/provenance fault. All 157 checkpoint
sidecars validate recursively; all six outcome accounts close; all 48 MP4 and
48 NPZ artifacts exist and match their declared SHA-256 hashes.

The increased hip exploration did expose the event during stochastic PPO
rollouts. Fifty-one of 156 rollout blocks recorded a nonzero confirmed
`legal_liftoff_bonus`, with the last event at 601,600 transitions. None
reached Apex. The useful event disappeared as the policy converged: every
held-out policy instead compressed the hip from the natural `-1.20 rad`
position toward approximately `-1.25 rad`, held the knee near `2.50 rad`, and
drove through the window on the ground.

The reward data flow explains this local optimum. `ascent_progress` is
currently gated by both the monotonic legal-window latch and confirmed
liftoff. Before confirmed liftoff, the only positive bridge credit is the
sparse one-shot `legal_liftoff_bonus`. The fixed physical impulse audit shows
that a useful hip pulse must begin around control tick 15--17; waiting until
tick 19 is too late. Held-out policies enter the window around tick 18--19 and
reach the spatial liftoff deadline around tick 26--27. Therefore a policy that
has not already learned an upward impulse receives no continuous vertical
credit while learning the action that causes the confirmed event.

## Considered approaches

1. **Enable positive vertical-velocity ascent credit after legal window entry
   (selected).** This supplies a dense, observable bridge signal exactly where
   the approved task contract permits jump/ascent reward. It changes one
   reward gate, not a weight, physical threshold, or deadline.
2. **Increase the one-shot liftoff bonus.** Rejected because stochastic
   liftoff already occurred and the sparse temporal-credit problem remains.
3. **Move the liftoff deadline.** Rejected because this changes the physical
   task window; early impulse evidence already proves sufficient authority and
   timing under the retained deadline.

## Selected contract

`ascent_progress` is activated by the monotonic legal jump-window latch:

```text
jump_window_entered AND com_vz > 0
```

It remains bounded by `ascent_progress_weight` and
`target_vertical_velocity`. Before legal window entry it is exactly zero,
including when the robot is early airborne, has positive vertical velocity,
or has a previously latched diagnostic jump signal.

The following remain gated by confirmed physical events:

- `legal_liftoff_bonus` requires the first post-window confirmed-liftoff
  transition;
- `stable_airborne_bonus` requires the first post-liftoff stable-airborne
  transition;
- `clearance_progress` requires confirmed liftoff;
- `apex_approach` requires stable airborne and ascending eligibility;
- Phase U success requires the complete Apex transition-band contract.

The change does not make early airborne legal liftoff, success, or terminal.
It does not change the authoritative 2 kg XML, geometry, +/-50 N m limits,
action mapping, observation/history, natural reset, jump-window geometry,
liftoff deadline, Apex thresholds, safety failures, PPO layout, optimizer,
episode horizon, fixed seeds, or candidate/continuation gates.

## Validation and execution

Red-green tests must prove:

- before the legal window, positive `com_vz` produces zero ascent progress;
- after legal window entry but before confirmed liftoff, positive `com_vz`
  produces bounded ascent progress;
- non-positive `com_vz` produces zero ascent progress;
- clearance and Apex terms remain zero before their existing physical gates;
- early airborne alone remains neither success nor done.

Update the reward semantics/hash and both stable Phase U configurations only
as required by the contract. Run focused tests, Phase U regressions,
compileall, full pytest, local preflight, and a fresh managed runtime gate.
Then run one fresh 256-environment engineering smoke. Only smoke integrity may
authorize a fresh run-bound formal experiment; no checkpoint from the
completed reward contract may be resumed. A future formal run remains capped
at 998,400 PPO-training transitions and must be audited at the fixed
checkpoints before any snapshot/continuation claim.

