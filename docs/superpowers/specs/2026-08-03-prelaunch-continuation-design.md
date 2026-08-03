# Prelaunch Continuation and Window-Gated Reward Design

## Purpose

An early liftoff is not by itself a physical failure. The simulator must keep
advancing when the robot becomes airborne before the legal takeoff window.
Takeoff and airborne-progress reward remains gated by the legal window, while
the existing physical posture and integrity limits remain authoritative.

This change reopens the fixed Gate B guideline audit. It does not authorize
expert training, feasibility training, continuation labeling, Soft Tube
construction, unified PPO, a learnability pilot, or formal training.

## Termination Contract

`prelaunch_airborne_count` remains a diagnostic signal for telemetry,
snapshots, and failure-video overlays. It must not:

- contribute to `hard_failure`;
- terminate or truncate an episode;
- produce `END_PRETAKEOFF_AIRBORNE`;
- prevent later jump-window entry or two-phase event extraction.

The existing hard posture thresholds remain unchanged:

```text
abs(roll) > max_roll_deg  = 35 degrees -> END_ROLL_LIMIT
abs(pitch) > max_pitch_deg = 75 degrees -> END_PITCH_LIMIT
```

The following existing termination or truncation conditions also remain
unchanged: prohibited structural contact, confirmed invalid wheel contact,
excessive backward motion, platform back-edge exit, takeoff task failures,
nonfinite dynamics, stage timeout, successful recovery, and explicitly
configured chain/stage-entry terminals.

The legacy `END_PRETAKEOFF_AIRBORNE` constant and metric may remain for schema
compatibility, but the new runtime must not emit that end code.

## Jump Window and Reward Contract

The legal takeoff window continues to determine when the jump signal and
window-gated takeoff/ascent reward path can begin. Before window entry, early
airborne motion receives no takeoff/window progress reward.

Window entry must not require current wheel support. The deployable event is:

```text
phase == approach
and root position is inside the configured takeoff window
and forward velocity >= 0.90 m/s
```

When this condition becomes true, `jump_signal_latched` becomes true and the
existing jump-window start/end bookkeeping activates. If the robot is already
airborne, phase/event logic may immediately recognize the corresponding
airborne continuation; it must not reset or reject the rollout merely because
wheel support was lost earlier.

The latch is monotonic for an entire rollout:

```text
false before legal-window entry
true on legal-window entry
true after leaving the window
```

Early airborne motion before this latch is diagnostic only. It cannot set the
formal liftoff-seen flag, satisfy Phase U, or substitute for full Apex-band
membership. `propulsion_ascent_success` continues to require every Apex gate,
including obstacle-relative position, clearance, kinematics, attitude, contact
legality, and absence of physical failure.

Legacy Takeoff counters and deadlines become active only after the formal jump
latch. Before the latch, positive-pitch, wheelie, liftoff-deadline, and
wheel-clearance-deadline failure conditions must remain false and their
counters must not advance. An early liftoff must not set
`dual_wheel_liftoff_seen`; legal liftoff accounting begins after the latch.

This is reward and event gating, not a success claim. Apex membership,
full-structure clearance, Descent-Recovery support, landing-region validity,
contact legality, stable hold, and physical-failure exclusions remain governed
by their existing Gate A/B contracts.

## Implementation Scope

The expected production change is limited to the smallest required portions
of the existing environment transition logic and directly affected Gate B
failure-audit semantics. No XML, geometry, action mapping, observation vector,
reset contract, matcher, virtual environment, or PPO algorithm is changed.

The old named `full_guideline_prelaunch_airborne` audit scenario is no longer
an expected failure after this change. The fixed diagnostic scenarios are
`full_guideline_continuation` and `launch_history_window_latch`. They record
the observed outcome and do not promise failure or success. Failure-video code
and tests must never require or falsely certify the retired end condition.
Any actual post-change Gate B pause is archived with MP4, state NPZ, telemetry,
manifest closure, actual `audit_outcome`, end code, terminal reason, event
ticks, and transition count. Rendering never decides Gate B status.

## Red-Green Tests

Tests must first demonstrate the old behavior and fail for the intended reason:

1. early airborne state lasting beyond
   `pretakeoff_airborne_fail_steps` does not terminate or emit end code 9;
2. the same rollout can subsequently enter the takeoff window and latch the
   jump signal without wheel support;
3. window-gated takeoff/ascent reward is zero before legal window entry and
   becomes eligible only after entry;
4. roll and pitch violations still terminate at the existing thresholds;
5. retained non-posture physical failures and timeout behavior are unchanged;
6. legacy snapshot fields remain round-trip compatible;
7. failure-video validation no longer requires the retired prelaunch terminal.
8. early airborne without window entry and valid Apex leaves
   `propulsion_ascent_success` false;
9. the jump latch follows false -> true -> true before/inside/after the window;
10. Takeoff counters, legal liftoff, and all four Takeoff failure codes remain
    inactive before the latch;
11. `prelaunch_airborne_count` survives timing-explicit snapshot round trip;
12. prohibited contact, invalid wheel contact, roll, pitch, backward motion,
    platform back-edge exit, and nonfinite dynamics remain hard failures.

## Gate B Recheck

After targeted tests pass, rerun the authoritative Gate B sequence from a new
ignored output directory using the same XML, config, reference, seed namespace,
geometry manifest, and threshold construction contract. Report event order,
first-event ticks, Apex width, recovery hold, terminal/truncation reason,
physical failures, timeouts, admitted snapshots, round-trip results, and exact
environment transitions.

If Gate B still pauses, preserve annotated videos and state traces for every
new named failure. If it passes, validate both phase banks and the formal
timing-explicit round trip. In either case, stop after Gate B and do not start
Gate C or any training.

Final verification includes static compilation, focused environment/reward and
Gate B tests, full pytest, `scripts/local_preflight.sh`, and a fresh runtime gate
because the environment transition fingerprint changes. Formal training
transitions remain zero.
