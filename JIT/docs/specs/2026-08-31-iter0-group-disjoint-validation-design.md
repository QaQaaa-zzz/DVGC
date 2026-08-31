# Iteration-0 Group-Disjoint Expansion Validation Design

## Scope

This protocol freezes the next validation experiment; it does not execute it.
Validation is conditioned on frozen `pi_0`, uses no PPO updates or expert
switching, and may later calibrate `C_up^0` / `C_down^0`. Its rows may never be
copied into expansion TRAIN or `Tube_1` supervision.

## Held-out parents

TRAIN uses only `transition_4988928__1000001` through `__1000005`.
Validation uses seed `1000006`:

- upstream: one `ascending_entry` anchor from each of transitions 4,988,928,
  7,987,200, and 9,977,856;
- downstream: one earliest `post_apex` anchor from transitions 4,988,928 and
  9,977,856.

The 7,987,200 handoff bank has no downstream seed-1000006 post-Apex snapshot.
The protocol records two downstream parents instead of fabricating a state or
borrowing a TRAIN group.

Every anchor is bound by catalog/label file hashes, parent group, role, tick,
state SHA-256, snapshot path, legacy snapshot-parent identity, and
`split=validation`. Downstream handoff snapshots retain the older
`seed-1000006` internal parent name while their catalog/label group identity is
transition-qualified; both values are bound rather than silently equated.
Audit code reads the label
row only for identity and split; it does not inspect validation outcomes. TEST
outcomes and final-evaluation data remain untouched.

## Fixed panels

Upstream uses all four action axes, both signs, strengths
`0.025/0.05/0.10`, and durations `1/2`. Downstream uses `hip,+1`, strengths
`0.15/0.20/0.30/0.32/0.35/0.40/0.45/0.50`, and duration `30`. These are compact
families chosen from the closed TRAIN evidence, before validation outcomes.

The exact schedule contains 160 attempts, at most 696 acquisition interactions
and at most 64,000 deterministic frozen-policy labeling interactions.

## Leakage and claim boundary

The preflight rejects TRAIN parent overlap, exact state overlap, validation
anchor duplication, non-validation split, and actor-observation near duplicates
within an absolute all-feature tolerance of `0.01`. This supplements, rather
than replaces, exact physical-state hashes.

Passing the audit means only that the protocol is runnable and leakage-checked.
It does not mean validation was executed, `C^0` was trained/calibrated,
`Tube_1` exists, or a JCE/JEL/safety claim is available.
