# Iteration-0 Group-Disjoint Expansion Validation Design

## Scope

This protocol freezes the next validation experiment; it does not execute it.
Validation is conditioned on frozen `pi_0`, uses no PPO updates or expert
switching, and may later calibrate `C_up^0` / `C_down^0`. Its rows may never be
copied into expansion TRAIN or `Tube_1` supervision.

The original 2026-08-31 declaration used an upstream `durations={1,2}` panel.
Before any validation rollout or outcome inspection, that panel was superseded
because it predominantly probes the already-known easy interior rather than the
`pi_0`-conditioned upstream TRAIN transition region. The held-out parents,
downstream panel, total attempt count, label budget, leakage boundary, frozen
policy and claim boundary remain unchanged. Git history preserves the original
declaration and the handoff record marks it `superseded_before_launch`.

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
Audit code reads the label row only for identity and split; it does not inspect
validation outcomes. TEST outcomes and final-evaluation data remain untouched.

## Fixed panels

### Upstream

The revised upstream validation panel remains symmetric across all four action
axes and both signs but is boundary-representative rather than interior-only:

- action axes: `steer`, `rear_wheel_drive`, `hip`, `knee`;
- signs: `-1`, `+1`;
- strengths: `0.025`, `0.10`;
- durations: `4`, `8`, `16` ticks;
- held-out parents: 3;
- attempts: `3 * 4 * 2 * 2 * 3 = 144`.

The low/high strength pair brackets relatively mild and strong perturbations
while the 4/8/16 duration grid covers the TRAIN region in which upstream
negative continuation evidence actually emerged. No validation outcome was
observed when this revision was made.

### Downstream

The downstream panel is unchanged from the 2026-08-31 declaration because it
already follows the TRAIN-localized failure direction:

- action: `hip`;
- sign: `+1`;
- strengths: `0.15/0.20/0.30/0.32/0.35/0.40/0.45/0.50`;
- duration: `30` ticks;
- held-out parents: 2;
- attempts: `16`.

The complete schedule therefore still contains exactly 160 attempts. The
revised acquisition ceiling is 1,824 environment interactions and the maximum
deterministic frozen-policy labeling budget remains 64,000 interactions.

## Runtime semantics

The runtime restores each audited legacy snapshot without reconstructing its
physical state or actor FIFO. It converts the source snapshot into the unified
phase metadata using the same start semantics as the existing unified Tube
reset: qpos/qvel/control, history, last action and upstream events are preserved;
administrative counters restart fresh; downstream event state is initialized at
the held-out downstream anchor.

Candidates are created only by authoritative `env.step` under the fixed
perturbation panel. If the perturbation causes a terminal transition, the last
finite nonterminal phase-local predecessor is saved. The perturbation terminal
outcome itself is provenance only and is never the continuation label. The
candidate is then restarted with a fresh 400-tick budget and labeled by frozen
`pi_0` under the same strict continuation semantics used for TRAIN evidence.

The runtime rejects candidate exact-state overlap with frozen TRAIN evidence,
rejects actor-observation near duplicates against TRAIN at the locked all-feature
absolute tolerance `0.01`, and rejects duplicate validation physical states.
Excluded attempts are not replaced, so the panel cannot adapt to validation
outcomes.

## Resume and interaction accounting

Acquisition is reusable only when it completed under the exact runtime protocol.
Labeling progress is written sequentially so a recoverable CUDA/JAX failure does
not require replaying already completed candidates. Interactions spent by a
failed labeling attempt are preserved in accounting; the interrupted candidate
may be deterministically replayed on resume. An incomplete acquisition fails
closed rather than being silently merged or overwritten.

## Leakage and claim boundary

The preflight rejects TRAIN parent overlap, exact state overlap, validation
anchor duplication, non-validation split, and actor-observation near duplicates
within an absolute all-feature tolerance of `0.01`. Runtime candidate filtering
reapplies TRAIN exact-state and near-observation isolation after real-dynamics
perturbation. This supplements, rather than replaces, parent-group disjointness.

Passing the audit means only that the protocol is runnable and leakage-checked.
Completing validation means only that held-out `pi_0` continuation evidence is
available for the next calibration decision. It does not mean `C^0` was trained,
`Tube_1` exists, or a JCE/JEL/safety claim is available.
