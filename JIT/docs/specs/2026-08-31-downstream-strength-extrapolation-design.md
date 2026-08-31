# Downstream Strength-Extrapolation Design

## Objective

The completed duration 17--32 local refinement is scientifically closed but
downstream-degenerate: all 1,909 new candidates are positive, including the
terminal-predecessor saturation at durations 31--32. Continuing the duration
grid would repeat recovery-success predecessors rather than bracket frozen
`pi_0` continuation feasibility.

Run one bounded TRAIN-only strength-extrapolation panel through the existing
downstream-refinement capability. The panel must find a real negative boundary
or close this single-action-basis route without training `C^0`, building
`Tube_1`, or launching `pi_1`.

## Locked panel

- Frozen policy: exact Iteration-0 `pi_0` already bound by the frozen manifest.
- Prior evidence: the completed 3,045-label repaired refinement summary.
- Phase/split: downstream TRAIN only; upstream labels remain frozen.
- Anchors: the same deterministic five parent-group-unique downstream Tube
  anchors selected under the existing frontier ceiling.
- Action family: all four canonical single-action basis axes, both signs.
- Strengths: exactly `[0.15, 0.20, 0.30]`.
- Duration: exactly 30 ticks, the last observed duration before terminal
  predecessor saturation began.
- Acquisition ceiling: `5 * 4 * 2 * 3 * 30 = 3,600` interactions.
- Labeling ceiling: `5 * 4 * 2 * 3 * 400 = 48,000` interactions.
- Label semantics: one fresh deterministic frozen-`pi_0` continuation per exact
  real-dynamics state, with the unchanged 400-tick horizon.

All strengths remain bounded through the existing action clipping. Physics,
reward, action mapping, Tube, policy, terminal clipping, deduplication, and
claim boundaries are unchanged.

## Stable-entrypoint implementation

Keep `refine_downstream_transition_band.py` and
`downstream_transition_refinement.py` as the single stable capability. Extend
the config validator with two explicit locked variants:

1. `contiguous_integer_local_refinement`: existing strengths and duration
   17--32 exactly;
2. `fixed_duration_strength_extrapolation`: new strengths and duration 30
   exactly.

The search/acquisition protocols must record distinct purposes derived from the
variant. Any other strength, duration, action order, sign set, or search mode is
rejected.

## Stop and claim boundary

The accumulated readiness rule remains at least 20 positive and 20 negative
candidates across at least three parent groups on each side. Stop immediately
if it is satisfied; otherwise stop after this one panel and reject further
blind single-axis strength/duration expansion.

This panel is boundary-search TRAIN evidence only. It cannot support a learned
continuation field, Tube expansion, policy training, certified safe-set, or
JCE/JEL claim by itself.
