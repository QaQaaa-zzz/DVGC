# Phase U Relative-X Onset Diagnostic Design

## Hypothesis

The first feedback-braking grid began its first nonzero command after the
formal jump-window latch was observable. It found 50 stable-airborne/ascending
branches, but all were late high-rotation pitch failures. A natural-reset
timing probe measured full-structure `obstacle_relative_x` at ticks 15--18 as
1.168, 1.116, 1.063, and 1.010 m; the formal latch occurred at tick 18.

Test only whether starting the identical bounded feedback controller about
three, two, or one real control ticks earlier separates vertical propulsion
from unsafe rotation.

## Frozen Comparison

Reuse every one of the first diagnostic's 384 feedback specs without filtering
or changing coefficients. Cross them with exact onset values:

```text
obstacle_relative_x_onset: 1.17, 1.12, 1.07 m
```

The action becomes active when the deployable full-structure
`obstacle_relative_x <= onset`, remains active for the existing 4 or 7 ticks,
then returns to neutral. The phase-expert adapter and formal latch remain
unchanged. Pre-latch action, upward velocity, or airborne state receives no
window/ascent reward and implies neither event success nor termination.

Run all 1,152 branches once from seed 731100 with horizon 80, giving a maximum
of 92,160 diagnostic transitions and exactly zero PPO transitions. Do not
adapt onset or controller parameters after outcomes.

## Evidence and Claims

Use the same physical fields, outcome accounting, ranking, media selection,
hash provenance, and claim boundary as the first diagnostic. Additionally
record onset tick and formal latch tick separately. A successful branch is
only a physical diagnostic candidate, not an expert, reachable/safe state,
Tube, or reset.

If the grid produces valid Apex or nonphysical stable-ascent with materially
lower residual/rate, use only the distinguishing timing-aware deployable
quantity for the next single PPO hypothesis. If all early onsets remain unsafe
or miss launch, record a true limitation of this controller family before
another training change.

## Tests

- exact 1,152 unique spec/onset pairs;
- neutral before the onset and active at equality;
- active age is monotonic from onset and independent of formal latch;
- unchanged feedback action values after activation;
- manifest binds the three onsets, 92,160 ceiling, zero PPO, and false claims;
- report records onset and latch ticks separately and closes all outcomes;
- media selection occurs only after outcomes and cannot affect pass/fail.
