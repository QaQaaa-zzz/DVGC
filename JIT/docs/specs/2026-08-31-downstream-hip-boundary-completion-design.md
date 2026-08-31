# Downstream Hip-Positive Boundary-Completion Design

## Evidence and objective

The symmetric duration-30 strength panel produced the first downstream
continuation negatives: all five `hip`, positive-sign, strength-0.30 probes
ended at a real `pitch_limit` predecessor and were negative under fresh frozen
`pi_0` continuation. Every other action/sign/strength combination remained
positive. Accumulated downstream evidence is 2,589 positive and 5 negative,
with both classes already spanning five parent groups.

Run one final strength-only TRAIN panel to seek the remaining 15 distinct
negative candidates required by the locked readiness rule. If it fails, close
the strength-only acquisition family instead of increasing the budget again.

## Locked panel

- Exact prior: completed 3,165-label symmetric strength panel.
- Policy/Tube/anchors: unchanged frozen `pi_0`, `Tube_0`, and the same five
  downstream parent-group-unique anchors.
- Action/direction: only `hip`, sign `+1`, based on the only observed negative
  combination.
- Strengths: exactly `[0.32, 0.35, 0.40, 0.45, 0.50]`.
- Duration: exactly 30.
- Attempts: `5 anchors * 5 strengths = 25`.
- Acquisition ceiling: 750 interactions.
- Labeling ceiling: 10,000 interactions.

Higher strengths are not interpreted as linear distance because strength 0.30
already clips the hip action on later ticks. Every retained candidate must
still have a distinct physical-state SHA-256; saturated duplicates are excluded
and cannot count toward readiness.

## Stop boundary

If accumulated downstream evidence reaches at least 20 negatives while
retaining the already-satisfied positive/parent-group minima, freeze the
Iteration-0 TRAIN evidence and proceed only to group-disjoint expansion
validation design. Otherwise stop this acquisition family. In neither case does
this panel itself authorize `C^0`, `Tube_1`, `pi_1`, or any JCE/JEL claim.
