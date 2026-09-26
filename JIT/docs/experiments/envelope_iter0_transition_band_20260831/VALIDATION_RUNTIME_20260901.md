# Iteration-0 Expansion Validation Runtime Addendum — 2026-09-01

## Status

The 2026-08-31 TRAIN transition-band closure report remains authoritative for
all completed TRAIN evidence. No validation outcome had been inspected when
this addendum was written.

The validation declaration recorded at the end of `REPORT.md` was reviewed
before launch. Its held-out sources and downstream panel remain valid, but its
upstream `durations={1,2}` panel predominantly probes the already-established
easy interior and is not sufficiently representative of the upstream TRAIN
transition region for later `C_up^0` calibration.

Therefore the original scientific protocol SHA
`16c944e2edd16d3ac57656f219506399d63256bd37e7760ee6938a95f225f2b7`
is retained as historical provenance and marked superseded before launch.

## Revised locked protocol

The successor protocol SHA is
`9ec0a1e8c314cc5710688a3537fbd339f520d4db3c4268d20715bcde938586b0`.

Unchanged:

- frozen `pi_0` identity;
- frozen 3,190-row TRAIN evidence;
- validation seed `1000006`;
- 3 upstream and 2 downstream held-out parent trajectories;
- no TRAIN parent, exact-state, or near-observation overlap;
- downstream panel;
- total attempt count `160`;
- maximum successful labeling interactions `64,000`;
- no PPO, expert switching, TEST, or final-evaluation use;
- validation rows cannot become TRAIN or Tube supervision.

Revised upstream panel, based only on frozen TRAIN evidence:

- axes: `steer/rear_wheel_drive/hip/knee`;
- signs: `-1/+1`;
- strengths: `0.025/0.10`;
- durations: `4/8/16`;
- parents: 3;
- attempts: 144.

Downstream remains:

- axis: `hip`;
- sign: `+1`;
- strengths: `0.15/0.20/0.30/0.32/0.35/0.40/0.45/0.50`;
- duration: 30;
- parents: 2;
- attempts: 16.

The resulting maximum acquisition budget is 1,824 interactions.

## Runtime implementation

The stable runtime is implemented in:

- `JIT/src/jit_dvgc/expansion_validation_runtime.py`;
- `JIT/src/jit_dvgc/expansion_validation_runtime_preflight.py`;
- `JIT/cli/run_expansion_validation.py`.

Candidate states are generated only through authoritative unified dynamics.
Terminal-causing perturbations are clipped to the last finite nonterminal
phase-local predecessor; the perturbation terminal itself is provenance only.
The frozen unified policy then receives a fresh 400-tick continuation budget and
produces the authoritative validation label.

Before launch, the host-only runtime preflight reconstructs the exact 76-D
actor observation seen by the unified policy from each legacy held-out FIFO and
phase task bit. It must agree with the cached source observation and remain
disjoint from frozen TRAIN observations at the locked all-feature absolute
`tolerance=0.01`.

No validation runtime result is claimed by this addendum. The next action is to
run focused CPU/GPU contracts, full JIT preflight, zero-interaction runtime
audit, and only then execute the fixed 160-attempt validation once.
