# DVGC Experiment State

Current as of 2026-09-02 for branch `agent/two-phase-soft-tube`.

This is the compact recovery marker. For detailed artifact history use
`JIT/docs/CURRENT_STATUS.md`. For the scientific contract use
`JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`.

## Current marker

The active method is:

`experts -> Tube_0 -> pi_0 -> C^0 -> Tube_1 -> pi_1 -> paired gate -> C^1 -> Tube_2 -> pi_2 -> ...`

The project has completed the **repaired iteration-1 policy training and freeze**,
but iteration 1 is **not accepted yet**.

Current state:

```text
first pi_1 candidate
  -> core FAIL / boundary PASS
  -> replay-dilution diagnosis
  -> predeclared retained-core replay repair
  -> fresh baseline-only acceptance-bank acquisition
  -> two-axis fresh bank PASS
  -> repaired pi_1 trained for 10,009,600 transitions
  -> repaired pi_1 frozen
  -> NEXT: formal paired core + fresh-boundary acceptance gate
```

Do not start accepted `C^1`, `Tube_2`, or `pi_2` before that gate passes.
Final TEST/JCE/JEL remains untouched.

## Stable authorities

### Tube_0

- 222 TRAIN entries = 117 upstream + 105 downstream
- manifest SHA-256: `c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`

### pi_0

Frozen manifest:

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- 10,009,600 PPO transitions
- actor SHA-256: `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`
- payload SHA-256: `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

- 3,119 TRAIN entries
- exact retained Tube_0 core: 222
- expansion: 2,897 = 310 upstream + 2,587 downstream
- manifest SHA-256: `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`
- validation embedded: 0
- TEST embedded: 0

## Rejected first pi_1 candidate

The first completed `pi_1` candidate remains immutable comparison provenance.
Its paired `pi_0 -> pi_1` gate completed scientifically:

- 222-state core: `pi_0` 222 successes, candidate 201 successes
- core regressions: 21 = 16 upstream + 5 downstream
- **CORE PASS = false**
- old 56-state boundary bank: baseline reproduced all failures
- candidate gains: 12 states across 5 parent groups
- **BOUNDARY PASS = true**
- **ITERATION ACCEPTED = false**

The completed 56-state boundary bank is consumed and cannot be the sole fresh
acceptance evidence for the repaired candidate.

## Repaired iteration-1 method

Config:

`JIT/configs/pi_unified_iter1_tube1_core_replay50_natural10.json`

Only the within-Tube replay contract changed:

- phase: 50% upstream / 50% downstream
- within each phase: 50% retained core / 50% expansion
- retained core uniform; expansion keeps existing weighted sampling
- with 90% Tube / 10% natural reset, effective episode mass is
  45% retained core / 45% expansion / 10% natural

PPO budget, seed 821101, fresh actor/critic/optimizer, Tube_1 support, physics,
reward and action semantics remained fixed.

## Fresh acceptance evidence — completed

Two baseline-only single-axis readiness probes were preserved as scientific
pre-training FAIL evidence:

1. support-wide probe: 659 states, 12 negatives, all upstream, only 2 upstream
   parent groups, downstream 0;
2. stronger single-axis probe: 1,272 states, 58 negatives, all upstream, still
   only 2 upstream parent groups, downstream 0.

This showed that increasing single-axis strength/duration was not sufficient.
The acquisition capability was then generalized to systematic sparse two-axis
directions while keeping real dynamics and the same frozen `pi_0` baseline.

Two-axis acquisition:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_two_axis_acquisition_20260902`

- 3,720 unique fresh TRAIN candidates
- upstream 1,560 / downstream 2,160
- 18,829 acquisition interactions
- zero exact-state overlap with the two consumed readiness probes
- no validation/TEST/training

Long single-process labeling hit CUDA/Warp allocator OOM. The logical labeling
job was therefore executed as four sequential independent GPU processes of 930
candidates each, with the same candidate set, frozen policy, seed and 400-tick
horizon, and then merged once in original catalog order.

Merged root:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_two_axis_sharded_20260902/merged`

Fresh locked-bank result:

- 3,720 labels
- 260 frozen-`pi_0` negatives
- upstream: 246 negatives / 4 parent groups
- downstream: 14 negatives / 5 parent groups
- Tube_1 overlap: 0
- **PRE-TRAINING FRESH ACCEPTANCE BANK = PASS**

The sharding is execution-only; it did not change any scientific protocol field.
`JIT/runs` remains local/ignored runtime evidence unless explicitly committed.

## Repaired pi_1 — completed and frozen

Formal run:

`JIT/runs/pi_unified/pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902`

Frozen candidate:

`JIT/runs/frozen_unified/pi_1_core_replay50_10009600_20260902/frozen_unified_policy.json`

- requested/completed transitions: 10,009,600 / 10,009,600
- fresh actor/critic/optimizer
- seed 821101
- 45% retained-core / 45% expansion / 10% natural effective reset mass
- validation: false
- TEST: false
- expert switching: false
- checkpoint payload SHA-256: `ea93a534c2c6bb3bf145684cbea82df94fefa2df8099dcdcdd9492bd8007e205`
- frozen manifest file SHA-256: `d5a1658530d475a67264aa5c621283d71c823200dbee6068f93413b93d06b7a8`

This frozen policy is a **candidate comparison authority**, not yet the accepted
iteration-1 authority.

## Immediate next step

Run exactly one formal repaired `pi_0 -> pi_1` paired acceptance audit using:

- core bank: all 222 Tube_0 states;
- boundary bank: the locked fresh 260-state two-axis frozen-`pi_0` negative bank;
- baseline: frozen `pi_0`;
- candidate: frozen repaired `pi_1`;
- deterministic continuation, 400 ticks;
- no expert switching, validation, TEST or training.

Acceptance rules remain unchanged from the original gate:

- core preservation: zero baseline-success -> candidate-failure regressions;
- baseline-negative reproduction: every locked fresh negative must reproduce;
- boundary gain: candidate success in at least 2 distinct parent groups;
- iteration accepted only if core and boundary both pass.

If PASS: accept iteration 1, then proceed to frozen-`pi_1`-conditioned `C^1`,
fresh validation, core-retaining `Tube_2`, and `pi_2`.

If FAIL: preserve and diagnose; do not retune the gate or replay ratio against the
same consumed bank.

## Remaining implementation debt

Before unattended later iterations:

- paired gate input/provenance must be generalized to the new fresh-bank artifact;
- `core_retaining_tube_iteration.py` still contains iteration-0/Tube_1 constants;
- continuation refit/fresh validation still depend on some upstream-specific
  helpers;
- generic `C^k -> Tube_(k+1)` construction must be completed;
- workflow must stop on scientific failure and never auto-tune.

## Immutable task identity

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`
