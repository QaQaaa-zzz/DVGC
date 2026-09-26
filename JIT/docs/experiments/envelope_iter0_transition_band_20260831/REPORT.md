# Iteration-0 Transition-Band Closure Report

## Executive result

Frozen unified policy `pi_0` now has a complete TRAIN-only transition-band
label set for both physical phases. The terminal artifact contains 3,190 unique
real-dynamics states:

| phase | positive | negative | positive groups | negative groups | ready |
| --- | ---: | ---: | ---: | ---: | --- |
| upstream | 545 | 26 | 5 | 5 | yes |
| downstream | 2,589 | 30 | 5 | 5 | yes |

This closes the Iteration-0 TRAIN boundary-evidence gate. It does not train a
continuation field, construct `Tube_1`, train `pi_1`, or establish JCE/JEL or
safety.

## 1. Recovery of the interrupted downstream refinement

The original downstream duration refinement failed before launch because its
config omitted the exact upstream/downstream `candidate_count` fields required
by strict readiness equality. The declaration was corrected and artifact audit
was expanded to bind the prior summary, labels, frozen policy, source Tube,
anchors, config, and repository source.

The first real refinement reached duration 23 and then hit a CUDA allocation
failure during labeling. No kernel OOM or NVIDIA Xid was recorded. A
source-bound repair resume preserved the failed 998 labeling interactions,
reused completed acquisition, wrote a retry label directory, and completed
durations 23 through 32 without another CUDA error.

Final local-refinement result:

- status: `search_exhausted`;
- acquisition interactions: 47,020;
- labeling interactions: 15,227, including the preserved 998-interaction
  aborted attempt exactly once;
- downstream: 2,474 positive, 0 negative;
- upstream remained ready at 545 positive / 26 negative.

The duration grid saturated on successful recovery and could not close the
downstream negative side.

## 2. Symmetric strength extrapolation

A fixed duration-30 panel was predeclared across all four action axes, both
signs, and strengths `0.15/0.20/0.30`, using five downstream TRAIN parents. It
ran once and exited normally:

- attempts/candidates: 120/120;
- positive/negative: 115/5;
- acquisition/labeling interactions: 3,595/215;
- all five negatives: `hip,+1,strength=0.30,duration=30`, one per parent group;
- terminal probe outcome for all five negatives: `pitch_limit`.

This established a reproducible negative direction but did not yet meet the
minimum 20 negative candidates.

## 3. Targeted hip-positive boundary completion

The final TRAIN panel retained `hip,+1,duration=30` and fixed strengths
`0.32/0.35/0.40/0.45/0.50`. It ran exactly 25 attempts:

- 25 candidates and 25 distinct physical-state hashes;
- 0 positive / 25 negative;
- 20 `pitch_limit`, 5 `roll_limit` terminal probes;
- acquisition interactions: 616;
- labeling interactions: 25;
- normal exit, status `transition_band_ready`.

Combining the strength and targeted panels gives exactly five downstream
negative labels at each strength `0.30/0.32/0.35/0.40/0.45/0.50`.
Cross-run attribution must use `(state_sha256, acquisition_protocol_sha256)`:
`candidate_id` is run-local and is reused.

## 4. Recent interaction accounting

| run | acquisition | labeling | total |
| --- | ---: | ---: | ---: |
| duration refinement, including repair | 47,020 | 15,227 | 62,247 |
| symmetric strength panel | 3,595 | 215 | 3,810 |
| targeted hip completion | 616 | 25 | 641 |
| **total** | **51,231** | **15,467** | **66,698** |

All are diagnostic/acquisition/labeling interactions. PPO training transitions
in these runs are zero.

## 5. Frozen TRAIN evidence

The accumulated labels were copied into a separately self-hashed artifact at
`JIT/runs/pi_unified_transition_band/pi_0_iter0_train_evidence_frozen_20260831`.

Key identities:

- frozen manifest SHA-256:
  `27832237a85eccfa0ae2eaea7575dd2efda12535a3815fee0c6d0660f58921b8`;
- copied labels SHA-256:
  `32e45f021e05a96b3098354a927909869dcb8f1e71d4798481ae2edc0f4e0323`;
- policy actor SHA-256:
  `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`;
- policy payload SHA-256:
  `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`;
- source `Tube_0` manifest SHA-256:
  `c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`;
- observation size: 76;
- new interactions/training transitions: 0/0.

The freeze rejects non-TRAIN rows, duplicate physical states, invalid or
non-finite observations, policy/protocol/readiness drift, and later label-file
tampering.

## 6. Group-disjoint validation protocol

The next experiment is predeclared and audit-ready but was not launched. TRAIN
parent groups are `transition_4988928__1000001..1000005`. Validation uses
held-out seed 1000006:

- upstream: transitions 4,988,928, 7,987,200, and 9,977,856;
- downstream: transitions 4,988,928 and 9,977,856.

The 7,987,200 downstream handoff bank lacks a real seed-1000006 post-Apex
snapshot. The protocol uses two downstream parents instead of fabricating a
state or borrowing TRAIN evidence.

The audit binds five snapshots and their catalog/label hashes, confirms
`split=validation` without reading outcomes, and reports:

- TRAIN parent overlap: 0;
- exact physical-state overlap: 0;
- observation near-duplicate overlap at absolute all-feature tolerance 0.01: 0;
- TEST/final-evaluation data used: false/false;
- fixed attempts: 160;
- acquisition ceiling: 696 interactions;
- labeling ceiling: 64,000 interactions;
- training transitions: 0.

Protocol SHA-256 is
`16c944e2edd16d3ac57656f219506399d63256bd37e7760ee6938a95f225f2b7`.

## 7. Validation and stopping point

The freeze implementation first passed 425 non-GPU and 14 GPU tests. After the
validation-protocol implementation, the final full JIT preflight passed 428
non-GPU tests (34 deselected) and all 14 GPU tests.

Per the requested stopping point, no group-disjoint validation simulation is
started and no `C_up^0`, `C_down^0`, `Tube_1`, or `pi_1` work is performed.
The next session must implement/execute the exact predeclared validation runtime
and inspect its results before continuation-field fitting or calibration.
