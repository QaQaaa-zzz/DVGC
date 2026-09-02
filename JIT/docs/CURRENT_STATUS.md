# Current JIT status — 2026-09-02

## Executive state

The project is at **iteration 1 acceptance**, not at iteration 2.

The first Tube_1 policy candidate was scientifically rejected because it lost 21
Tube_0 core states. A retained-core replay repair was then predeclared, fresh
acceptance evidence was acquired under frozen `pi_0`, and the repaired
iteration-1 candidate was trained and frozen.

Current marker:

```text
Tube_0 / pi_0 / C^0 / Tube_1
  -> first pi_1 candidate
  -> paired gate: core FAIL, boundary PASS
  -> replay-dilution diagnosis
  -> fresh pre-training acceptance-bank acquisition
  -> fresh two-axis bank PASS
  -> repaired pi_1 training complete
  -> repaired pi_1 frozen
  -> NEXT: repaired pi_0 vs pi_1 paired acceptance gate
```

**Iteration 1 is not accepted yet.** Do not start accepted `C^1`, `Tube_2`, or
`pi_2` until the repaired candidate passes both core preservation and fresh
boundary gain.

Final TEST/JCE/JEL remains untouched.

## Immutable task identity

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control rate: 50 Hz
- hip/knee torque limits: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`
- expert switching in unified-policy evaluation/training: forbidden
- validation rows in Tube: forbidden
- TEST/final data before final policy selection: forbidden

## Completed bootstrap and Tube_1

### Experts

`pi_up_star`

- 9,977,856 transitions
- actor SHA-256: `f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081`

`pi_down_star`

- 25,600 transitions
- actor SHA-256: `7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be`

Frozen expert manifest:

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

- 222 TRAIN entries
- manifest SHA-256: `c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b`

### pi_0

`JIT/runs/pi_unified/pi_unified_round1_natural10_10009600_seed821101_20260831`

Frozen authority:

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

- 10,009,600 PPO transitions
- actor SHA-256: `43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029`
- payload SHA-256: `fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719`
- role: iteration-0 envelope-expansion authority, not final JCE/JEL authority

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

- exact retained Tube_0 core: 222
- expansion: 2,897 = 310 upstream + 2,587 downstream
- total: 3,119 = 427 upstream + 2,692 downstream
- manifest SHA-256: `817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80`
- entries SHA-256: `61c6796aaf4c4b1e43624c5cf06bce0d39736a6d1743c5142c6c250d23155ec9`
- validation rows embedded: 0
- TEST rows embedded: 0
- training guidance only; not a certified safe set

## First iteration-1 candidate — rejected and preserved

Completed run:

`JIT/runs/pi_unified/pi_1_tube1_natural10_10009600_seed821101_20260901_retry01`

Frozen comparison authority:

`JIT/runs/frozen_unified/pi_1_iter1_10009600_20260901/frozen_unified_policy.json`

Paired gate:

`JIT/runs/pi_unified_gate/pi_0_to_pi_1_paired_core_boundary_20260901_retry01`

Protocol SHA-256:

`24a126ee94472eebbcb59fff66618ae00dae41074a1d1cfee8bb816afaff410a`

Result:

- core bank: all 222 Tube_0 states
- `pi_0` core success: 222 / 222
- rejected `pi_1` core success: 201 / 222
- regressions: 21 = 16 upstream + 5 downstream
- **core preservation: FAIL**
- old boundary bank: 56 frozen `pi_0` continuation-negative TRAIN states
- baseline reproduction failures: 0
- rejected `pi_1` successes: 12 / 56 across 5 parent groups
- **boundary gain: PASS**
- **iteration accepted: false**

This is immutable scientific rejection evidence. The old 56-state boundary bank
is consumed for repaired-candidate selection and may only be used descriptively.

## Replay-dilution diagnosis and repaired method

Zero-interaction diagnosis:

`JIT/runs/pi_unified_gate_analysis/pi_0_to_pi_1_core_regression_20260901/diagnosis.json`

SHA-256:

`61e12385ef0e77180b773a2e0de04b36e2a27f649c24d509b4f18345c00a7689`

Material finding:

- Tube-conditional old-core probability under the rejected sampler: ~13.98%
- all-episode old-core reset mass: ~12.59%
- expansion reset mass: ~77.41%
- natural reset mass: 10%
- all 21 regressions were physical failures

The precise interpretation is **core-support under-replay / capability
regression under the expanded training distribution**. The evidence does not
prove replay dilution was the only mechanism.

Repaired formal config:

`JIT/configs/pi_unified_iter1_tube1_core_replay50_natural10.json`

Only the within-Tube replay contract changes:

- phase mixture: 50% upstream / 50% downstream
- inside each phase: 50% retained source core / 50% current expansion
- retained core: uniform
- expansion: existing value/continuation-weighted rule
- all-episode reset mass: 45% retained core / 45% expansion / 10% natural
- PPO budget, seed, physics, reward, action semantics, Tube_1 support and fresh
  actor/critic/optimizer initialization remain unchanged

## Fresh acceptance-bank acquisition — completed on 2026-09-02

The repaired candidate was not allowed to train until a fresh, baseline-only
boundary acceptance bank was available.

### Readiness probe 1 — support-wide single-axis shell: FAIL

Local runtime paths:

- acquisition: `JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_acquisition_20260902`
- labels: `JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_labels_20260902`

Result:

- 659 fresh candidates
- 647 positives / 12 `pi_0` negatives
- upstream: 12 negatives across 2 parent groups
- downstream: 0 negatives across 0 parent groups
- readiness requirement remained >=10 negative states and >=3 parent groups per phase
- **readiness: FAIL**

This was a pre-training readiness failure, not a candidate-policy result.

### Readiness probe 2 — extended single-axis shell: FAIL

Protocol used stronger real-dynamics pulses while keeping the acceptance rule
unchanged.

Local runtime paths:

- acquisition: `JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_extended_shell_acquisition_20260902`
- labels: `JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_extended_shell_labels_20260902`

Result:

- 1,272 fresh candidates
- 58 `pi_0` negatives
- upstream: 58 negatives, still only 2 parent groups
- downstream: 0 negatives
- exact-state overlap with the first readiness probe: 0
- **readiness: FAIL**

This ruled out the simple explanation that the single-axis shell was merely too
weak.

### Two-axis real-dynamics acquisition: PASS

The production acquisition capability was generalized from one active action
axis to configurable sparse action directions. `active_action_dimensions=1`
retains historical behavior; `=2` enumerates systematic action pairs and sign
combinations. The iteration contract was also generalized from hard-coded
`0 -> 1` to `k -> k+1`.

Predeclaration:

`JIT/configs/envelope_iter1_repair_acceptance_boundary_acquisition_two_axis.json`

Two-axis acquisition runtime path:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_two_axis_acquisition_20260902`

Result:

- 3,720 unique fresh TRAIN candidates
- upstream: 1,560
- downstream: 2,160
- acquisition interactions: 18,829 / 20,160 ceiling
- exact overlap with both prior readiness probes: 0
- training transitions: 0
- validation/TEST/final data: none

### Labeling engineering failure and process-sharded recovery

Long single-process labeling repeatedly exhausted CUDA/Warp allocator state even
though the scientific candidate set and protocol were valid. These attempts are
engineering-error provenance and do not count as scientific readiness results.

The successful execution strategy partitioned the same immutable 3,720-candidate
logical labeling job into four sequential independent GPU processes of 930
candidates each. Every process used the same frozen `pi_0`, deterministic policy,
protocol seed, and 400-tick horizon. The processes were merged back in original
catalog order before a single acceptance-bank lock.

This was an execution-only partition: no candidate, label rule, seed, horizon,
physics, threshold, or acceptance criterion changed.

Merged local runtime root:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_two_axis_sharded_20260902/merged`

Merged result:

- labels: 3,720
- frozen-`pi_0` negatives: 260
- upstream negatives: 246 across 4 parent groups
- downstream negatives: 14 across 5 parent groups
- Tube_1 overlap: 0
- training transitions: 0
- validation/TEST/final data: none
- **fresh acceptance-bank readiness: PASS**

The raw `JIT/runs` files are runtime artifacts and are normally ignored by Git.
This document records their current local paths and scientific role; exact
runtime file hashes remain in the generated manifests/JSON artifacts.

## Repaired pi_1 — training complete and frozen

Formal run:

`JIT/runs/pi_unified/pi_1_tube1_core_replay50_natural10_10009600_seed821101_20260902`

Frozen candidate:

`JIT/runs/frozen_unified/pi_1_core_replay50_10009600_20260902/frozen_unified_policy.json`

Result:

- requested/completed PPO transitions: 10,009,600 / 10,009,600
- seed: 821101
- actor/critic/optimizer: fresh initialization
- natural reset: 10%
- Tube reset: 90%
- effective all-episode retained-core mass: 45%
- effective all-episode expansion mass: 45%
- validation data used: false
- TEST data used: false
- expert switching: false
- exact final checkpoint restored before freeze

Known exact identities from the completed local run:

- final checkpoint payload SHA-256: `ea93a534c2c6bb3bf145684cbea82df94fefa2df8099dcdcdd9492bd8007e205`
- frozen manifest file SHA-256: `d5a1658530d475a67264aa5c621283d71c823200dbee6068f93413b93d06b7a8`

Training/freeze completion is **not** iteration acceptance.

## Current scientific blocker / immediate next step

The only active scientific blocker is the repaired `pi_0 -> pi_1` paired
acceptance gate.

The gate must use:

1. **core bank**: all 222 Tube_0 core states;
2. **fresh boundary bank**: the locked 260-state two-axis frozen-`pi_0` negative
   bank described above;
3. frozen baseline: exact `pi_0` authority;
4. frozen candidate: exact core-replay repaired `pi_1`;
5. deterministic continuation, 400-tick horizon, no expert switching;
6. no validation and no TEST/final data.

The scientific acceptance rules remain the original paired-gate rules:

- **core preservation PASS**: zero baseline-success -> candidate-failure
  regressions across the 222-state core;
- **boundary reproduction PASS**: every locked fresh baseline-negative challenge
  must reproduce as baseline failure under the gate runtime;
- **boundary gain PASS**: repaired `pi_1` converts failures to successes in at
  least 2 distinct parent groups;
- **iteration accepted** only if both core and boundary gates pass.

Do not raise/lower these rules after seeing repaired-candidate outcomes.

If the repaired gate passes, iteration 1 becomes accepted and work may proceed
to frozen-`pi_1`-conditioned `C^1`, independent fresh validation, core-retaining
`Tube_2`, then `pi_2`.

If either gate fails, preserve the result and diagnose. Do not sweep replay
ratios, thresholds, reward, PPO settings, or the gate against the same consumed
bank.

## Repository / implementation state

Completed on the active branch:

- generic retained-core replay contract in the existing Tube-RSI/formal stack;
- formal preflight made schema-driven rather than pi1-filename-driven;
- semantic predeclaration identity verification and machine-readable readiness
  artifact;
- acquisition generalized to configurable sparse action directions;
- acquisition iteration contract generalized to `k -> k+1`;
- long-labeling CUDA/Warp failure diagnosed as engineering execution pressure;
- process-sharded labeling demonstrated without changing scientific semantics.

Remaining migration debt before unattended later iterations:

- formal paired gate must be pointed at the new locked 260-state bank and its
  snapshot provenance root;
- `core_retaining_tube_iteration.py` still contains Tube_1 / iteration-0
  constants;
- shared continuation refit/fresh validation still depend on some
  upstream-specific evidence/CV helpers;
- generic `C^k -> Tube_(k+1)` construction must be completed before unattended
  `pi_1 -> pi_2` progression;
- workflow automation must stop on scientific failure and never auto-tune.
