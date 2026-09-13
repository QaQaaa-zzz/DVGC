# Fresh Acceptance Bank and Repaired pi1 Execution Design

## Objective

Complete the pre-training fresh acceptance-bank blocker for the repaired
iteration-1 unified policy, and launch exactly one formal repaired `pi_1` run
only if every engineering and scientific-readiness gate passes.

This work does not change the scientific protocol. It preserves the frozen
`pi_0`, Tube_0, Tube_1, the rejected `pi_1`, the consumed 56-state audit, the
repair replay ratio, PPO budget, seed, reward, physics, reset mixture, action
semantics, validation isolation, and TEST isolation.

## Existing Evidence to Reuse

The completed real-dynamics acquisition is immutable and must not be repeated
or overwritten:

`JIT/runs/pi_unified_gate_prelock/pi_0_repair_acceptance_supportwide_acquisition_20260902`

It contains 659 unique TRAIN candidates produced from the previously audited
8 upstream and 8 downstream anchors, with 1,151 environment interactions under
a ceiling of 1,152. Its anchor audit is byte-identical to the preceding
zero-interaction audit. Validation, TEST, expert switching, and training were
not used.

## Confirmed Engineering Defects

### Canonical-copy identity mismatch

The acquisition CLI loads the predeclaration and writes a sorted, indented JSON
copy into the run directory. The labeling CLI compares the byte-level SHA-256
of that canonicalized copy with the byte-level SHA-256 of the source config.
The parsed payloads are identical and their declared canonical protocol hash is
identical, but formatting changes the file hash. Consequently the current
labeling CLI rejects the valid completed acquisition before GPU rollout.

The repair will validate all three relevant identities:

1. the source predeclaration file SHA-256 equals the SHA recorded by the anchor
   audit;
2. the copied and source predeclarations parse to identical JSON payloads;
3. the canonical protocol SHA-256 matches the predeclared value.

This permits the existing canonicalized copy without weakening semantic or
provenance validation. A changed field, changed protocol, wrong source file, or
wrong audit binding still fails before rollout.

### Missing readiness-failure artifact

The current negative-bank selector raises when either phase misses its state or
parent-group minimum. The completed labels remain on disk, but the phasewise
readiness counts exist only in the exception text.

The repair will preserve a machine-readable readiness status next to the label
artifacts. A readiness failure is recorded as a scientific/pre-training
readiness failure, not an engineering error. It includes each phase's observed
negative-state count, observed negative-parent-group count, required minima,
and readiness boolean. It does not create a locked acceptance bank and it
prevents repaired-policy training.

Engineering exceptions remain distinct and retain their existing provenance.

## Component Changes

Only existing stable capability files are modified:

- `JIT/cli/label_unified_continuations.py` validates the canonicalized
  predeclaration copy through source SHA, semantic equality, and canonical
  protocol identity before constructing the GPU runtime.
- `JIT/src/jit_dvgc/continuation/__init__.py` exposes readiness evidence when
  selection cannot meet the predeclared phasewise minima and persists a
  non-locked readiness-status artifact.
- Existing tests under `JIT/tests/` cover the new behavior. No retry-specific or
  stage-specific production module is added.

## Execution Flow

1. Add regression tests that reproduce the canonical-copy SHA mismatch and the
   missing readiness artifact; confirm they fail for the intended reasons.
2. Implement the minimum compatibility and readiness-persistence changes.
3. Run focused tests, compileall, the relevant non-GPU regression set, and the
   repository preflight appropriate to the change.
4. Validate the existing acquisition files, snapshot identities, interaction
   accounting, anchor identity, and TRAIN-only isolation again.
5. Run deterministic frozen-`pi_0` labeling once with the locked protocol:
   400 ticks, seed 9511005, no expert switching, no validation, no TEST.
6. Automatically select every baseline-negative candidate outside Tube_1,
   require physical-state uniqueness, and check both phasewise minima:
   10 states and 3 parent groups per phase.
7. If readiness fails, preserve labels and the readiness-status artifact,
   report the exact gaps, and stop with zero repaired-policy training
   transitions.
8. If readiness passes, write the locked acceptance bank, verify its canonical
   hash and file hash, verify Tube_1 overlap is zero, and update current
   provenance/status documentation.
9. Validate the repaired formal config and output identity, run zero-interaction
   plotting/runtime preflight and required GPU gates, and confirm no existing
   output directory would be overwritten.
10. Launch exactly one fresh repaired iteration-1 `pi_1` formal run for
    10,009,600 transitions with seed 821101, fresh actor/critic/optimizer,
    Tube_1, 45% core / 45% expansion / 10% natural reset mass, no validation,
    no TEST, and no expert switching.
11. Preserve terminal artifacts and report the exact run/checkpoint identities.
    Training completion alone does not accept the policy or authorize `C^1`,
    Tube_2, or `pi_2`.

## Error and Stop Semantics

- Any integrity, provenance, runtime, compilation, GPU, or accounting failure
  is an engineering stop. The output remains immutable and no replacement run
  is silently created.
- Insufficient phasewise negatives is a pre-training readiness FAIL. It is
  preserved separately from engineering errors and blocks training.
- A passing bank is launch authority only for the single predeclared repaired
  candidate. It is not evidence that the candidate will preserve the core or
  gain boundary capability.
- Final acceptance still requires the complete 222-state core-preservation gate
  and a paired boundary-gain gate using this newly locked bank.

## Verification Requirements

Before reporting the bank as PASS, verify and report acquisition counts and
SHAs, labeling totals and outcome closure, phasewise candidate/positive/negative
counts, phasewise negative parent groups, Tube_1 overlap, bank canonical SHA,
bank file SHA, environment interactions, zero training transitions, and false
validation/TEST flags.

Before launching training, independently verify the formal config identity,
fresh initialization contract, exact transition budget, reset probabilities,
seed, Tube_1 manifest, output-directory nonexistence, and all required
zero-interaction/GPU preflight results.

No success or completion claim is made from code inspection alone.
