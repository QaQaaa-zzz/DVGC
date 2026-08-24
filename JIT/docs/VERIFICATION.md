# JIT Phase U Verification

## Verified delivery

The independent `JIT/` Propulsion-Ascent engineering stack passed its declared
first-delivery gate on 2026-08-24. This establishes environment, PPO update,
checkpoint, accounting, and saved-video integrity only. It does not establish
Phase U learnability or a trained expert.

## Immutable inputs

| Input | Verified SHA-256 |
|---|---|
| `assets/orange_bike_4kg_horizontal.xml` | `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192` |
| `data/reference_jump.csv` | `612fe758eb1042481b9c7642cc9b92d3e9c14b4a75c9deaf5340183c928bc41f` |
| Resolved smoke config | `e96f4f4e35df041ffdf1525ee13fc356598917f5a600db15f9284c3e1d9ebed3` |

Host model tests additionally verified the 2 kg payload, hip/knee force limits
of `[-50, 50]`, action order `[steer, rear-wheel drive, hip, knee]`, and the
in-memory `0.005 s` simulation timestep.

## Fresh JIT preflight evidence

Command:

```bash
bash JIT/scripts/local_preflight.sh
```

Result: exit code 0.

- Static compilation and AST legacy-import scan passed.
- Non-GPU suite: `66 passed, 5 deselected in 15.87s`.
- GPU suite: `5 passed in 13.04s`.
- GPU coverage includes JIT reset/step, 50 ticks = 1 second, 1,024-environment
  finite rollout, outer wrapper-field preservation, and the real Brax wrapper
  extra-field contract.
- Offline reference analysis revalidated the retained CSV boundary.
- Successful run provenance verification passed.

No `jit_dvgc` runtime or CLI source imports the existing `dvgc` package.

## Repository compatibility

The repository-level preflight collected all 1,031 tests after JIT added an
isolated test-package boundary. Its complete run reported `1029 passed, 2
failed`; one JIT subprocess-path test was then fixed and passed independently.
The remaining failure is in the user's pre-existing modified
`tests/test_phase_u_launch_diagnostic.py`, where `frozen_manifest_payload()` is
called with an unsupported `mode=` argument. That JIT-external user work was
not changed.

A fresh repository suite with only that specific known user failure deselected
then reported `1030 passed, 1 deselected, 1 warning in 158.15s`. Thus JIT has
no remaining observed repository regression, while the unmodified full root
preflight is accurately recorded as not green because of the pre-existing
dirty-path mismatch.

## PPO engineering smoke

Successful run ID:
`phase_u_1024_one_block_20260824_seed820001_retry2`.

| Evidence | Result |
|---|---|
| JAX backend | GPU, NVIDIA GeForce RTX 4090 D |
| Parallel environments | 1,024 |
| Training transitions | exactly 25,600 |
| Brax evaluation transitions | 0 |
| Fixed evaluation transitions | 0 |
| Restored-policy diagnostic transitions | 31 |
| Total environment transitions | 25,631 |
| Final checkpoint restored | yes |
| Final checkpoint payload SHA-256 | `b4edf62e9b47b311df3893d6327f606320d755ae08a7e72c68995e9e6e10cb0d` |
| Captured states | 32 |
| Encoded video frames | 32 |
| Diagnostic terminal cause | `roll_limit` |

The restored-policy diagnostic reached the legal jump window but did not
liftoff, become stably airborne, ascend, or satisfy Apex. It ended on the roll
limit after 31 transitions; Apex success rate was 0 and physical failure rate
was 1. The final training KL was approximately 436.49. These values are
recorded as engineering diagnostics and are not promotion or learnability
evidence.

## Abnormal-attempt evidence

Two earlier run directories are retained and closed rather than overwritten:

1. The original run closed as `engineering_error` with 0 transitions after a
   Brax scan detected that wrapper-added `info` fields were not preserved.
2. `_retry1` closed as `engineering_error` with 0 transitions because Brax
   timeout bootstrapping required an explicit `time_out` adapter.

Both causes received focused regression tests before `_retry2`. No physics,
reward, observation, parallelism, or PPO-budget parameter was weakened for the
successful retry.

## Artifact and claim boundary

Run manifests, statuses, metrics, checkpoints, diagnostic states, and video
are under ignored `JIT/runs/` paths and are not Git delivery content. Rendering
replayed the 32 saved states without calling `env.step`, so it consumed zero
additional environment transitions and encoded no duplicated tail frames.

This delivery does not implement or claim a frozen Phase U expert, Phase D,
continuation labels, `V_up`/`V_down`, learned soft Tubes, unified Tube-RSI PPO,
or independent JCE/JEL certification.

## Formal Phase U run

Formal run `phase_u_formal_998400_seed820101_20260824_retry1` completed after
the engineering delivery above. Its strict verifier closed:

- 998,400 exact training transitions;
- 904 fixed-evaluation transitions and zero Brax-evaluation/diagnostic transitions;
- six identity-bound checkpoint payload hashes;
- five eight-label fixed panels and every trace hash;
- final checkpoint restore and deterministic finite inference;
- 22 saved/encoded frames for the final 21-transition representative trace.

All five panels returned 0 Apex successes and 8 `roll_limit` physical failures.
The final policy also regressed from reaching the window at 102,400/256,000 to
failing before the window at later milestones. It is not a trained expert and
must not be promoted. The complete modification rationale, PPO trends,
milestone tables, limitations, action diagnosis, interaction cost, and next-step
decision are in
`experiments/phase_u_formal_998400_seed820101_20260824/REPORT.md`.

The latest JIT preflight reported 106 non-GPU tests plus 5 GPU tests passing.
The fresh repository compatibility run reported 1,070 passed and only the
specific user-dirty relative-x manifest case deselected.
