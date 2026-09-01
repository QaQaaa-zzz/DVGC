# JIT code organization and lifecycle

## The problem being corrected

The historical tree accumulated one Python module per experiment stage:
`upstream_*`, `downstream_*`, `unified_*`, CV repairs, parent-diversity passes,
and temporary diagnostics. The issue was not that implementation lived under
`src`; the issue was that experiment identity became source-code structure.

Envelope iteration must invert that relationship: **code represents stable
capabilities; configs and artifacts represent iterations.**

## Stable package APIs

Production CLIs should import from package roots:

| Capability | Package API |
| --- | --- |
| unified PPO / formal preflight / freeze | `jit_dvgc.training` |
| Soft Tube / Tube-RSI / Tube iteration | `jit_dvgc.tube` |
| snapshot formats / pools | `jit_dvgc.snapshots` |
| boundary / transition-band acquisition | `jit_dvgc.acquisition` |
| continuation labels / fields | `jit_dvgc.continuation` |
| TRAIN diagnostics | `jit_dvgc.analysis` |
| resumable iteration orchestration | `jit_dvgc.workflow` |

Each package root exposes the stable public API. We intentionally removed the
previous three-line-per-module facade layer; categorization must reduce cognitive
load, not multiply files.

## Modify-first rule

Before creating a production file, answer these questions in order:

1. Can the existing implementation file be modified?
2. Can an existing package API expose the capability?
3. Can an existing CLI accept another schema/config instead of creating another
   CLI?
4. Can an existing test file cover the behavior?
5. Is this really a new durable capability, or just a new iteration/run?

Only the final case normally justifies a new production file.

Iteration numbers, retry numbers, checkpoint counts, model seeds, and Tube IDs
belong in configs/manifests/run paths. They must not generate `pi2_*.py`,
`tube2_*.py`, `*_retry02.py`, or new source trees.

## Active vs historical files

A file belongs in the active tree only if it is one of:

- reusable production implementation
- current stable CLI
- current contract/regression test
- current method/status/verification documentation
- path-bound provenance/configuration required to reproduce retained artifacts

When stage-specific code has been superseded and no current implementation or
loader imports it, delete it from the active branch. Git history is the source
archive. Do not copy obsolete Python into `archive/` and keep both versions.

Exception: configs, frozen manifests, handoff locks, and other paths recorded by
artifact identity should remain at their original paths unless a dedicated
compatibility migration proves relocation safe.

## Current compatibility debt

Three iteration-0/upstream-named modules cannot yet be removed because current
production code still uses parts of them:

- `upstream_boundary.py` — legacy physical-state/hash helpers are still consumed
  by bootstrap/fresh-validation code
- `upstream_checkpoint_train_evidence.py` — current shared refit/Tube builder
  still reads the frozen upstream evidence through this loader
- `upstream_matched_checkpoint_domain_cv.py` (and its lower-level checkpoint-CV
  dependency) — generic code still imports sigmoid/tiny-MLP/CV helpers from it

The correct cleanup is to move those generic responsibilities into existing
iteration/continuation/snapshot authorities, then retire or reduce the upstream
modules. Do not delete them first and repair import failures afterward.

## Iteration-generic requirement

Reusable scientific code must accept `iteration=k` from locked protocol data.
It must not encode `pi_0`, `Tube_1`, `C_up^0`, exact iteration-0 candidate
counts, exact validation thresholds, or iteration-0 snapshot roots as Python
constants unless those values define a genuinely immutable method invariant.

Existing Tube_1 artifacts remain immutable. Generalization must be backward
compatible with their locked schemas and SHA-bound config semantics.

## CLI and tests

`JIT/cli/` files parse arguments, dispatch one production capability, print a
machine-readable result, and exit. They should not contain fitting algorithms,
physics logic, or scientific gate definitions.

Tests should protect current production contracts. When a retired research
module is removed, its experiment-specific tests should leave the active suite
with it; durable behavior should already be covered by the generic replacement.

## Workflow boundary

`jit_dvgc.workflow` is orchestration, not science. It may sequence commands,
verify files/JSON assertions, export artifact identities, persist state, and
resume. It may not reinterpret a failed gate, tune thresholds, alter PPO/reward
settings, or touch final TEST evidence.
