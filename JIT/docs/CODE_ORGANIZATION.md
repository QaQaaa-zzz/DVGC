# JIT code organization

## Rule of thumb

`JIT/cli/` contains executable entry points only. `JIT/tests/` contains pytest
coverage only. Importable implementation belongs under `JIT/src/jit_dvgc/`.
The problem in the historical tree was not that implementation lived in
`src`; it was that experiment-stage modules accumulated as one flat namespace.

From this point forward, new production code must use a categorized canonical
namespace instead of adding another stage-named file at `src/jit_dvgc/` root.

## Canonical namespaces

| Responsibility | Canonical package | Legacy implementation path retained for compatibility |
| --- | --- | --- |
| unified training / formal PPO / freeze | `jit_dvgc.training` | `unified_training.py`, `unified_formal.py`, `unified_policy_freeze.py` |
| Soft Tube / Tube-RSI / Tube iteration | `jit_dvgc.tube` | `soft_tube.py`, `tube_rsi.py`, `tube_rsi_smoke.py`, `core_retaining_tube_iteration.py` |
| snapshot formats and pools | `jit_dvgc.snapshots` | `handoff_snapshot.py`, `unified_envelope_snapshot.py`, `snapshot_pool.py` |
| TRAIN-only diagnostics | `jit_dvgc.analysis` | `unified_diagnostic.py`, `unified_natural_evaluation.py` |
| continuation labels / fields / validation | `jit_dvgc.continuation` | continuation-related flat modules |
| real-dynamics boundary acquisition | `jit_dvgc.acquisition` | `unified_boundary.py`, `unified_transition_band_search.py` |

The legacy modules are intentionally **not deleted or renamed** in this phase.
Their module paths may be referenced by historical scripts, serialized Python
objects, frozen-artifact tooling, or external notebooks. The categorized
modules are compatibility facades over the existing authorities. Production
CLIs now import the canonical categorized paths.

## Migration policy

1. Do not create a new top-level production module merely because a new
   envelope iteration starts. `pi_2`, `Tube_2`, etc. are data/config/run
   identities, not reasons to create `pi2_*.py` files.
2. Put reusable orchestration in the appropriate package and expose it through
   a thin CLI.
3. Tests should mirror the capability package where practical. Existing flat
   tests may remain until touched.
4. Config files are run declarations and must stay immutable after a run starts.
   Completion/failure state belongs in run artifacts or status documentation,
   not by mutating the training config after the fact.
5. A legacy implementation may be physically moved only after a dedicated
   compatibility change proves old import paths and frozen artifacts still
   load. Until then, keep the old module as the authority or a shim.
6. `JIT/runs/` remains ignored runtime evidence. Do not copy large local run
   payloads into source control merely to document status.

## CLI boundary

CLI files should normally do only: parse arguments, load/dispatch one production
capability, print a machine-readable result, and exit. Scientific algorithms,
rollout logic, fitting logic, snapshot semantics, and provenance validation do
not belong in CLI files.

## Formal unified training hardening

The canonical `jit_dvgc.training.formal.run_unified_formal` performs a static
full-Tube snapshot/plot-support preflight before delegating to the existing
formal trainer. This consumes zero environment interactions and prevents the
mixed-snapshot plotting failure seen in the first pi_1 attempt from recurring
after a milestone panel has already been rolled out.
