# JIT code organization and lifecycle

## Principle

The active tree represents **stable capabilities**. Configs, artifacts, run directories and manifests represent iterations/experiments.

Do not turn experiment identity into source-code structure.

## Stable package APIs

| Capability | Package API |
| --- | --- |
| unified PPO / formal preflight / freeze | `jit_dvgc.training` |
| Soft Tube / Tube-RSI / raw Tube iteration | `jit_dvgc.tube` |
| snapshot formats / pools | `jit_dvgc.snapshots` |
| real-dynamics frontier acquisition | `jit_dvgc.acquisition` |
| continuation labels / fields | `jit_dvgc.continuation` |
| paired evaluation / capability analysis | `jit_dvgc.analysis` |
| resumable iteration orchestration | `jit_dvgc.workflow` |

## Durable analysis capabilities

### Capability progression

```text
JIT/src/jit_dvgc/analysis/capability_progression.py
JIT/cli/analyze_capability_progression.py
```

Separates:

```text
frontier progression
from
phase-aware single-policy realization
```

Historical strict gate reports remain readable and are not rewritten.

### Resolution-aware capability Tube

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

This is a durable capability because every current/future Tube can be projected through the same physical resolution schema independently of iteration number.

Responsibilities:

- project snapshot qpos/qvel into physical coordinates;
- maintain separate root-geometry and full-physical cell profiles;
- apply per-variable resolutions;
- quantify raw-to-cell duplication;
- compare source/target Tube generations in cell space;
- build 0.10 m x-slice cross-sections;
- generate machine-readable geometry and optional visual plots.

Raw Soft Tube artifacts remain replay/training data; the physical Tube analysis is a derived scientific representation and does not mutate them.

## Durable acquisition capability

Resolution-aware frontier parent selection:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

This capability revises a still-outcome-blind frontier plan so parent diversity requires both provenance separation and distinct `root_geometry_v1` cells.

It is iteration-generic and must not encode Tube_2/pi_2 identities.

## Current decision architecture

```text
raw replay Tube
        ↓
physical capability projection
        ↓
resolution-aware geometry/cells
        ↓
frontier parent plan
        ↓
real-dynamics acquisition
        ↓
continuation / raw next Tube
        ↓
policy training
        ↓
locked paired evaluation
        ↓
capability progression
  A. frontier progression
  B. phase-aware policy realization
        ↓
prospective policy selection
```

The latest physical-Tube layer is not yet integrated as first-class stages in one complete automatic workflow DAG. Until that integration is explicitly recorded, the operator must invoke the geometry and parent-plan revision capabilities before frontier role execution.

## Modify-first rule

Before creating a production file, answer in order:

1. Can an existing implementation file be modified?
2. Can an existing package API expose the capability?
3. Can an existing CLI accept another schema/config?
4. Can an existing test file cover the behavior?
5. Is this genuinely a new durable scientific/runtime capability?

Only the final case normally justifies a new production file.

Iteration numbers, retry numbers, checkpoint counts, seeds and Tube IDs belong in configs/manifests/run paths. Do not create `pi3_*.py`, `tube3_*.py` or retry-named production modules merely because a new run exists.

## Active vs historical files

A file belongs in the active tree when it is one of:

- reusable production implementation;
- current stable CLI;
- current contract/regression test;
- current method/status/verification documentation;
- path-bound provenance/configuration required by retained artifacts.

Superseded narrative documents should be clearly historical or replaced by current authority. Do not keep conflicting current-status narratives.

Git history is the source archive for removed source code. Do not duplicate obsolete Python into an archive directory.

Path-bound configs, frozen manifests, handoff locks and artifact references may need to remain at original paths for reproducibility.

## Compatibility debt and deletion gate

Historical flat/upstream modules may remain while active loaders or artifact reproduction still depend on them. Do not delete based on naming alone.

Before deleting any Python/CLI/test file:

1. prove no production import, package API, current CLI, current test, artifact loader, config or frozen reproducibility path depends on it;
2. run at least:
   - `python -m compileall -q JIT/src JIT/cli`
   - targeted imports/tests for the affected capability;
3. stop if dependency closure is uncertain.

## Iteration-generic requirement

Reusable scientific code must obtain iteration identity from locked protocol data and artifacts rather than hard-coded `pi_k`, Tube IDs or current row counts.

Existing Tube/frozen artifacts remain immutable. Generalization must be backward compatible with locked schemas and SHA-bound provenance unless an explicit migration is introduced.

## CLI rules

`JIT/cli/` files should:

- parse arguments;
- call one production capability;
- write/print machine-readable results;
- avoid containing fitting algorithms, physics, or complex scientific decision logic.

## Current regression tests

Capability progression:

```text
JIT/tests/test_capability_progression.py
JIT/tests/test_iteration_policy_selection_capability.py
```

Capability Tube resolution:

```text
JIT/tests/test_capability_tube_resolution.py
```

Resolution-aware parent diversity:

```text
JIT/tests/test_resolution_frontier_parent_selection.py
```

Generic workflow/raw Tube contracts:

```text
JIT/tests/test_iterative_envelope_automation.py
```

## Workflow boundary

`jit_dvgc.workflow` is orchestration, not science.

It may:

- sequence commands;
- validate files/JSON assertions;
- persist/resume state;
- stop when a declared scientific artifact fails.

It may not:

- reinterpret a failed candidate itself;
- tune capability margins;
- change reward/replay/PPO/model/physics settings;
- turn retrospective evidence into a formal selection;
- touch final TEST evidence.

The next workflow integration task, after retrospective validation of the resolution schema, is to add explicit stages for:

```text
source Tube physical analysis
resolution-aware frontier-plan revision
next Tube physical analysis
```

Do not integrate those stages blindly before the Tube_0/1/2 retrospective outputs are inspected.
