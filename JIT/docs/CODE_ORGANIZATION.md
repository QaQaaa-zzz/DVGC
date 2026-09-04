# JIT code organization and lifecycle

## Principle

The active tree must represent **stable capabilities**, while configs, artifacts,
run directories, and manifests represent iterations/experiments.

Do not turn experiment identity into source-code structure.

## Stable package APIs

| Capability | Package API |
| --- | --- |
| unified PPO / formal preflight / freeze | `jit_dvgc.training` |
| Soft Tube / Tube-RSI / Tube iteration | `jit_dvgc.tube` |
| snapshot formats / pools | `jit_dvgc.snapshots` |
| boundary / transition-band acquisition | `jit_dvgc.acquisition` |
| continuation labels / fields | `jit_dvgc.continuation` |
| paired evaluation / capability progression | `jit_dvgc.analysis` |
| resumable iteration orchestration | `jit_dvgc.workflow` |

The capability-progression revision introduced one new durable analysis module:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

This is justified as a stable capability because it changes the reusable
scientific decision semantics for all future iterations.  It is not a `pi_2`
experiment-specific module.

CLI:

`JIT/cli/analyze_capability_progression.py`

The CLI remains thin and delegates scientific logic to the package module.

## Current decision architecture

The code now separates:

```text
locked paired policy evaluation
        ↓
strict behavioral diagnostic fields
        ↓
capability_progression analysis
  A. frontier progression
  B. phase-aware policy realization
        ↓
selection only for prospective A+B pass
```

Historical strict gate reports remain readable and are not rewritten.

`select_iteration_policy.py` supports:

- historical strict selection for old reproducibility paths;
- future prospective selection with a non-retrospective
  `jit_capability_progression_decision_v1` artifact.

Retrospective decisions cannot select a candidate.

## Modify-first rule

Before creating a production file, answer in order:

1. Can an existing implementation file be modified?
2. Can an existing package API expose the capability?
3. Can an existing CLI accept another schema/config?
4. Can an existing test file cover the behavior?
5. Is this genuinely a new durable scientific/runtime capability?

Only the final case normally justifies a new production file.

Iteration numbers, retry numbers, checkpoint counts, model seeds, and Tube IDs
belong in configs/manifests/run paths.  Do not create `pi3_*.py`, `tube3_*.py`, or
`retry04_*.py` merely because the next experiment exists.

## Active vs historical files

A file belongs in the active tree when it is one of:

- reusable production implementation;
- current stable CLI;
- current contract/regression test;
- current method/status/verification documentation;
- path-bound provenance/configuration required by retained artifacts.

Superseded narrative documents should either be replaced by current authority or
clearly marked historical.  Do not keep two conflicting “current status” stories.

Git history is the source archive for removed source code.  Do not duplicate
obsolete Python into an `archive/` directory.

Path-bound configs, frozen manifests, handoff locks, and artifact references may
need to remain at original paths for reproducibility.

## Current compatibility debt

Some historical flat/upstream modules remain because active loaders or artifact
reproduction still depend on them.  Do not delete based on naming alone.

Before deleting any Python/CLI/test file:

1. prove no production import/package API/current CLI/current test/artifact loader/
   config/frozen reproducibility path depends on it;
2. run at least:
   - `python -m compileall -q JIT/src JIT/cli`
   - targeted imports/tests for the affected capability;
3. stop if dependency closure is uncertain.

## Iteration-generic requirement

Reusable scientific code must obtain iteration identity from locked protocol data
and artifacts, not from hard-coded `pi_2`, `Tube_2`, or exact current counts.

Current run-specific values belong in reports/config/artifact metadata.

Existing Tube/frozen artifacts remain immutable.  Generalization must be backward
compatible with locked schemas and SHA-bound provenance unless an explicit
migration is introduced.

## CLI and tests

`JIT/cli/` files should:

- parse arguments;
- call one production capability;
- write/print machine-readable results;
- avoid containing fitting algorithms, physics, or complex scientific decision
  logic.

Tests should protect durable contracts.

Current capability-progression regression coverage:

`JIT/tests/test_capability_progression.py`

Generic workflow/Tube regression coverage:

`JIT/tests/test_iterative_envelope_automation.py`

## Workflow boundary

`jit_dvgc.workflow` is orchestration, not science.

It may:

- sequence commands;
- validate files/JSON assertions;
- persist/resume state;
- stop when a declared scientific artifact fails its prospective contract.

It may not:

- reinterpret a failed candidate itself;
- tune capability margins;
- change reward/replay/PPO/model/physics settings;
- turn a retrospective decision into a formal selection;
- touch final TEST evidence.

The current future DAG includes a separate capability-progression stage after the
locked paired evaluation and before policy selection.
