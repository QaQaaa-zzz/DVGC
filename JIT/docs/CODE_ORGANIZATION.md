# JIT code organization and lifecycle

## Principle

The active tree represents stable scientific/runtime capabilities. Configs, manifests and run directories represent iteration identities and experiments.

Do not encode experiment numbers into production module structure.

## Stable package areas

| Capability | Package area |
|---|---|
| unified PPO / formal training / freeze | `jit_dvgc.training` and current unified modules |
| raw Soft Tube / Tube-RSI / iterative replay | `jit_dvgc.tube` and current Tube modules |
| exact snapshot formats | snapshot modules under `jit_dvgc` |
| real-dynamics frontier acquisition | `jit_dvgc.acquisition` |
| continuation labels / fields | continuation modules under `jit_dvgc` |
| physical / capability analysis | `jit_dvgc.analysis` |
| resumable orchestration | `jit_dvgc.workflow` |

## Durable analysis capabilities

### Capability progression

```text
JIT/src/jit_dvgc/analysis/capability_progression.py
JIT/cli/analyze_capability_progression.py
```

Separates frontier progression from phase-aware single-policy realization.

### Physical capability Tube

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

Responsibilities:

- project exact qpos/qvel snapshots into physical coordinates;
- build `root_geometry_v1` and `full_physical_v1` cells;
- apply per-variable physical resolutions;
- quantify raw-to-cell duplication;
- compare Tube generations in cell space;
- build 0.10 m x-slice geometry and visualizations.

### Nominal jump centerline

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

Responsibilities:

- consume one completed successful canonical natural evaluation;
- use the existing every-frame qpos/qvel trace artifact;
- select one real frame per nominal 0.1 m x slice;
- stop at first valid landing or 4.2 m, whichever is earlier;
- enforce descending post-Apex points;
- forbid qpos/qvel interpolation.

This is a geometry scaffold, not an Actor goal/intent interface.

### Task-semantic Jump-Tube view

```text
JIT/src/jit_dvgc/analysis/jump_tube_view.py
JIT/cli/analyze_jump_tube_view.py
```

Responsibilities:

- preserve raw/control Tube immutability;
- filter capability accounting to nominal centerline x support;
- require downstream `root_vz < 0`;
- exclude post-landing recovery from Jump-Tube claims;
- report source-vs-target filtered root-cell growth;
- generate Jump-Tube x-z and x-z-vx plots.

## Durable acquisition capability

Trajectory-centered resolution-aware frontier revision:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The CLI requires a locked nominal centerline.

Parent diversity/semantics:

```text
newest raw Tube shell
+ unique parent group
+ unique exact state
+ unique root_geometry_v1 cell
+ centerline-supported x slice
+ downstream root_vz < 0
+ no post-landing recovery
```

Selection is local and x-balanced rather than globally lowest-score.

## Current decision architecture

```text
successful canonical rollout
        ↓
nominal centerline
        ↓
raw source Tube
        ↓
physical projection
        ↓
source Jump-Tube view
        ↓
outcome-blind ordinary frontier plan
        ↓
trajectory-centered x-balanced plan revision
        ↓
real-dynamics frontier roles
        ↓
continuation / raw next Tube
        ↓
next physical projection
        ↓
next Jump-Tube view
        ↓
require filtered Jump-Tube root-cell growth
        ↓
policy training
        ↓
locked paired evaluation
        ↓
capability progression + policy realization
        ↓
prospective selection
```

`JIT/cli/prepare_iterative_envelope_workflow.py` now emits this prospective stage structure when given a successful canonical evaluation report.

## Raw/Control versus Jump-Tube artifacts

Do not mutate historical Soft Tube artifacts to remove late recovery.

```text
raw/control Tube
  exact restartable snapshots
  replay/provenance

Jump-Tube view
  derived semantic/geometry artifact
  capability accounting/frontier eligibility
```

This split is intentional and should be preserved in future APIs.

## Modify-first rule

Before creating a production file:

1. Can an existing implementation file be modified?
2. Can an existing package API expose the capability?
3. Can an existing CLI accept another schema/config?
4. Can an existing test cover the behavior?
5. Is this genuinely a new durable scientific/runtime capability?

The centerline and Jump-Tube view modules were added because they are durable cross-iteration scientific capabilities, not run-specific scripts.

Iteration numbers, retry numbers, checkpoint counts and Tube IDs belong in configs/manifests/run paths. Do not create `pi3_*.py`, `tube3_*.py`, or retry-named production modules.

## Active versus historical files

Active tree files should be reusable production code, current CLIs, current regression tests, current authority documentation, or path-bound reproducibility inputs.

Superseded narratives should be clearly historical. Git history is the source archive for removed code; do not create obsolete Python archive folders.

## Compatibility and deletion gate

Before deleting any Python/CLI/test file:

1. prove no production import, package API, current CLI, current test, artifact loader, config or frozen reproducibility path depends on it;
2. run at least `python -m compileall -q JIT/src JIT/cli` plus targeted tests/imports;
3. stop if dependency closure is uncertain.

## Iteration-generic requirement

Reusable scientific code must obtain iteration identity from protocol/artifacts rather than hard-coded `pi_k`, Tube IDs, seeds or current row counts.

The only current hard task corridor constants are method-contract values for trajectory-centered Jump-Tube v1 (`2.5 m`, `4.2 m`, `0.1 m`); changing them requires a new method decision, not an experiment-specific patch.

## CLI rules

`JIT/cli/` files should:

- parse arguments;
- call production capability code;
- write/print machine-readable results;
- avoid embedding fitting algorithms or physics logic.

## Current regression tests

```text
JIT/tests/test_nominal_jump_centerline.py
JIT/tests/test_jump_tube_view.py
JIT/tests/test_capability_tube_resolution.py
JIT/tests/test_resolution_frontier_parent_selection.py
JIT/tests/test_capability_progression.py
JIT/tests/test_iteration_policy_selection_capability.py
JIT/tests/test_iterative_envelope_automation.py
```

The frontier test now explicitly rejects non-descending downstream states and late x support.

## Workflow boundary

`jit_dvgc.workflow` is orchestration, not science.

It may:

- sequence declared commands;
- validate files/JSON assertions;
- persist/resume state;
- stop when a declared scientific artifact fails.

It may not:

- invent a successful centerline;
- reinterpret failed candidates;
- tune margins/replay/PPO/physics;
- turn retrospective evidence into formal selection;
- touch final TEST evidence.

The current workflow now requires a pre-existing successful canonical evaluation report. It then locks the centerline and performs source/target Jump-Tube analyses as explicit recorded stages.
