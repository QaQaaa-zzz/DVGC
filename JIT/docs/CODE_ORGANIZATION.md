# JIT code organization and lifecycle

## Principle

The active tree represents durable scientific/runtime capabilities. Configs, manifests and run directories represent experiment identities and iteration state.

Do not encode experiment numbers into production module structure.

The current scientific architecture is **causal, trajectory-centered, resolution-aware Jump-Capability identification**. Code organization must preserve the distinction between:

```text
forward reachability
continuation viability
raw/control replay support
causal capability evidence
single-policy realization
```

---

## Stable package areas

| Capability | Package area |
|---|---|
| unified PPO / formal training / freeze | unified training modules under `jit_dvgc` |
| raw Soft Tube / Tube-RSI / iterative replay | Tube modules under `jit_dvgc` |
| exact snapshot formats | snapshot modules under `jit_dvgc` |
| frontier / reachability acquisition | `jit_dvgc.acquisition` |
| continuation labels / fields | continuation modules under `jit_dvgc` |
| causal role orchestration | `jit_dvgc.causal_frontier_protocol` |
| physical / capability analysis | `jit_dvgc.analysis` |
| resumable orchestration | `jit_dvgc.workflow` + workflow CLIs |

---

## Durable analysis capabilities

### Physical capability projection

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

Responsibilities:

- project exact snapshot qpos/qvel into physical coordinates;
- build `root_geometry_v1` and `full_physical_v1` cells;
- apply declared physical resolutions;
- build 0.10 m x slices;
- quantify raw-to-cell duplication;
- compare all-state control-support geometry across historical Tubes.

This module does **not** establish natural-start reachability.

### Nominal jump centerline v2

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

Responsibilities:

- consume one completed successful natural-start canonical evaluation;
- select real trace frames at 0.1 m x spacing;
- stop at first valid landing or 4.2 m;
- forbid qpos/qvel interpolation;
- record physical-state SHA and transitions-from-ground per point;
- provide one fixed cross-iteration geometric scaffold.

The centerline is not an Actor goal/intent interface.

### Historical semantic Jump-Tube view

```text
JIT/src/jit_dvgc/analysis/jump_tube_view.py
JIT/cli/analyze_jump_tube_view.py
```

Responsibilities:

- diagnose historical raw/control Tube contamination;
- filter by centerline corridor;
- require descending downstream;
- exclude late recovery from jump-geometry accounting.

This is a retrospective semantic diagnostic and **not** a reachability proof.

### Causal Jump Capability analysis

```text
JIT/src/jit_dvgc/analysis/causal_jump_capability.py
JIT/cli/analyze_causal_jump_capability.py
```

Responsibilities:

- validate causal acquisition catalogs;
- validate `jit_ground_reachability_provenance_v1`;
- match exact reached snapshots to continuation labels;
- enforce phase/descending/pre-contact semantics;
- build resolution-aware `Reachable ∩ Viable` capability cells;
- report TRAIN/CALIBRATION/ACCEPTANCE evidence separately;
- compare current causal capability against an optional previous causal summary.

This is the primary scientific capability artifact for future rounds.

### Capability progression / realization

```text
JIT/src/jit_dvgc/analysis/capability_progression.py
JIT/cli/analyze_capability_progression.py
```

Responsibilities:

- separate frontier/capability progression from one-policy realization;
- never erase prior capability evidence merely because a later Actor regresses.

Historical gate evidence must retain the method version under which it was acquired.

---

## Durable acquisition capabilities

### Causal ground-connected acquisition

```text
JIT/src/jit_dvgc/acquisition/causal_jump.py
```

Responsibilities:

```text
natural ground reset
-> frozen-policy prefix
-> enter declared look-back window
-> bounded action perturbation
-> env.step only
-> capture valid target-slice candidate
```

Hard provenance requirements:

```text
natural_start_connected = true
generated_by_env_step_only = true
rsi_used_to_establish_reachability = false
qpos_qvel_injection_used = false
proposal_anchor_used_as_reset = false
```

Initial look-back family is 0.1/0.2/0.3 m.

### Every-slice causal frontier planning

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Responsibilities:

- use locked centerline slices as proposal targets;
- create deterministic pre-outcome TRAIN/TRAIN/TRAIN/CALIBRATION/ACCEPTANCE proposal families per usable slice;
- treat proposal anchors as identifiers, never physical reset states;
- preserve role isolation before outcomes.

Active revision schema: `jit_causal_trajectory_frontier_plan_revision_v2`.

---

## Causal role protocol

```text
JIT/src/jit_dvgc/causal_frontier_protocol.py
JIT/cli/run_causal_jump_frontier_role.py
```

Responsibilities:

```text
run ground-connected forward acquisition
-> validate reachability provenance
-> restore exact already-reached state
-> run continuation evaluation
-> emit logical role artifacts
```

The sequence is intentionally asymmetric:

```text
reachability first
continuation second
```

RSI must never be used to establish reachability.

---

## Raw/control Soft Tube lifecycle

Historical Soft Tubes remain immutable exact replay artifacts.

```text
raw/control Tube
  replay / Tube-RSI / provenance
  may contain historical RSI-only or late-recovery rows

causal Jump Capability Tube
  natural-start forward reachable
  continuation positive
  physical-resolution cell support
```

`JIT/src/jit_dvgc/iterative_tube.py` retains source core exactly for historical reproducibility. For future causal TRAIN expansion it must:

- verify causal acquisition identity/provenance;
- reject expansion that lacks natural-start-connected evidence;
- copy ground-reachability provenance into new expansion rows.

Do not mutate historical core rows to pretend they were causal.

---

## Current decision architecture

```text
locked successful natural-start centerline
        ↓
pre-outcome every-x causal frontier plan
        ↓
causal TRAIN/CALIBRATION/ACCEPTANCE acquisition
        ↓
reachability provenance
        ↓
continuation evaluation
        ↓
causal capability analysis
        ↓
require new TRAIN causal root cells > 0
        ↓
C^k
        ↓
raw/control Tube_(k+1) with provenance on new causal rows
        ↓
policy training
        ↓
locked evaluation
        ↓
capability progression + policy realization
        ↓
selection or STOP
```

`JIT/cli/prepare_iterative_envelope_workflow.py` prepares the prospective causal workflow around a locked centerline.

---

## Modify-first rule

Before creating a production file:

1. Can an existing implementation be extended safely?
2. Can an existing package API expose the capability?
3. Can an existing CLI accept the new schema/config?
4. Can an existing regression test cover it?
5. Is this genuinely a durable scientific/runtime capability?

New modules are justified only when they represent a durable method distinction, such as causal reachability versus continuation evaluation.

Do not create `pi3_*.py`, `tube3_*.py`, retry-number production modules, or experiment-specific package branches.

---

## Active versus historical files

Current authority code/docs describe the causal method.

Older trajectory-centered-but-noncausal reports may remain only as explicitly superseded historical records. They must not appear ahead of the causal report in any authority read order.

Git history is the source archive for removed code; do not create obsolete Python archive folders.

---

## Compatibility and deletion gate

Before deleting any Python/CLI/test file:

1. prove no production import, package API, current CLI, current test, artifact loader, config or frozen reproducibility path depends on it;
2. run `python -m compileall -q JIT/src JIT/cli` plus targeted tests/imports;
3. stop if dependency closure is uncertain.

Never delete historical run artifacts merely because their scientific interpretation changed.

---

## Iteration-generic requirement

Reusable code must obtain iteration identity from artifacts/protocols rather than hard-coded current policy numbers or row counts.

Current task/method constants such as `2.5 m`, `4.2 m`, `0.1 m` x spacing and the initial causal look-back family belong to declared method contracts. Changing them requires an explicit method revision, not an experiment-specific hidden patch.

---

## CLI rules

`JIT/cli/` files should:

- parse arguments;
- call reusable production logic;
- print/write machine-readable results;
- avoid embedding fitting algorithms or physics logic;
- fail loudly on provenance or method-contract drift.

---

## Regression-test focus

Causal-method tests should cover at minimum:

```text
centerline uses natural-start real frames only
centerline does not interpolate qpos/qvel
centerline stores ground-connected state identity
proposal anchors cannot be used as physical resets
causal acquisition starts from natural reset
reachability provenance forbids RSI/qpos injection
non-descending or post-contact downstream candidates are rejected
role acquisition precedes continuation labeling
causal capability analyzer joins matching acquisition/labels only
raw Tube causal expansion requires verified provenance
workflow requires causal capability growth before new policy training
```

Current targeted tests include the existing centerline/frontier/capability/workflow suites plus causal reachability contract coverage.

---

## Workflow boundary

Orchestration may:

- sequence declared commands;
- validate JSON/file assertions;
- persist/resume state;
- stop when a scientific artifact fails.

It may not:

- fabricate a successful centerline;
- use RSI to claim reachability;
- edit qpos/qvel to populate capability cells;
- tune proposal families/replay/PPO/physics after seeing outcomes without a new decision;
- turn historical RSI evidence into causal evidence;
- touch final TEST data during development.
