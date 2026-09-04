# JIT agent and maintenance rules

## Superseding active contract — fixed jump start and policy-family landing

The active experiment is conditioned on the fixed ground jump start at
`root x = 2.5 m`, not on the historical natural reset.  Lock the real-frame
`pi_0` centerline; use `pi_0` for forward proposal acquisition; evaluate each
exact reached candidate under frozen `pi_0`, `pi_1`, and `pi_2`; label it
positive if any evaluator reaches the first valid landing before physical
failure.  Post-landing recovery is ignored.  TRAIN positives may expand replay
without fitting a class-balanced continuation classifier.  CALIBRATION and
ACCEPTANCE remain holdout-only.  Contradictory natural-start/`pi_1`/stable-
recovery requirements later in this file are historical and superseded.

## Scope and safety

- `JIT/` is the active implementation area.
- Work only on `agent/two-phase-soft-tube` unless explicitly authorized otherwise.
- Never reset, clean, stash, rebase, force-push, overwrite or reformat unrelated user work.
- Use `/home/qy/mujoco_playground/.venv/bin/python`.
- Fixed task: `assets/orange_bike_4kg_horizontal.xml`, SHA `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`, payload 2 kg, 50 Hz, hip/knee +/-30 N m, actions `[steer, rear-wheel drive, hip, knee]`.
- Unified runtime never switches experts.
- Final TEST/JCE/JEL remains untouched until method/stopping/final policy are frozen.

## Active scientific contract — causal reachable Jump Tube

The paper/mainline object is no longer “all continuation-successful RSI states.”

A state counts as empirical Jump Capability evidence only if:

```text
1. it was reached from the natural ground reset by real env.step dynamics;
2. continuation from that exact reached state then succeeds.
```

Use:

```text
J_k = R_k^forward ∩ V_k^continuation
```

with the explicit caveat that both are empirical/proposal-family-conditioned. Do not claim formal reachability, a viability kernel, certified safety, invariance, or the complete physical limit.

RSI is allowed for continuation evaluation and training. RSI is **not** allowed to establish forward reachability.

## Centerline v2

The centerline is one locked successful natural-start trajectory used only as longitudinal scaffold.

```text
x_min = 2.5 m
x_hard_max = 4.2 m
dx = 0.1 m
actual end = first valid landing if earlier
```

Requirements:

```text
real captured simulator frames only
no qpos/qvel interpolation
natural-start connected
physical-state SHA at every point
pre-Apex / Apex / post-Apex semantics
post-Apex downstream requires vz < 0
post-landing excluded
not recomputed every iteration
not an Actor intent
not a reward target
```

Implementation:

```text
jit_dvgc.analysis.nominal_jump_centerline
JIT/cli/build_nominal_jump_centerline.py
```

Legacy `jit_nominal_jump_centerline_v1` lacks causal hashes and must not drive prospective causal workflows.

## Causal forward acquisition

Implementation:

```text
jit_dvgc.acquisition.causal_jump
jit_dvgc.causal_frontier_protocol
JIT/cli/run_causal_jump_frontier_role.py
```

Prospective candidate generation:

```text
natural reset
-> frozen policy prefix
-> start bounded perturbation within declared spatial lookback
-> env.step only
-> first semantically valid state in target 0.1 m slice
```

Initial lookback family:

```text
0.1 / 0.2 / 0.3 m
```

Do not change these based on observed outcomes without a new pre-outcome protocol decision.

Every candidate must carry verified `jit_ground_reachability_provenance_v1`:

```text
natural_start_connected=true
generated_by_env_step_only=true
rsi_used_to_establish_reachability=false
qpos_qvel_injection_used=false
proposal_anchor_used_as_reset=false
```

For downstream causal candidates additionally require:

```text
post-Apex active phase
root_vz < 0
valid_contact_seen == false
```

## Every-slice proposal plan

Prospective planning implementation:

```text
jit_dvgc.acquisition.resolution_frontier
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The locked centerline is mandatory.

Every usable 0.1 m centerline slice gets five pre-outcome proposal families:

```text
TRAIN
TRAIN
TRAIN
CALIBRATION
ACCEPTANCE
```

Proposal anchors are geometric/role identities only:

```text
entry_index = -1
global_index = -1
proposal_anchor_is_physical_reset = false
```

Never restore these anchors as Tube states. The causal runner always begins from ground.

## Resolution-aware physical space

Actor observation is not capability metric space.

```text
root x/y/z                      0.10 m
root vx/vy/vz                   0.10 m/s
roll/pitch/yaw                  0.50 deg
root angular velocity           2.0 deg/s
steering/hip/knee angle         0.50 deg
steering/hip/knee rate          2.0 deg/s
wheel tangential speed          0.10 m/s
phase                           discrete
```

Profiles:

- `root_geometry_v1`: primary macroscopic capability geometry;
- `full_physical_v1`: fine state diversity.

Implementation:

```text
jit_dvgc.analysis.capability_tube
JIT/cli/analyze_capability_tube.py
```

## Three different Tube-like artifacts — do not mix them

### 1. Raw/Control Soft Tube

Exact replayable TRAIN support. Historical core may include RSI-only or late-recovery states. This is allowed for training/reproducibility but is not capability proof.

Historical counts:

```text
Tube_0 222
Tube_1 3119
Tube_2 3776
```

### 2. Semantic Jump-Tube view

`jit_dvgc.analysis.jump_tube_view` filters corridor/downstream semantics from historical physical Tube geometry. Useful diagnostic, but still not forward-reachability proof.

### 3. Causal Jump Capability evidence

Authoritative paper object:

```text
jit_dvgc.analysis.causal_jump_capability
JIT/cli/analyze_causal_jump_capability.py
```

It joins ground-connected acquisition with continuation labels.

Primary TRAIN curriculum capability:

```text
locked centerline cells UNION TRAIN-positive causal cells
```

CALIBRATION/ACCEPTANCE positives remain holdout evidence only.

## Raw Tube construction under causal roles

`jit_dvgc.iterative_tube.build_iterative_tube` remains backward-compatible with historical roles.

For a prospective causal TRAIN manifest it must:

- verify the acquisition catalog is `ground_connected_causal_rollout_v1`;
- verify reachability provenance self-hashes;
- reject RSI/qpos-injection/proposal-reset reachability;
- require TRAIN label and acquisition state identity to agree;
- copy `ground_reachability` provenance into every new expansion row.

Historical source/core rows are retained exactly and explicitly may contain noncausal RSI support. Raw Tube is therefore still not the causal capability set.

## Continuation models

- `V_up/V_down`: bootstrap expert-conditioned continuation evidence;
- `C_up^k/C_down^k`: frozen-policy-conditioned continuation evidence;
- neither estimates forward reachability;
- PPO critic is not a JIT continuation field.

Historical C^1:

```text
upstream 64x64 AUC 0.6903137789904502 -> original formal gate FAIL, engineering selection only
downstream 64x64 AUC 1.0 / recall 1.0 -> formal calibration PASS
```

Never rewrite the upstream result as formal PASS.

## Historical pi_2 evidence

```text
pi_1 source panel 3115/3119
pi_2 source panel 3002/3119
upstream 423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
old pi_1-negative challenge pi_2 13/14
```

Interpretation after causal correction:

- source panel = valid policy-realization evidence;
- 13/14 = historical continuation/frontier evidence only;
- 13/14 is not causal ground-connected capability expansion proof;
- pi_2 remains unselected.

## Data roles

- `TRAIN`: may fit continuation and provide qualifying causal replay expansion;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development comparison only;
- final TEST/JCE/JEL untouched.

Forward reachability must be established before RSI continuation labeling for all prospective causal roles.

## Automatic iteration status

`JIT/cli/prepare_iterative_envelope_workflow.py` now requires:

```text
--nominal-centerline
```

and optionally:

```text
--source-causal-summary
```

The first causal round uses the locked centerline as causal baseline and therefore omits `--source-causal-summary`.

Prospective DAG:

```text
source diagnostics
-> causal every-slice plan
-> ground-connected TRAIN/CAL/ACCEPT acquisition
-> RSI continuation labels
-> reachable∩viable capability artifact
-> require new TRAIN causal root cells > 0
-> C^k
-> raw Tube with causal provenance on new rows
-> smoke/isolation/baseline
-> unified candidate train/freeze
-> locked evaluation
-> capability/realization decision
-> select or STOP
```

Code is integrated but no complete prospective causal run has been demonstrated yet.

## Current position

```text
historical experts through pi_2                   DONE
all-state resolution analysis                     DONE
causal method definition                          DONE
centerline v2 implementation                      DONE
causal acquisition implementation                 DONE
causal role protocol                              DONE
causal capability analyzer                        DONE
causal Tube-entry provenance guard                DONE
local compile/tests                               PENDING operator
locked real centerline artifact                   PENDING
first causal frontier run                         NOT RUN
first causal capability summary                   NOT RUN
next policy under causal method                   NOT AUTHORIZED
```

## Repository policy

- modify/consolidate before adding run-specific files;
- durable scientific capabilities may get new modules;
- CLIs remain thin;
- preserve immutable historical artifacts;
- no unrelated cleanup;
- compile/test before deletion;
- never reinterpret a historical RSI result as if it had prospective causal provenance.

## Current read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md`
5. `PROJECT.md`
6. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
7. `JIT/docs/CODE_ORGANIZATION.md`
8. `JIT/docs/CODEX_HANDOFF_20260904.md`
9. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md` historical
