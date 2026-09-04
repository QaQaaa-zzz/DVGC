# DVGC Repository Instructions

## Current research truth — 2026-09-04

### Superseding fixed-jump-start policy-family landing contract

The user-approved active experiment now conditions reachability on the fixed
ground jump start at `root x = 2.5 m`; a natural-start connection is not
required.  The one locked scaffold is the real-frame `pi_0` trajectory.  `pi_0`
also generates every acquisition candidate through authoritative `env.step`
with bounded lookback perturbations.  Each exact reached candidate is restored
only for evaluation under the frozen `{pi_0, pi_1, pi_2}` family.  A candidate
is positive when **any** family member reaches the first valid landing before
physical failure.  Post-landing recovery is outside the label.

Thus the active object is empirical, conditional jump-start landing capability:

```text
J_family = R_jump-start,pi0^forward
           INTERSECT (L_pi0 OR L_pi1 OR L_pi2)
```

TRAIN positives may enter replay directly; class balance and a fitted binary
continuation field are not prerequisites.  CALIBRATION and ACCEPTANCE remain
disjoint holdout evidence and never enter replay.  Any natural-start, `pi_1`
centerline, or stable-recovery requirements below are superseded historical
text, not the active experiment contract.

DVGC/JIT now studies **causal, trajectory-centered, resolution-aware Jump-Capability-Tube identification with just-in-time curriculum generation** for one fixed single-track two-wheeled robot task.

The central correction is mandatory:

```text
RSI continuation success != forward reachability from the ground.
```

The active empirical capability object is:

```text
J_k = R_k^forward ∩ V_k^continuation
```

where:

- `R_k^forward`: states actually reached from the locked natural ground reset using real `env.step` dynamics under the declared proposal-controller family;
- `V_k^continuation`: exact reached states that then complete the task under the declared frozen continuation policy/protocol.

JIT does **not** claim a formal reachability set, viability kernel, certified safe set, invariant set, or proof of the physical jump limit.

The final runtime target remains **one unified Actor**. Experts and frozen intermediate policies are discovery/control probes only; runtime expert switching is forbidden.

## Active scientific chain

Historical engineering chain:

```text
pi_up_star + pi_down_star
-> raw Tube_0 -> pi_0 -> C^0
-> raw Tube_1 -> pi_1 repair02
-> C^1 engineering path
-> raw Tube_2 -> pi_2
-> locked pi_1 vs pi_2
```

Active causal method:

```text
one successful natural-start jump
-> LOCK one real-frame centerline
-> every 0.1 m x slice is a declared exploration target
-> start each acquisition attempt from natural ground reset
-> policy prefix + bounded causal lookback action perturbation
-> reach candidate only through env.step
-> lock reachability provenance
-> RSI continuation evaluation of the already-reached candidate
-> reachable AND continuation-positive cells form causal Jump capability evidence
-> TRAIN-only causal positives may extend replay/curriculum support
-> train/evaluate one unified policy
-> repeat
```

Do not start a pi_3-like training run until the new causal acquisition has been locally compiled/tested, a locked centerline artifact exists, and a first causal discovery round produces meaningful new TRAIN-positive physical cells.

Final TEST/JCE/JEL remains untouched.

## Centerline contract v2

The centerline is a method scaffold, **not** an Actor intent, reward target, or reference-tracking command.

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x spacing       = 0.1 m
actual end      = first valid landing if earlier
```

Requirements:

```text
successful natural-start full-chain rollout
real captured simulator frames only
no qpos/qvel interpolation
natural-start connected
one physical-state SHA per centerline point
post-Apex downstream points require root_vz < 0
post-landing recovery excluded
centerline locked once; not recomputed each iteration
```

Implementation:

- `JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py`
- `JIT/cli/build_nominal_jump_centerline.py`

## Causal reachability contract

A prospective Jump capability candidate may not be created by restoring a Tube/RSI state and calling that restoration reachable.

Allowed reachability evidence:

```text
natural ground reset
-> deterministic frozen policy prefix
-> bounded predeclared action perturbation inside a causal lookback window
-> authoritative env.step only
-> exact candidate physically enters target x slice
```

Initial lookback family:

```text
0.1 m, 0.2 m, 0.3 m
```

Hard provenance flags:

```text
natural_start_connected = true
generated_by_env_step_only = true
rsi_used_to_establish_reachability = false
qpos_qvel_injection_used = false
proposal_anchor_used_as_reset = false
```

Implementation:

- `JIT/src/jit_dvgc/acquisition/causal_jump.py`
- `JIT/src/jit_dvgc/causal_frontier_protocol.py`
- `JIT/cli/run_causal_jump_frontier_role.py`

RSI is permitted only **after** forward reachability has been established, to evaluate continuation from the exact reached state.

## Every-slice frontier plan

The old global lowest-score/newest-shell reset-anchor frontier is historical only.

Prospective planning uses the locked centerline as the x scaffold. Every usable 0.1 m slice receives five disjoint pre-outcome proposal families:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE
```

These proposal anchors are identifiers only and must have no valid Tube reset index.

Implementation:

- `JIT/src/jit_dvgc/acquisition/resolution_frontier.py`
- `JIT/cli/prepare_resolution_aware_frontier_plan.py`

## Physical resolution contract v1

Actor observation space is not capability metric space.

| Quantity | Resolution |
|---|---:|
| root x/y/z | 0.10 m |
| root vx/vy/vz | 0.10 m/s |
| roll/pitch/yaw | 0.50 deg |
| root angular velocity | 2.0 deg/s |
| steering/hip/knee angle | 0.50 deg |
| steering/hip/knee rate | 2.0 deg/s |
| wheel tangential speed | 0.10 m/s |
| phase | discrete |

Profiles:

- `root_geometry_v1`: primary macroscopic capability geometry;
- `full_physical_v1`: fine physical-state diversity.

Implementation:

- `JIT/src/jit_dvgc/analysis/capability_tube.py`
- `JIT/cli/analyze_capability_tube.py`

## Raw/Control Tube versus causal Jump Capability Tube

Historical raw Soft Tubes are immutable training/replay artifacts:

```text
Tube_0 raw = 222
Tube_1 raw = 3119
Tube_2 raw = 3776
```

Retrospective all-state physical occupancy:

```text
Tube_0 root/full = 100 / 112
Tube_1 root/full = 2142 / 2404
Tube_2 root/full = 2446 / 2871
```

These are **not causal Jump-Capability counts**. Historical raw Tube states may include RSI-only states and late recovery.

The actual scientific capability artifact is built from ground-connected acquisition + continuation labels:

- `JIT/src/jit_dvgc/analysis/causal_jump_capability.py`
- `JIT/cli/analyze_causal_jump_capability.py`

Primary TRAIN curriculum capability:

```text
locked centerline root cells
UNION
TRAIN-positive ground-connected causal root cells
```

CALIBRATION/ACCEPTANCE cells remain holdout evidence and never enter TRAIN support.

The semantic `jump_tube_view` remains useful for diagnosing historical Tube contamination, but it is not reachability proof.

## Historical pi_2 claim boundary

Historical locked result:

```text
source panel:
pi_1 3115/3119
pi_2 3002/3119
upstream 423/427 -> 312/427
downstream 2692/2692 -> 2690/2692

old pi_1-negative challenge:
pi_2 13/14
upstream 4/5
downstream 9/9
3 successful parent groups
0 baseline reproduction failures
```

Current interpretation:

- the source-panel result is valid policy-realization evidence;
- `13/14` is valid **historical continuation/frontier success evidence**;
- it is **not** proof of ground-connected causal Jump-Capability expansion because the old challenge originated from the RSI-anchor frontier method;
- `pi_2` remains unselected as next authority.

## Continuation fields

- `V_up/V_down`: bootstrap expert-conditioned continuation evidence;
- `C_up^k/C_down^k`: frozen-policy-conditioned continuation evidence;
- they do not estimate forward reachability;
- PPO critic is not a JIT continuation field.

Current C^1 historical truth:

```text
upstream 64x64 AUC 0.6903137789904502 < 0.70 formal gate -> engineering-selected only
downstream 64x64 AUC 1.0 / recall 1.0 -> formal calibration PASS
```

Do not rewrite upstream as a formal pass.

## Data roles

- `TRAIN`: may fit continuation and contribute qualifying causal replay expansion;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development comparison only;
- final TEST/JCE/JEL: untouched.

For prospective causal roles, forward reachability must be established before RSI labeling.

## Automatic workflow status

`JIT/cli/prepare_iterative_envelope_workflow.py` now accepts a **locked causal centerline**, not a fresh centerline generated each iteration.

The prospective DAG includes:

```text
source diagnostics
-> causal every-slice plan
-> causal TRAIN/CAL/ACCEPT forward acquisition
-> RSI continuation labels
-> causal reachable∩viable analysis
-> require new causal TRAIN root cells > 0
-> C^k
-> raw Tube with causal provenance on new rows
-> smoke/isolation/baseline
-> candidate train/freeze
-> locked evaluation
-> capability progression + realization
-> selection or STOP
```

Code integration exists. A complete prospective causal run has **not yet been executed**, so do not claim demonstrated end-to-end causal automation.

## Immutable task identity

- branch: `agent/two-phase-soft-tube`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- simulation substep: 0.005 s
- control interval: 0.020 s = 50 Hz
- hip/knee actuator range: +/-30 N m
- action order: `[steer, rear-wheel drive, hip, knee]`
- runtime: one unified Actor, no expert switching
- no silent XML/physics/reward/action/task-geometry changes
- final TEST/JCE/JEL untouched

## Repository/Git safety

- preserve unrelated work;
- never reset, clean, stash, rebase or force-push;
- use `/home/qy/mujoco_playground/.venv/bin/python`;
- do not repeatedly recalculate SHA-256 values that are already locked in authority documents, manifests, or frozen artifacts; reuse those recorded identities during routine work and avoid redundant manual `sha256sum` checks;
- retain protocol-required automatic provenance/self-hash validation for newly generated artifacts and only perform an explicit manual hash calculation when diagnosing concrete identity drift;
- keep CLIs thin and scientific logic in `JIT/src/jit_dvgc/`;
- compile and run targeted tests after structural changes before cleanup;
- never turn a retrospective method correction into a fabricated historical PASS.

## Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md`
5. `PROJECT.md`
6. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
7. `JIT/docs/CODE_ORGANIZATION.md`
8. `JIT/docs/CODEX_HANDOFF_20260904.md`
9. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md` (historical)
10. superseded/intermediate Tube-redesign reports
