# JIT Causal Reachable Jump-Capability Iteration Protocol

## Status — 2026-09-04

This is the active protocol after the historical `pi_1 -> C^1 -> raw Tube_2 -> pi_2` round, physical Tube visualization review, and the subsequent causal-reachability correction.

The active method is:

> **causal, trajectory-centered, resolution-aware Jump-Capability-Tube identification with just-in-time curriculum generation.**

The key scientific rule is:

```text
RSI continuation success != forward reachability from the natural ground start
```

The empirical capability object is

```text
J_k = R_k^forward ∩ V_k^continuation
```

Final TEST/JCE/JEL remains untouched.

---

## 1. Scientific claim boundary

JIT distinguishes:

```text
F*    conceptual physical/task feasible set; not proved
R_k   empirical natural-start-connected forward-reachability evidence
V_k   empirical continuation-viability evidence
J_k   empirical causal Jump Capability Tube = R_k ∩ V_k
S_k   raw/control Soft Tube for replay and RSI
P_k   one-policy realization coverage
```

The method does not prove an exact reachable set, viability kernel, invariant set, certified safe set, or the physical maximum jump envelope.

All claims are empirical and conditioned on the declared policy/proposal family, action perturbations, horizon and fixed robot/task.

---

## 2. Immutable physical/task contract

- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- simulation substep: 0.005 s
- control interval: 0.020 s = 50 Hz
- hip/knee actuator range: +/-30 N m
- action order: `[steer, rear-wheel drive, hip, knee]`
- one unified runtime Actor
- no runtime expert switching
- no silent reward/action/XML/task-geometry changes
- no final TEST/JCE/JEL use during development

---

## 3. No goal intent in the current method

Do not add desired distance/apex intent to the Actor in the current mainline.

The nominal trajectory is a **method coordinate scaffold**, not a controller target or reward reference. At a fixed x slice, many physically different successful states may belong to the Tube.

Goal-conditioned or multi-intent JIT may be studied later only if prospective causal evidence shows that one target-free Actor cannot represent distinct real trajectory branches.

---

## 4. Locked centerline v2

Every causal study must lock one successful natural-start full-chain jump before frontier outcomes.

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x step          = 0.1 m
actual terminal = first valid landing if earlier
```

Construction rules:

```text
real captured simulator frames only
nearest real frame per target x slice
maximum x mismatch <= 0.05 m
no qpos/qvel interpolation
natural-start connected
physical-state SHA per point
real environment-transition count from ground per point
```

Branch semantics:

```text
pre-Apex                      upstream
Apex neighbourhood            handoff marker
post-Apex + root_vz < 0       downstream
first valid landing            Jump-Capability terminal
post-contact / late recovery   excluded from capability frontier
```

The centerline is locked across iterations. Recomputing it every iteration would move the coordinate frame and invalidate direct cross-iteration Tube comparisons.

Implementation:

```text
jit_dvgc.analysis.nominal_jump_centerline
JIT/cli/build_nominal_jump_centerline.py
```

Schema: `jit_nominal_jump_centerline_v2`.

---

## 5. Physical capability state and resolution

Actor observation is not the physical capability metric space.

Primary `root_geometry_v1` coordinates:

```text
root x/y/z
root vx/vy/vz
roll/pitch/yaw
root wx/wy/wz
phase
```

`full_physical_v1` adds:

```text
steering/hip/knee angles
steering/hip/knee rates
front/rear wheel tangential speeds
```

Resolution v1:

```text
position                     0.10 m
linear velocity              0.10 m/s
orientation                  0.50 deg
root angular velocity        2.0 deg/s
joint angle                  0.50 deg
joint angular velocity       2.0 deg/s
wheel tangential velocity    0.10 m/s
phase                        discrete
```

Implementation:

```text
jit_dvgc.analysis.capability_tube
JIT/cli/analyze_capability_tube.py
```

Raw snapshot count is not a capability-volume measure.

---

## 6. Causal frontier planning

The old global-lowest-score newest-shell reset-anchor frontier is historical only.

Prospective planning uses every usable 0.1 m centerline slice as an exploration target. Each slice receives five deterministic pre-outcome proposal families:

```text
TRAIN
TRAIN
TRAIN
CALIBRATION
ACCEPTANCE
```

These proposal anchors are identifiers, not physical states. They must not carry a valid Tube reset index and may never be used to establish reachability by RSI.

Implementation:

```text
jit_dvgc.acquisition.resolution_frontier
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Active revision schema: `jit_causal_trajectory_frontier_plan_revision_v2`.

Planning must occur before outcomes are observed.

---

## 7. Causal forward acquisition

Forward reachability is established only by a rollout connected to the natural ground reset.

Allowed sequence:

```text
natural ground reset
-> frozen-policy prefix
-> enter declared look-back window before target x
-> bounded declared action perturbation
-> authoritative env.step only
-> capture first semantically valid state in target slice
```

Initial look-back family:

```text
0.1 m
0.2 m
0.3 m
```

These values are an engineering probe family, not a theorem. Changing them after outcomes requires a new explicit pre-outcome method decision.

Forbidden as reachability evidence:

```text
Tube/RSI reset to proposal state
manual qpos/qvel editing
coordinate dilation
interpolated state injection
post-outcome target reselection
TEST-informed acquisition
```

Every accepted candidate must carry `jit_ground_reachability_provenance_v1` with hard flags:

```text
natural_start_connected = true
generated_by_env_step_only = true
rsi_used_to_establish_reachability = false
qpos_qvel_injection_used = false
proposal_anchor_used_as_reset = false
```

Implementation:

```text
jit_dvgc.acquisition.causal_jump
```

Downstream candidates must be post-Apex, `root_vz < 0`, and pre-contact.

---

## 8. Reachability before continuation

The order is mandatory:

```text
forward acquisition
        ↓
lock reachability provenance
        ↓
restore exact already-reached state
        ↓
continuation evaluation
```

RSI is allowed only at the continuation-evaluation stage and later for training curriculum.

A continuation-positive state with no natural-start-connected provenance is **not** causal Jump-Capability evidence.

Causal role implementation:

```text
jit_dvgc.causal_frontier_protocol
JIT/cli/run_causal_jump_frontier_role.py
```

---

## 9. Logical data roles

### TRAIN

May:

- fit `C_up^k/C_down^k`;
- contribute qualifying new causal replay support;
- appear in the primary curriculum-capability set after positive continuation evidence.

### CALIBRATION

May calibrate thresholds only. Positive causal states remain holdout evidence and must not enter TRAIN curriculum.

### ACCEPTANCE

Locked development comparison only. Positive causal states must not enter training.

### Final TEST/JCE/JEL

Untouched until the method, stopping rule and final policy are frozen.

Parent/proposal family isolation and deterministic role assignment remain mandatory.

---

## 10. Causal Jump Capability construction

Implementation:

```text
jit_dvgc.analysis.causal_jump_capability
JIT/cli/analyze_causal_jump_capability.py
```

The analyzer verifies:

```text
causal acquisition mode
ground-provenance self-hash
natural-start connection
env.step-only generation
exact candidate/snapshot identity
correct jump phase/downstream semantics
matching continuation label
```

Primary TRAIN curriculum capability:

```text
locked centerline cells
UNION
TRAIN-positive causal root cells
```

CALIBRATION/ACCEPTANCE positives are reported separately.

If a previous causal summary exists, cross-iteration growth is measured against the previous causal set. Historical noncausal Soft-Tube occupancy is never used as the causal baseline.

---

## 11. Raw/control Soft Tube construction

Raw Soft Tube remains a core-retaining replay/provenance artifact:

```text
S_(k+1)
= every S_k row retained exactly
+ qualifying logical-TRAIN replay snapshots
```

Future causal TRAIN expansion rows must pass the causal acquisition-catalog provenance check before admission. Ground-reachability provenance is copied into each new expansion row.

Historical core rows are retained for reproducibility and may include RSI-only/late-recovery support. Their presence must be declared and must not be converted into causal capability claims.

`build_iterative_tube.py` is responsible for this gate.

---

## 12. Continuation models

- `V_up/V_down`: bootstrap expert-conditioned continuation evidence;
- `C_up^k/C_down^k`: frozen-policy-conditioned continuation evidence;
- PPO critic is not a JIT continuation field;
- continuation models do not estimate forward reachability.

Current historical C^1 truth remains:

```text
upstream 64x64 AUC 0.6903137789904502 < 0.70 formal gate
  -> engineering-selected only

downstream 64x64 AUC 1.0 / recall 1.0
  -> formal calibration PASS
```

Do not rewrite upstream as formal PASS.

---

## 13. Policy training

The final deployment target remains one unified Actor.

Historical/current replay configuration:

```text
90% Tube reset
10% natural reset
inside Tube:
75% retained source support
25% newest expansion
```

This is an engineering configuration, not part of the scientific definition of reachability.

Do not automatically sweep replay ratios after candidate failure.

A new policy may be trained only after:

```text
causal roles completed
causal capability summary completed
new TRAIN causal root cells > 0
role isolation accepted
raw next Tube provenance checks pass
```

---

## 14. Candidate evaluation

Evaluation separates two questions.

### Capability progression

Did the causal `Reachable ∩ Viable` support expand?

Report at minimum:

```text
new causal root cells
new causal full cells
new cells by x slice
new cells by phase
holdout CALIBRATION/ACCEPTANCE positives separately
```

### Single-policy realization

How much locked prior support does one unified candidate still realize?

Policy realization is distinct from cumulative capability evidence. A later policy may lose realization of previously demonstrated states without erasing the historical capability evidence.

Do not use strict zero-regression as the sole scientific definition of capability progression.

---

## 15. Historical pi_2 interpretation

Locked source panel:

```text
pi_1 3115/3119
pi_2 3002/3119
upstream   423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
```

Old pi_1-negative challenge:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
3 successful parent groups
0 baseline reproduction failures
```

Current interpretation:

```text
source-panel loss -> valid single-policy realization degradation evidence
13/14 -> valid historical RSI-anchored continuation/frontier evidence
13/14 -> NOT natural-start-connected capability proof
pi_2 -> not selected as next authority
```

---

## 16. Automatic causal workflow

`JIT/cli/prepare_iterative_envelope_workflow.py` prepares the prospective causal DAG around a locked centerline.

Intended sequence:

```text
locked centerline
-> source diagnostics
-> causal every-x pre-outcome plan
-> causal TRAIN/CALIBRATION/ACCEPTANCE acquisition
-> continuation labels
-> causal capability analysis
-> REQUIRE new TRAIN causal root cells > 0
-> fit/calibrate C^k
-> raw Tube_(k+1) with causal provenance on new rows
-> smoke / role isolation / baseline lock
-> train / freeze candidate
-> locked evaluation
-> capability progression + policy realization
-> select or stop
```

Code integration exists. No complete prospective causal round has yet validated the entire DAG.

---

## 17. Stopping rule direction

Future stopping should be based on causal evidence, including:

- negligible new causal root/full cells;
- repeated inability to widen under-covered x slices;
- no meaningful new cross-section support;
- repeated proposal-family saturation;
- unacceptable single-policy realization loss;
- resource budget.

Only after the method and stopping rule are frozen may untouched final TEST/JCE/JEL be used.

---

## 18. Immediate operator sequence

```text
1. pull current branch
2. compile + targeted pytest
3. materialize/verify locked natural-start centerline v2
4. generate first causal every-x frontier plan
5. inspect plan before outcomes
6. run causal TRAIN/CALIBRATION/ACCEPTANCE
7. build first causal Jump Capability summary
8. inspect per-x expansion
9. only then decide whether to construct the next training Tube and train a new policy
```

No pi_3-like training before steps 1–8 are accepted.
