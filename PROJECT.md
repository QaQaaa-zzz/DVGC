# DVGC / JIT Project

## 1. Scientific objective

DVGC/JIT studies **causal, trajectory-centered, resolution-aware jump-capability discovery with just-in-time curriculum generation** for one fixed single-track two-wheeled robot.

The final deployment target is still one unified Actor. Phase experts and frozen intermediate policies are discovery/training instruments only; runtime expert switching is not part of the final system.

The central scientific distinction is now:

```text
Forward reachability:
Can the robot actually reach state s from the real natural ground start?

Continuation viability:
After state s has really been reached, can the frozen policy continue and complete the jump?
```

The empirical Jump Capability Tube is defined as

```text
J_k = R_k^forward ∩ V_k^continuation
```

where both sets are empirical and conditioned on the declared controller/perturbation family. JIT does not claim an exact reachable set, viability kernel, invariant set, certified safe set, or proof of the physical jump limit.

---

## 2. Why this definition is necessary

RSI is useful for continuation evaluation and policy curriculum, but RSI cannot establish forward reachability.

A state restored directly at high altitude may be easy to finish from even if the robot can never jump from the ground to that state. Therefore:

```text
RSI continuation success != demonstrated jump capability
```

Every future capability state must first have a natural-start-connected forward trajectory generated only by authoritative `env.step`. Only then may the exact reached state be restored for continuation evaluation.

This causal separation is the active paper and engineering mainline.

---

## 3. Main research objects

```text
F*    conceptual physical/task feasibility under fixed dynamics; not proved
R_k   empirical natural-start-connected forward-reachability evidence
V_k   empirical continuation-viability evidence
J_k   causal Jump Capability Tube = R_k ∩ V_k
S_k   raw/control Soft Tube used for replay/RSI curriculum
P_k   realization coverage of one unified policy over locked support
```

`S_k` and `J_k` are intentionally different. Historical replay rows may remain useful for control training even if they are not valid causal capability evidence.

---

## 4. Nominal trajectory: geometric scaffold, not controller intent

The current Actor remains target-free. No desired jump distance, desired apex, or trajectory target is appended to the observation, and reward semantics are unchanged.

One successful natural-start full-chain jump defines a fixed geometric centerline used to index capability cross-sections longitudinally.

Centerline v2 contract:

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x spacing       = 0.1 m
actual terminal = first valid landing if earlier
```

Rules:

- each point is a real captured simulator frame;
- no qpos/qvel interpolation;
- centerline is locked once as the cross-iteration coordinate scaffold;
- each point records its physical-state SHA and real transition count from the natural ground reset.

Branch semantics:

```text
pre-Apex                      upstream
Apex neighbourhood            handoff marker
post-Apex + root_vz < 0       downstream
first valid landing            Jump-Tube terminal
post-contact / late recovery   excluded from capability frontier
```

Implementation:

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

---

## 5. Physical capability coordinate system

Actor observation space is not the capability metric space. FIFO history, last action, acceleration history and validity bits matter to control but do not define physical-cell identity.

Resolution contract v1:

| Quantity | Resolution |
|---|---:|
| root x/y/z | 0.10 m |
| root vx/vy/vz | 0.10 m/s |
| roll/pitch/yaw | 0.50 deg |
| root angular velocity wx/wy/wz | 2.0 deg/s |
| steering/hip/knee angle | 0.50 deg |
| steering/hip/knee rate | 2.0 deg/s |
| wheel tangential speed | 0.10 m/s |
| phase | discrete |

Profiles:

```text
root_geometry_v1
  root pose + root linear/angular velocity
  primary macroscopic Tube geometry

full_physical_v1
  root geometry + joint pose/rates + wheel tangential speeds
  fine physical-state diversity
```

Implementation:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

---

## 6. Causal forward acquisition

The active frontier acquisition does not reset directly to a proposal state.

For a target longitudinal slice `x_i`, the method starts from the real natural reset, follows the frozen policy, enters a predeclared look-back window, perturbs admissible actions, and advances only through real dynamics until it reaches the target slice.

Initial look-back family:

```text
0.1 m
0.2 m
0.3 m
```

These are first engineering probe windows, not proven optimal constants.

Every accepted candidate stores `jit_ground_reachability_provenance_v1`, including:

```text
natural_start_state_sha256
proposal target x
look-back distance
actual perturbation-start x
perturbation-start state SHA
environment transitions before perturbation
perturbed transitions
total transitions from ground
proposal family / variant identity
```

Hard provenance conditions:

```text
natural_start_connected = true
generated_by_env_step_only = true
rsi_used_to_establish_reachability = false
qpos_qvel_injection_used = false
proposal_anchor_used_as_reset = false
```

Implementation:

```text
JIT/src/jit_dvgc/acquisition/causal_jump.py
```

---

## 7. Every 0.1 m slice is an exploration target

The old global-lowest-score newest-shell selector is retired for prospective causal JIT.

The centerline is divided into 0.1 m longitudinal slices. Each usable slice receives pre-outcome proposal families assigned to logical roles:

```text
TRAIN
TRAIN
TRAIN
CALIBRATION
ACCEPTANCE
```

These are proposal identities, not physical reset anchors. Candidate acquisition must begin from the natural ground reset.

This prevents a dense region from monopolizing the exploration budget and makes Tube widening interpretable as `x -> cross-section` growth.

Plan implementation:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Active plan revision schema: `jit_causal_trajectory_frontier_plan_revision_v2`.

---

## 8. Continuation evaluation comes after reachability

Correct order:

```text
natural-start forward acquisition
        ↓
reachability provenance locked
        ↓
restore the exact already-reached state
        ↓
continuation evaluation
```

Logical data roles remain separated:

- `TRAIN`: may fit continuation fields and contribute qualifying curriculum support;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development comparison only;
- final TEST/JCE/JEL: untouched until the method/stopping/final policy are frozen.

Causal role implementation:

```text
JIT/src/jit_dvgc/causal_frontier_protocol.py
JIT/cli/run_causal_jump_frontier_role.py
```

---

## 9. Causal Jump Capability artifact

The scientific capability artifact is built by joining forward-reachability evidence with continuation labels.

Implementation:

```text
JIT/src/jit_dvgc/analysis/causal_jump_capability.py
JIT/cli/analyze_causal_jump_capability.py
```

It verifies candidate identity, reachability provenance, exact snapshot identity, jump-phase semantics and continuation-label identity.

Primary curriculum-capability support:

```text
locked centerline cells
UNION
TRAIN-positive ground-connected causal cells
```

CALIBRATION and ACCEPTANCE positive cells are reported separately and never merged into TRAIN curriculum.

---

## 10. Raw/control Soft Tube versus causal Jump Capability Tube

Historical Soft Tubes remain immutable replay/provenance artifacts.

```text
Raw/Control Soft Tube
  exact restartable snapshots
  replay / Tube-RSI / training support
  historical rows may be RSI-only or late recovery
  not capability proof

Causal Jump Capability Tube
  natural-start forward reachable
  continuation positive
  physically resolution-aware
  primary scientific capability object
```

For future causal TRAIN roles, `build_iterative_tube.py` verifies the causal acquisition catalog and copies ground-reachability provenance into admitted new expansion rows.

---

## 11. Historical engineering chain

Completed:

```text
pi_up_star + pi_down_star
-> bootstrap V_up / V_down
-> raw Tube_0
-> pi_0
-> C^0
-> raw Tube_1
-> pi_1 repair02
-> C^1 engineering path
-> raw Tube_2
-> pi_2
-> locked pi_1 vs pi_2 evaluation
-> physical-resolution analysis
```

Key identities/evidence:

```text
pi_up_star
  9,977,856 transitions
  actor f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081

pi_down_star
  25,600 transitions
  actor 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be

raw Tube_0 = 222
raw Tube_1 = 3,119
raw Tube_2 = 3,776
```

Retrospective all-state physical occupancy:

```text
Tube_0 root/full = 100 / 112
Tube_1 root/full = 2142 / 2404
Tube_2 root/full = 2446 / 2871
```

These are control-support occupancy counts, not causal capability sizes.

Historical pi_2 locked source-panel result:

```text
pi_1 3115/3119
pi_2 3002/3119
upstream   423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
```

Historical pi_1-negative challenge:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
3 successful parent groups
0 baseline reproduction failures
```

The `13/14` result is valid historical continuation/frontier evidence under the old RSI-anchored protocol. It is **not** retroactively reclassified as natural-start-connected Jump Capability expansion.

`pi_2` also demonstrated substantial upstream single-policy realization loss and is not the selected authority.

---

## 12. Current project position

```text
historical experts / Tube_0 / pi_0 / C^0       DONE
Tube_1 / pi_1                                  DONE
C^1 engineering path                           DONE
Tube_2 / pi_2                                  DONE
locked pi_1 vs pi_2                            DONE
physical resolution analysis                   DONE
causal capability definition                   IMPLEMENTED
centerline v2                                  IMPLEMENTED
causal forward acquisition                     IMPLEMENTED
causal role protocol                           IMPLEMENTED
causal capability analyzer                     IMPLEMENTED
raw-Tube causal provenance guard               IMPLEMENTED
causal automatic workflow preparation          IMPLEMENTED

local compile / targeted pytest                NEXT
real locked centerline artifact                NEXT
first prospective causal frontier              NOT RUN
first measured causal J_1 beyond centerline    NOT RUN
next unified policy under causal method         NOT AUTHORIZED YET
final TEST/JCE/JEL                              UNTOUCHED
```

No pi_3-like training should begin before the causal infrastructure passes local tests and the first causal capability frontier is reviewed.

---

## 13. Prospective causal JIT loop

```text
LOCKED successful natural-start centerline
        ↓
predeclare every-x causal frontier plan
        ↓
TRAIN / CALIBRATION / ACCEPTANCE
natural-start forward acquisition only
        ↓
lock reachability provenance
        ↓
RSI continuation evaluation
        ↓
causal capability analysis: Reachable ∩ Viable
        ↓
require new TRAIN causal root cells > 0
        ↓
fit/calibrate C^k
        ↓
build raw/control Tube_(k+1)
with causal provenance on new rows
        ↓
train one unified candidate
        ↓
locked evaluation
        ↓
separate:
  capability progression
  single-policy realization
        ↓
SELECT or STOP
```

The curriculum is generated from newly demonstrated causal frontier support, not from a hand-designed easy-to-hard schedule.

---

## 14. Paper narrative

The paper should not be sold as “RSI training for jumping.” The stronger story is:

> JIT iteratively identifies a ground-connected empirical jumping capability tube by intersecting forward-reachable states with continuation-viable states, then converts newly demonstrated frontier states into just-in-time curriculum for a single deployable policy.

Core contributions:

1. **causal capability definition**: forward reachable AND continuation viable;
2. **trajectory-centered physical parameterization**: 0.1 m longitudinal slices with high-dimensional cross-sections;
3. **causal frontier acquisition**: new states must be physically reached from the real start, not synthesized by RSI;
4. **capability vs realization separation**: cumulative demonstrated capability and one-policy coverage are reported separately;
5. **closed-loop JIT curriculum**: newly demonstrated TRAIN frontier support becomes the next training distribution.

Required experimental evidence before publication claims:

- prospective causal rounds, not retrospective relabeling only;
- per-x Tube cross-section growth;
- reachability provenance audits;
- ablations against RSI-only/global-frontier variants;
- policy-realization curves;
- stopping behavior;
- final untouched TEST/JCE/JEL after method freeze.

---

## 15. Immediate next operator sequence

1. Pull the causal branch and run compile + targeted tests.
2. Produce/lock one successful natural-start `pi_1` centerline v2 if not already materialized.
3. Generate the first causal every-x frontier plan.
4. Inspect plan coverage before any outcome collection.
5. Run causal TRAIN/CALIBRATION/ACCEPTANCE forward acquisition and continuation evaluation.
6. Build the first causal Jump Capability summary.
7. Inspect new cells by x slice and phase.
8. Only if causal TRAIN expansion is real and role isolation is clean, proceed to C^k / raw Tube construction.
9. Do not train the next policy until this evidence is accepted.

---

## 16. Immutable task contract

- repository: `QaQaaa-zzz/DVGC`
- branch: `agent/two-phase-soft-tube`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA-256: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee actuator range: +/-30 N m
- actions: `[steer, rear-wheel drive, hip, knee]`
- runtime: one unified Actor, no expert switching
- no silent XML/physics/reward/action/task-geometry changes
- final TEST/JCE/JEL untouched

---

## 17. Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md`
5. `PROJECT.md`
6. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
7. `JIT/docs/CODEX_HANDOFF_20260904.md`
8. `JIT/docs/CODE_ORGANIZATION.md`
9. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md` — historical evidence only
