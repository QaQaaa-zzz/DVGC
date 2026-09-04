# DVGC/JIT Technical Handoff — 2026-09-04

## Superseding current handoff

The fixed-jump-start policy-family landing round is complete through training
and prospective selection.  Use the locked real-frame `pi_0` centerline and
`pi_0` proposal rollouts.  A reached state is positive if any frozen
`pi_0/pi_1/pi_2` evaluator reaches first valid landing; recovery is not required.
Results are TRAIN 525/527, CALIBRATION 181/184, ACCEPTANCE 181/184, 382 new
TRAIN root cells, and a 3644-row TRAIN replay Tube.  `pi_2_landing_replay` was
Actor-only warm-started from frozen `pi_1`, trained for exactly 10,009,600
transitions, retained 3119/3119 source-core landings, and landed 4/6 locked
baseline-negative ACCEPTANCE states across two parent groups.  It is selected
as the next engineering iteration authority.  Final TEST/JCE/JEL is untouched.
Older natural-start/`pi_1`-centerline instructions below are superseded.

## Purpose

This is the active takeover guide after the historical `pi_1 -> C^1 -> raw Tube_2 -> pi_2` engineering round, physical Tube visualization review, trajectory-centered redesign, and the later **causal reachability correction**.

Read first:

```text
AGENTS.md
JIT/AGENTS.md
JIT/docs/CURRENT_STATUS.md
JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md
PROJECT.md
```

Historical quantitative context:

```text
JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md
```

Older trajectory-centered-but-noncausal reports are superseded method history only.

---

## 1. Current project definition

JIT now studies a **causal empirical Jump Capability Tube** around one fixed successful jump trajectory.

The central rule is:

```text
RSI continuation success != forward reachability from the ground
```

The active empirical capability object is:

```text
J_k = R_k^forward ∩ V_k^continuation
```

where:

- `R_k^forward`: states actually reached from the natural ground reset through real dynamics under the declared proposal family;
- `V_k^continuation`: those exact reached states that then complete the remaining task under the frozen continuation protocol.

RSI is retained for continuation evaluation and training, but never for establishing reachability.

The final runtime target remains one unified Actor with no expert switching.

---

## 2. Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- branch: `agent/two-phase-soft-tube`
- local repo: `~/DVGC`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee: +/-30 N m
- actions: `[steer, rear-wheel drive, hip, knee]`
- runtime expert switching: none
- final TEST/JCE/JEL: untouched

Never silently change XML, physics, reward, task geometry, action semantics, or final evaluation data.

---

## 3. Historical completed engineering chain

```text
pi_up_star + pi_down_star
-> raw Tube_0 = 222
-> pi_0
-> C^0
-> raw Tube_1 = 3,119
-> pi_1 repair02
-> C^1 engineering path
-> raw Tube_2 = 3,776
-> pi_2
-> locked pi_1 vs pi_2 evaluation
```

Expert identities:

```text
pi_up_star
  9,977,856 transitions
  actor f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081

pi_down_star
  25,600 transitions
  actor 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

Historical raw Tube paths:

```text
Tube_0 JIT/runs/soft_tube/soft_tube_train_v1_20260828
Tube_1 JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
Tube_2 JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904
```

Historical raw/all-state geometry remains replay/control evidence, not causal capability proof.

---

## 4. Historical physical-resolution evidence

Measured all-state occupancy:

```text
Tube_0
  raw 222
  root cells 100
  full cells 112

Tube_1
  raw 3119
  root cells 2142
  full cells 2404

Tube_2
  raw 3776
  root cells 2446
  full cells 2871
```

These values showed two things:

1. raw snapshot cardinality overstates independent physical-state growth;
2. historical downstream-labelled support contains late recovery and therefore cannot be read directly as a jump envelope.

They remain useful retrospective diagnostics only.

---

## 5. Historical pi_2 interpretation

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
successful parent groups 3
baseline reproduction failures 0
```

Current interpretation:

- the source-panel result remains valid single-policy realization evidence;
- `13/14` remains valid historical continuation/frontier success evidence under the old RSI-anchor protocol;
- `13/14` is **not** natural-start-connected Jump Capability proof;
- `pi_2` is not selected as the next authority.

Do not rewrite history to make old candidates causal.

---

## 6. Current centerline contract v2

One successful natural-start full-chain rollout provides one locked cross-iteration geometric scaffold.

```text
x start 2.5 m
x hard max 4.2 m
dx 0.1 m
actual terminal first valid landing if earlier
```

Requirements:

```text
real trace frames only
no qpos/qvel interpolation
natural-start connected
physical-state SHA per point
transition count from ground per point
post-Apex downstream requires root_vz < 0
post-landing excluded
centerline locked once across iterations
```

Implementation:

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

Schema: `jit_nominal_jump_centerline_v2`.

The centerline is a method scaffold, not an Actor goal/intent.

---

## 7. Current physical resolution contract

```text
position 0.10 m
linear velocity 0.10 m/s
orientation 0.50 deg
root angular velocity 2.0 deg/s
joint angle 0.50 deg
joint rate 2.0 deg/s
wheel tangential speed 0.10 m/s
phase discrete
```

Primary macroscopic profile: `root_geometry_v1`.
Fine profile: `full_physical_v1`.

Implementation:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

---

## 8. New causal forward acquisition

Implementation:

```text
JIT/src/jit_dvgc/acquisition/causal_jump.py
```

Correct reachability sequence:

```text
natural ground reset
-> frozen-policy prefix
-> enter predeclared look-back window before target x
-> bounded action perturbation
-> env.step only
-> reach target slice physically
-> capture candidate
```

Initial look-back family:

```text
0.1 m
0.2 m
0.3 m
```

Every accepted state carries `jit_ground_reachability_provenance_v1`.

Mandatory flags:

```text
natural_start_connected = true
generated_by_env_step_only = true
rsi_used_to_establish_reachability = false
qpos_qvel_injection_used = false
proposal_anchor_used_as_reset = false
```

Downstream candidates must still be descending and pre-contact.

---

## 9. Every-x causal frontier plan

Implementation:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The old newest-shell Tube reset anchors are no longer the active frontier definition.

Every usable 0.1 m centerline slice receives five deterministic pre-outcome proposal families:

```text
TRAIN
TRAIN
TRAIN
CALIBRATION
ACCEPTANCE
```

These proposal anchors are identifiers only and may never be used as reset states.

Active revision schema: `jit_causal_trajectory_frontier_plan_revision_v2`.

---

## 10. Reachability before continuation

Causal role implementation:

```text
JIT/src/jit_dvgc/causal_frontier_protocol.py
JIT/cli/run_causal_jump_frontier_role.py
```

Mandatory order:

```text
forward acquisition from ground
-> verify reachability provenance
-> restore exact already-reached state
-> continuation evaluation
```

RSI is legal only at the continuation stage.

Logical roles remain TRAIN / CALIBRATION / ACCEPTANCE and final TEST remains untouched.

---

## 11. Primary causal capability artifact

Implementation:

```text
JIT/src/jit_dvgc/analysis/causal_jump_capability.py
JIT/cli/analyze_causal_jump_capability.py
```

Primary TRAIN capability support:

```text
locked centerline cells
UNION
TRAIN-positive natural-start-connected causal cells
```

CALIBRATION and ACCEPTANCE positive causal cells are reported separately and never merged into TRAIN curriculum.

If a previous causal summary is provided, progression is measured against that previous causal set. Historical noncausal Soft-Tube occupancy is never used as a causal baseline.

---

## 12. Raw/control Tube policy

Historical Soft Tubes are not deleted or rewritten.

For future causal TRAIN expansions, `JIT/src/jit_dvgc/iterative_tube.py` must verify the causal acquisition catalog and copy ground-reachability provenance into each admitted new replay row.

Historical core may remain RSI-only/recovery-heavy for reproducibility. It must not be described as causal capability.

---

## 13. Current code state

```text
historical experts / Tube_0 / pi_0 / C^0       DONE
Tube_1 / pi_1                                  DONE
C^1 engineering path                           DONE
Tube_2 / pi_2                                  DONE
locked pi_1 vs pi_2                            DONE
physical resolution analysis                   DONE
centerline v2                                  IMPLEMENTED
causal forward acquisition                     IMPLEMENTED
causal every-x planning                        IMPLEMENTED
causal role protocol                           IMPLEMENTED
causal capability analyzer                     IMPLEMENTED
raw-Tube causal provenance guard               IMPLEMENTED
causal workflow preparation                    IMPLEMENTED

local compile / targeted pytest                DONE for active landing round
real locked pi_0 centerline artifact            DONE
fixed-jump TRAIN/CALIBRATION/ACCEPTANCE          DONE
measured landing-capability growth              DONE: 382 TRAIN root cells
next unified policy                             SELECTED: pi_2_landing_replay
final TEST/JCE/JEL                              UNTOUCHED
```

---

## 14. First commands in a new conversation

```bash
cd ~/DVGC
git pull --ff-only origin agent/two-phase-soft-tube

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli

$PY -m pytest -q \
  JIT/tests/test_nominal_jump_centerline.py \
  JIT/tests/test_causal_jump_reachability_contract.py \
  JIT/tests/test_resolution_frontier_parent_selection.py \
  JIT/tests/test_jump_tube_view.py \
  JIT/tests/test_capability_tube_resolution.py \
  JIT/tests/test_capability_progression.py \
  JIT/tests/test_iteration_policy_selection_capability.py \
  JIT/tests/test_iterative_envelope_automation.py
```

If any test fails, stop and diagnose. Do not skip a failed causal-provenance test merely to reach training.

---

## 15. Immediate scientific task after tests

Do **not** train a new policy first.

1. Materialize or verify one successful natural-start `pi_1` centerline v2.
2. Generate the first every-x causal frontier plan.
3. Inspect slice coverage and role/family allocation before outcomes.
4. Run causal TRAIN/CALIBRATION/ACCEPTANCE forward acquisition.
5. Continue-label only the exact states that were physically reached.
6. Build the first causal `Reachable ∩ Viable` summary.
7. Inspect per-x new root/full cells.
8. Only then decide whether continuation fitting, raw Tube expansion and next-policy training are justified.

---

## 16. What not to do

Do not:

- train pi_3 before first causal capability evidence is reviewed;
- use RSI reset to establish reachability;
- edit qpos/qvel to populate a desired cell;
- treat proposal anchors as reset states;
- reinterpret historical Tube1/Tube2 occupancy as causal capability;
- reinterpret old pi_2 13/14 as ground-connected proof;
- add intent/reward changes in this method version;
- rewrite C_up^1 as formal PASS;
- run automatic replay-ratio repairs;
- touch final TEST/JCE/JEL.

---

## 17. Paper direction

Core statement:

> JIT iteratively identifies a ground-connected empirical jumping capability tube by intersecting forward-reachable states with continuation-viable states, and converts newly demonstrated TRAIN frontier states into just-in-time curriculum for a single deployable policy.

Required evidence before publication claims:

- prospective causal rounds;
- per-x Tube cross-section growth;
- provenance audits;
- RSI-only/global-frontier ablations;
- capability-vs-realization separation;
- method stopping evidence;
- untouched final TEST/JCE/JEL after freeze.
