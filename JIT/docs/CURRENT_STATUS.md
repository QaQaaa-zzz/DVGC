# Current JIT status — 2026-09-04

## Superseding live status — fixed-jump-start family landing round

The active round has completed acquisition, three-policy landing labels, role
separation, physical-cell analysis, and TRAIN-only replay construction.

```text
centerline: pi_0 real frames, 14 slices/cells, x=2.5..3.8 m
proposal acquisition: pi_0 from the fixed jump start, env.step only
label: first valid landing under any of pi_0/pi_1/pi_2; no recovery requirement
TRAIN:       525/527 positive; 382 unique positive root-geometry cells
CALIBRATION: 181/184 positive; 134 unique positive root-geometry cells
ACCEPTANCE:  181/184 positive; 135 unique positive root-geometry cells
replay Tube: 3119 retained + 525 TRAIN positives = 3644 rows
```

No new policy has been trained.  The old class-balanced C^1 fitting step is
intentionally bypassed because the family labels contain only two TRAIN
negatives and zero downstream negatives.  The next decision is the identity
and warm start of the new unified Actor; final TEST/JCE/JEL remains untouched.

Natural-start, `pi_1` centerline, stable-recovery, and “artifact not yet
generated” statements below are superseded history.

## Executive state

The historical engineering chain through `pi_2` is complete. The project is **paused before any pi_3-like training** because two previous interpretations were too weak for a publishable jump-capability claim:

1. raw/replay Tube growth was being confused with independent physical capability growth;
2. RSI continuation success was being confused with forward reachability from the real ground start.

The active method is now:

> **causal, trajectory-centered, resolution-aware Jump Capability Tube identification with just-in-time curriculum generation.**

The central empirical capability object is

```text
J_k = R_k^forward ∩ V_k^continuation
```

A state is capability evidence only if the robot first reaches it from the natural ground reset by real dynamics and then the exact reached state passes continuation evaluation.

RSI is retained for continuation evaluation and training, but can never establish reachability.

---

## Current chain

```text
pi_up_star + pi_down_star
  -> raw Tube_0 / pi_0 / C^0
  -> raw Tube_1 / pi_1 repair02
  -> C^1 engineering path
  -> raw Tube_2 / pi_2
  -> locked pi_1 vs pi_2 evaluation
  -> physical resolution analysis
  -> trajectory-centered redesign
  -> causal reachability redesign
  -> CURRENT: local verification + first real causal centerline/frontier artifacts
```

Do not start pi_3, do not run replay-ratio sweeps, and do not touch final TEST/JCE/JEL.

---

## Scientific objects

```text
F*    conceptual physical/task feasibility; not proved
R_k   natural-start-connected forward-reachability evidence
V_k   continuation-viability evidence
J_k   causal Jump Capability Tube = R_k ∩ V_k
S_k   raw/control Soft Tube used for replay/RSI
P_k   realization coverage of one unified Actor
```

The runtime target remains one unified Actor with no expert switching.

A later policy failing on an old state does not erase prior capability evidence. Conversely, a continuation-positive RSI state is not capability evidence unless it has ground-connected forward provenance.

---

## Nominal centerline v2

One successful natural-start full-chain rollout defines the fixed geometric scaffold used across causal iterations.

```text
x nominal start = 2.5 m
x hard maximum  = 4.2 m
x spacing       = 0.1 m
actual end      = first valid landing if earlier
```

Rules:

```text
real captured simulator frames only
no qpos/qvel interpolation
physical-state SHA per point
real transition count from natural start per point
centerline locked across iterations
```

Branch semantics:

```text
pre-Apex                      upstream
Apex neighbourhood            handoff marker
post-Apex + root_vz < 0       downstream
first valid landing            Jump-Tube terminal
post-contact / late recovery   excluded from capability frontier
```

No goal/intent variable is added. Reward and Actor observation remain unchanged.

Implementation:

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

Schema: `jit_nominal_jump_centerline_v2`.

---

## Physical resolution contract

```text
root x/y/z                      0.10 m
root vx/vy/vz                   0.10 m/s
roll/pitch/yaw                  0.50 deg
root angular velocity wx/wy/wz  2.0 deg/s
steering/hip/knee angle         0.50 deg
steering/hip/knee rate          2.0 deg/s
wheel tangential speed          0.10 m/s
phase                           discrete
```

Profiles:

- `root_geometry_v1`: primary macroscopic capability geometry;
- `full_physical_v1`: finer physical-state diversity.

Actor FIFO/history/last-action fields do not define capability cells.

---

## Why the old downstream cloud is not causal Jump Capability

Historical resolution-aware plots showed a large downstream-labelled cloud extending into late x positions such as around 4.5 m. That happened because old phase semantics remained downstream after Apex through landing/recovery, and the old frontier definition did not require negative vertical velocity or stop capability accounting at landing.

Those rows may still be useful replay states. They are not valid Jump-Capability frontier evidence.

The old semantic `Jump-Tube view` remains a retrospective diagnostic only. It filters corridor/descending/post-landing states but does **not** prove natural-start reachability.

---

## Causal forward acquisition

The active acquisition rule is:

```text
natural ground reset
        ↓
frozen-policy prefix
        ↓
enter predeclared look-back window before target x
        ↓
apply bounded action perturbation
        ↓
advance only through env.step
        ↓
capture a semantically valid state in the target x slice
```

Initial look-back family:

```text
0.1 m
0.2 m
0.3 m
```

Every accepted candidate carries `jit_ground_reachability_provenance_v1` and must declare:

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

For downstream causal acquisition, the candidate must remain post-Apex, descending (`root_vz < 0`) and pre-contact.

---

## Every-x frontier plan

The old global-lowest-score newest-shell selector is retired for prospective causal JIT.

Every usable 0.1 m centerline slice receives pre-outcome proposal families:

```text
TRAIN
TRAIN
TRAIN
CALIBRATION
ACCEPTANCE
```

These are exploration identities only and are never used as physical reset states.

Implementation:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Active plan revision schema: `jit_causal_trajectory_frontier_plan_revision_v2`.

---

## Reachability must precede continuation

Correct order:

```text
forward acquisition from natural start
        ↓
lock reachability provenance
        ↓
restore exact already-reached state
        ↓
continuation label
```

Causal role runner:

```text
JIT/src/jit_dvgc/causal_frontier_protocol.py
JIT/cli/run_causal_jump_frontier_role.py
```

Logical roles:

- `TRAIN`: may fit continuation fields and contribute qualifying curriculum support;
- `CALIBRATION`: threshold calibration only;
- `ACCEPTANCE`: locked development comparison only;
- final TEST/JCE/JEL: untouched.

---

## Causal capability analyzer

Implementation:

```text
JIT/src/jit_dvgc/analysis/causal_jump_capability.py
JIT/cli/analyze_causal_jump_capability.py
```

Primary curriculum-capability support is:

```text
locked centerline cells
UNION
TRAIN-positive natural-start-connected causal cells
```

CALIBRATION and ACCEPTANCE positives are reported separately and never merged into TRAIN support.

This is the first artifact that should be used to make a causal Jump-Capability growth claim.

---

## Historical raw Tubes

```text
Tube_0 =   222 raw snapshots
Tube_1 = 3,119 raw snapshots
Tube_2 = 3,776 raw snapshots
```

These remain immutable replay/provenance artifacts.

Retrospective all-state physical occupancy:

```text
Tube_0 root/full = 100 / 112
Tube_1 root/full = 2142 / 2404
Tube_2 root/full = 2446 / 2871
```

These are control-support occupancy counts, not causal Jump-Capability sizes.

Future causal TRAIN expansion rows admitted into a raw Tube must carry verified ground-reachability provenance. Historical core rows remain unchanged for reproducibility.

---

## Current pi_2 evidence

Training completed at 10,009,600 transitions.

Locked source panel:

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
successful parent groups 3
baseline reproduction failures 0
```

Interpretation under the current method:

- the source-panel drop remains valid evidence of substantial upstream single-policy realization loss;
- the old `13/14` remains valid historical continuation/frontier evidence;
- `13/14` is **not** retroactively called natural-start-connected capability expansion because the old acquisition began from RSI Tube anchors.

`pi_2` is not the selected next authority.

---

## Current code state

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
causal every-x frontier plan                   IMPLEMENTED
causal role protocol                           IMPLEMENTED
causal capability analyzer                     IMPLEMENTED
raw-Tube causal provenance guard               IMPLEMENTED
causal workflow preparation                    IMPLEMENTED

local compile / targeted pytest                NEXT
real locked centerline artifact                NEXT
first causal TRAIN/CALIBRATION/ACCEPTANCE       NOT RUN
first measured causal Jump Capability growth   NOT RUN
next policy under causal method                 NOT AUTHORIZED
final TEST/JCE/JEL                              UNTOUCHED
```

---

## Automatic workflow vNext

Prospective causal DAG:

```text
locked centerline
-> source raw/control geometry diagnostic
-> retrospective semantic Jump view diagnostic
-> predeclare causal every-x frontier plan
-> causal TRAIN/CALIBRATION/ACCEPTANCE forward acquisition
-> continuation evaluation
-> causal capability analysis
-> REQUIRE new TRAIN causal root cells > 0
-> C^k
-> raw/control Tube_(k+1) with causal provenance on new rows
-> smoke/isolation
-> baseline lock
-> candidate train/freeze
-> locked evaluation
-> capability progression + policy realization
-> select or stop
```

The code is integrated for prospective use but no complete causal round has been executed yet. Do not claim end-to-end causal JIT experimental validation until a recorded run closes this DAG.

---

## Immediate next tasks

1. Pull current branch and run compile + targeted tests.
2. Materialize/verify one successful natural-start `pi_1` centerline v2.
3. Generate the first causal every-x frontier plan.
4. Inspect x-slice coverage and role/family partition before collecting outcomes.
5. Run causal TRAIN/CALIBRATION/ACCEPTANCE forward acquisition and continuation evaluation.
6. Build the first causal Jump Capability summary.
7. Inspect new root cells by x slice and phase.
8. Only then decide whether C^k/raw Tube construction and a new policy are justified.

No pi_3-like training before this review.

---

## Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- branch: `agent/two-phase-soft-tube`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA: `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee actuator range: +/-30 N m
- actions: `[steer, rear-wheel drive, hip, knee]`
- runtime expert switching: none
- final TEST/JCE/JEL: untouched

---

## Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAUSAL_REACHABLE_JUMP_TUBE_REPORT_20260904.md`
5. `PROJECT.md`
6. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
7. `JIT/docs/CODEX_HANDOFF_20260904.md`
8. `JIT/docs/CODE_ORGANIZATION.md`
9. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md` — historical evidence only
