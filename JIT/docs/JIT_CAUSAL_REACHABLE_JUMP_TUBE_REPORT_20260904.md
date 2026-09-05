# JIT Causal-Reachable Jump Capability Tube — Current Method and Task Report

> Historical redesign record. This document preserves the 2026-09-04
> natural-start causal proposal and is no longer current execution authority.
> The user-approved active experiment now conditions arrival on the fixed
> `x = 2.5 m` jump start and frozen π0 proposer. The subsequent π3 comparison
> also exposed a mixed success-endpoint problem. Read `AGENTS.md`,
> `JIT/docs/CURRENT_STATUS.md` and
> `JIT/docs/JIT_SCIENTIFIC_REVIEW_RESPONSE_20260905.md` for current truth. Any
> later wording in this report that calls the natural-start design or π3-like
> progression “current” is historical, not operational authority.

Date: 2026-09-04  
Branch: `agent/two-phase-soft-tube`

## Superseding experiment addendum

The executed round is conditional on the fixed ground jump start (`x=2.5 m`).
It locks the real-frame `pi_0` centerline, uses `pi_0` for forward proposal
acquisition, and ORs first-valid-landing outcomes from frozen
`pi_0/pi_1/pi_2`.  Post-landing recovery is excluded.  The completed evidence
is TRAIN 525/527, CALIBRATION 181/184, ACCEPTANCE 181/184, with 382 new TRAIN
root-geometry cells beyond the 14-cell centerline.  A TRAIN-only replay Tube of
3644 rows was built (3119 retained + 525 causal additions).  No new Actor has
been trained.  Conflicting natural-start or stable-recovery text later in this
report is superseded method history.

## 0. Executive conclusion

The project has completed the historical engineering chain from phase experts through `Tube_2` and `pi_2`. That work established a functioning two-phase-to-unified-policy pipeline, continuation-field machinery, Tube-RSI training, paired evaluation, and automated iteration infrastructure. It also exposed two methodological weaknesses in the old capability interpretation:

1. raw/replay Tube growth was being confused with independent physical-state-space growth;
2. RSI continuation success was being confused with forward reachability from the real ground start.

The second issue is decisive. A robot that succeeds after being restored directly into a high state has not demonstrated that it can jump from the ground into that state. Therefore the active JIT definition is now:

> **Empirical Jump Capability = ground-connected forward reachability AND continuation viability.**

In symbols, for the fixed robot/task and the declared proposal-controller family,

```text
J_k = R_k^forward ∩ V_k^continuation
```

where both sets are empirical and policy/proposal-family conditioned. JIT does **not** claim an exact reachability set, viability kernel, certified safe set, invariant set, or proof of the physical jump limit.

The active method is therefore:

> **causal, trajectory-centered, resolution-aware Jump-Capability-Tube identification with just-in-time curriculum generation.**

The final deployment target remains one unified Actor with no expert switching.

---

## 1. Core research question

For one fixed XML, actuator limits, task geometry and initial ground state, the physically interesting question is not merely:

> If the robot were already placed at state `s`, could the policy finish the jump?

It is:

> Can the robot actually reach `s` from the real start under admissible control, and from `s` can it still complete the jump?

Those are two different predicates:

```text
Forward reachability:
R(s) = state s was physically reached from the natural ground reset by real env.step dynamics.

Continuation viability:
V(s) = after the exact reached state s is restored, the frozen continuation policy completes the declared event chain.
```

The empirical Jump Capability Tube contains only states satisfying both.

RSI is therefore retained as an efficient **continuation-evaluation and training mechanism**, but RSI can never manufacture reachability evidence.

---

## 2. Why this makes JIT a publishable method rather than an RSI trick

The paper-worthy contribution is not “we restart RL from many states.” State initialization and curriculum ideas already exist broadly.

The stronger contribution is the closed loop:

```text
one real successful ground-to-landing trajectory
        ↓
lock a longitudinal 0.1 m centerline scaffold
        ↓
causally perturb earlier actions from the real ground start
        ↓
observe which new physical cells are actually reached
        ↓
RSI-evaluate continuation only after reachability is established
        ↓
construct empirical reachable ∩ viable Jump-Tube cells
        ↓
use TRAIN-only successful frontier evidence as just-in-time curriculum support
        ↓
train one unified policy
        ↓
repeat and measure cross-section expansion + policy realization
```

This separates three quantities that conventional reward-only reporting tends to conflate:

1. **physical/empirical reachability evidence**;
2. **continuation viability**;
3. **one-policy realization coverage**.

That separation is the central scientific narrative.

---

## 3. The objects JIT now distinguishes

### 3.1 Unknown physical/task feasibility `F*`

Conceptual set of states compatible with some admissible control sequence under the fixed robot/task. Current JIT does not compute or prove `F*`.

### 3.2 Ground-connected reachable evidence `R_k`

States actually produced by a forward rollout beginning at the locked natural ground reset and advancing only through authoritative dynamics.

`R_k` is empirical and conditioned on the controller/perturbation family used to explore it. Failure to reach a state is **not** proof that no controller could reach it.

### 3.3 Continuation viability evidence `V_k`

Exact reached states that successfully complete the remaining task under the declared frozen continuation policy and evaluation protocol.

### 3.4 Causal Jump Capability Tube `J_k`

```text
J_k = R_k ∩ V_k
```

The primary capability object.

### 3.5 Raw/Control Soft Tube

Exact restartable TRAIN snapshots used for replay/RSI. Historical core rows may contain RSI-only or late recovery states. They remain valid training/provenance artifacts but are not capability proof.

### 3.6 Single-policy realization `P_k`

How much of cumulative locked support a single unified Actor realizes. This is distinct from whether capability has ever been empirically demonstrated.

---

## 4. Nominal trajectory: scaffold, not intent

The current Actor remains target-free. No desired distance, desired apex or trajectory reference is appended to the observation. Reward semantics are unchanged.

The nominal trajectory is a **method reference** used to index the jump longitudinally.

Current centerline contract:

```text
nominal x start = 2.5 m
hard x maximum  = 4.2 m
x resolution    = 0.1 m
actual end      = first valid landing if earlier
```

Construction:

```text
successful natural-start full-chain rollout
-> saved real qpos/qvel frames
-> nearest real frame to x=2.5,2.6,...
-> no qpos/qvel interpolation
-> stop at first valid landing
```

Phase semantics:

```text
pre-Apex                     upstream
exact Apex neighborhood       upstream terminal marker / handoff
post-Apex + root_vz < 0       downstream
first valid landing           Jump-Tube terminal
post-contact / late recovery  excluded from capability frontier
```

The centerline is locked as one method reference and is not recomputed every iteration. Otherwise the coordinate system itself would drift with the policy and make cross-iteration Tube comparisons ambiguous.

Implementation:

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

Current schema: `jit_nominal_jump_centerline_v2`.

Each point records a physical-state SHA and its number of real environment transitions from the natural ground reset.

---

## 5. Physical state resolution

Actor observation is not used as the physical capability metric. FIFO history, last action and validity bits matter to control but should not cause two nearly identical physical robot states to count as independent capability regions.

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

Two profiles remain useful:

```text
root_geometry_v1
  root pose + root linear/angular velocity
  primary macroscopic Tube geometry

full_physical_v1
  root geometry + joints + wheel tangential speed
  fine physical-state diversity
```

Implementation:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

---

## 6. Why the historical downstream cloud was wrong as Jump-Capability evidence

Resolution-aware plots showed a large historical downstream-labelled cloud around late x positions such as approximately 4.5 m. The old phase state remained downstream after Apex through landing and recovery, and the old global frontier selector did not require negative vertical velocity or a jump-corridor terminal.

Therefore:

```text
active_phase == downstream
```

was not equivalent to:

```text
physically descending jump state
```

The historical all-state Soft Tube can still contain such states because a unified policy must learn recovery. But late recovery can no longer contribute to Jump-Capability expansion.

The intermediate semantic diagnostic remains available:

```text
JIT/src/jit_dvgc/analysis/jump_tube_view.py
JIT/cli/analyze_jump_tube_view.py
```

It filters x corridor / descending downstream / post-landing support, but it is **not a reachability proof**. It is a diagnostic view over historical raw Tube data.

---

## 7. The critical correction: causal forward acquisition

The old acquisition procedure restored a Tube state, perturbed actions for a short horizon, and then continuation-labeled the resulting state. Even though the local perturbation used real dynamics, the initial Tube state itself could have originated from RSI. Therefore the resulting state was not necessarily connected to the real ground start.

The new acquisition rule is stricter:

```text
natural ground reset
        ↓
frozen policy prefix
        ↓
enter a predeclared spatial lookback window before target x
        ↓
apply bounded action perturbation
        ↓
advance only by env.step
        ↓
capture first semantically valid state in target 0.1 m slice
```

For example, to test whether the robot can occupy a higher state around `x=3.2 m`, the method does **not** teleport to `x=3.2`. It can start perturbing around `x=2.9/3.0/3.1` and lets the robot physically arrive at the 3.2 m slice.

Initial causal lookback family:

```text
0.1 m
0.2 m
0.3 m
```

These are first engineering probe windows, not proven optimal values. Changing them after results are seen requires a new pre-outcome method decision.

Implementation:

```text
JIT/src/jit_dvgc/acquisition/causal_jump.py
```

Every accepted candidate carries `jit_ground_reachability_provenance_v1`, including:

```text
natural_start_state_sha256
proposal target x
lookback distance
actual perturbation-start x
perturbation-start state SHA
environment transitions before perturbation
perturbed transitions
total transitions from ground
proposal family
variant identity
```

Hard flags require:

```text
natural_start_connected = true
generated_by_env_step_only = true
rsi_used_to_establish_reachability = false
qpos_qvel_injection_used = false
proposal_anchor_used_as_reset = false
```

---

## 8. Every 0.1 m centerline slice is now a declared exploration target

The old selector could let one dense or low-score region dominate the frontier budget.

The active causal plan instead creates five pre-outcome proposal families for **every usable centerline slice**:

```text
TRAIN
TRAIN
TRAIN
CALIBRATION
ACCEPTANCE
```

Those anchors are slice/family identities only. They have sentinel Tube indices and may never be used as physical resets.

This design gives each longitudinal slice a chance to widen rather than allowing a dense late region to absorb the entire budget.

Implementation:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Current causal plan revision schema:

`jit_causal_trajectory_frontier_plan_revision_v2`.

The perturbation variants are deterministically partitioned across the five proposal families, so TRAIN/CALIBRATION/ACCEPTANCE do not simply duplicate the same perturbation recipe at the same slice.

---

## 9. Data roles and the reachability/continuation sequence

The correct order is now mandatory:

```text
forward acquisition from ground
        ↓
reachability provenance locked
        ↓
RSI restore the exact already-reached state
        ↓
continuation label
```

Never reverse those semantics.

Logical roles remain:

### TRAIN

May fit continuation models and contribute qualifying new replay support.

### CALIBRATION

Threshold calibration only. Positive cells are useful holdout capability evidence but are not merged into TRAIN curriculum.

### ACCEPTANCE

Locked development comparison only. Positive cells remain holdout evidence and never enter training.

### Final TEST/JCE/JEL

Untouched until the method, stopping rule and final policy are frozen.

Causal role runner:

```text
JIT/src/jit_dvgc/causal_frontier_protocol.py
JIT/cli/run_causal_jump_frontier_role.py
```

---

## 10. Causal capability evidence artifact

A new analysis explicitly joins forward-reachability evidence with continuation labels.

Implementation:

```text
JIT/src/jit_dvgc/analysis/causal_jump_capability.py
JIT/cli/analyze_causal_jump_capability.py
```

It verifies:

```text
catalog uses causal acquisition mode
ground provenance self-hashes correctly
RSI did not establish reachability
exact snapshot matches candidate SHA
downstream is still descending and pre-contact
continuation label belongs to same candidate
```

The primary curriculum-capability cell set is:

```text
locked centerline cells
UNION
TRAIN-positive ground-connected causal cells
```

CALIBRATION and ACCEPTANCE positive cells are reported separately and never merged into this TRAIN support.

The optional previous causal summary allows cross-iteration cell-growth accounting without using historical noncausal Tube cells as the baseline.

---

## 11. Raw replay Tube still has a role

The historical raw Soft Tube is not deleted. It contains useful exact restart states and is part of the engineering history.

The new distinction is:

```text
Raw/Control Soft Tube
  training/replay support
  may include historical RSI-only states and late recovery
  not capability proof

Causal Jump Capability Tube
  natural-start-forward-reachable
  continuation-positive
  physically resolved
  primary scientific capability object
```

For future causal TRAIN roles, `build_iterative_tube.py` now verifies the causal acquisition catalog before admitting new expansion rows and copies ground-reachability provenance into each new replay entry.

Historical core rows remain untouched for reproducibility; the manifest explicitly states that historical core may include noncausal RSI support.

---

## 12. Historical completed engineering chain

### Phase experts

```text
pi_up_star
  9,977,856 transitions
  actor f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081

pi_down_star
  25,600 transitions
  actor 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

Role in the current story: bootstrap capability/control probes, not final runtime controllers.

### Tube_0

```text
raw snapshots 222
upstream 117
downstream 105
```

Retrospective all-state resolution analysis:

```text
root cells 100
full cells 112
```

This is control/replay occupancy, not causal Jump-Capability size.

### pi_0

10,009,600 transitions. First unified-policy demonstration that the two phase-specific bootstrap supports can be consumed by one Actor.

### Tube_1

```text
raw snapshots 3119
root cells 2142
full cells 2404
new all-state root cells vs Tube_0 = 2042
```

This was a large physical-state-space expansion in the all-state control-support sense. It must not be called a 21x causal Jump-Capability expansion because forward ancestry was not required and late downstream/recovery was included.

### pi_1 repair02

Historical engineering authority.

Quickcheck:

```text
Tube_0 222/222
upstream 117/117
downstream 105/105
boundary 26/260 across 4 parent groups
```

Historical formal Iteration-1 PASS remains unclaimed because three old baseline-reproduction mismatches remain quarantined.

### C^1

```text
upstream 64x64
  AUC 0.6903137789904502
  recall 0.5934515688949522
  original AUC>=0.70 formal gate FAIL
  engineering-selected only

downstream 64x64
  AUC 1.0
  recall 1.0
  formal calibration PASS
```

### Tube_2

```text
raw snapshots 3776
root cells 2446
full cells 2871
new all-state root cells vs Tube_1 = 304
new all-state full cells vs Tube_1 = 467
```

Again these are control-support occupancy numbers, not causal capability growth.

### pi_2

Completed 10,009,600 transitions.

Locked source panel:

```text
pi_1 3115/3119
pi_2 3002/3119
upstream   423/427 -> 312/427
downstream 2692/2692 -> 2690/2692
```

Historical locked pi_1-negative challenge:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
successful parent groups 3
baseline reproduction failures 0
```

Under the old method this was strong continuation-frontier progression evidence. Under the new causal paper definition it **cannot be promoted to proof of ground-connected Jump-Capability expansion**, because those challenge states were generated from the older RSI-anchor frontier protocol.

It remains useful evidence that pi_2 can continue successfully from many previously difficult states and that policy realization shifted strongly in upstream.

pi_2 remains unselected as next policy authority.

---

## 13. What the historical work still contributed

The causal correction does not make the previous engineering work useless.

It established:

1. phase expert bootstrap works;
2. continuation snapshot/restore works;
3. unified Actor training works;
4. Tube-RSI replay works;
5. continuation models and threshold calibration work;
6. role isolation / locked acceptance infrastructure exists;
7. iterative Tube construction and policy freeze/evaluation exist;
8. raw-state visualization exposed the method flaw;
9. pi_2 demonstrated the distinction between frontier acquisition and single-policy retention.

The new causal method reuses most of this infrastructure while correcting what is allowed to count as capability evidence.

---

## 14. Automatic iteration after the causal redesign

`JIT/cli/prepare_iterative_envelope_workflow.py` now requires one locked `jit_nominal_jump_centerline_v2` artifact and optionally the previous causal capability summary.

Prospective DAG:

```text
locked centerline
        ↓
source raw/control Tube geometry diagnostic
        ↓
source semantic Jump view diagnostic
        ↓
unrevised role plan
        ↓
causal every-slice plan revision
        ↓
causal TRAIN / CALIBRATION / ACCEPTANCE forward acquisition
        ↓
RSI continuation labeling after reachability
        ↓
causal reachable∩viable capability analysis
        ↓
REQUIRE new TRAIN causal root cells > 0
        ↓
fit/calibrate C^k
        ↓
build raw Tube_(k+1), preserving causal provenance for new rows
        ↓
raw/control Tube geometry + semantic diagnostic
        ↓
smoke / role isolation / baseline lock
        ↓
train/freeze one candidate policy
        ↓
locked paired evaluation
        ↓
capability progression + policy realization
        ↓
prospective selection or STOP
```

The workflow code is integrated, but no full prospective causal iteration has been executed yet. Therefore the project must not claim demonstrated end-to-end causal automation yet.

---

## 15. Paper outline

### Working paper thesis

> A robot's empirical jumping capability should be identified from states that are both causally reachable from the real launch condition and viable for task continuation. JIT iteratively expands this ground-connected capability Tube and turns newly discovered frontier states into just-in-time curriculum for a single deployable policy.

### Proposed structure

#### 1. Introduction

Problem: reward/return alone does not tell us what part of a robot's jump state space is physically realized, reachable, or merely continuation-capable after artificial reset.

Contributions:

1. causal ground-connected Jump Capability Tube definition;
2. trajectory-centered 0.1 m longitudinal parameterization with anisotropic physical resolution;
3. two-stage forward-reachability then continuation-viability identification;
4. JIT frontier-to-curriculum loop for one unified policy;
5. explicit separation between discovered capability and single-policy realization.

#### 2. Problem formulation

Define fixed dynamics/task, natural ground start, action limits, successful terminal event chain, physical state projection and resolution.

Define empirical reachable evidence `R`, continuation evidence `V`, causal capability Tube `J=R∩V`, and policy realization.

#### 3. Bootstrap

Explain `pi_up`, `pi_down`, continuation bootstrap, why experts are discovery probes rather than runtime switchers.

#### 4. Nominal centerline

Ground-connected successful rollout, real-frame x indexing, 2.5 m to first landing/hard max4.2, 0.1m slices.

#### 5. Causal frontier acquisition

Natural reset, policy prefix, spatial lookback perturbation, env.step-only arrival, reachability provenance, downstream semantics.

#### 6. Continuation identification

RSI after reachability, continuation labels, `C_up/C_down`, TRAIN/CALIBRATION/ACCEPTANCE isolation.

#### 7. JIT curriculum and unified policy

How TRAIN-positive causal frontier states become replay support; raw replay vs capability evidence; one Actor training.

#### 8. Evaluation

Report at least:

```text
causal root cells by x slice
new causal cells per iteration
cross-section growth in z/v/pitch/rates
forward-reachable but continuation-negative frontier
continuation-positive but noncausal RSI diagnostic region, if studied
single-policy realization coverage
```

#### 9. Ablations

Important candidates:

- causal acquisition vs RSI-anchor acquisition;
- global frontier selection vs every-x-slice probing;
- raw snapshot count vs resolution-aware cell accounting;
- 0.1/0.2/0.3 m lookback families;
- with/without core replay for policy realization, only if predeclared later.

#### 10. Limitations

Empirical reachability is proposal-family conditioned; no formal reachability certificate; fixed XML/task; resolution choices are engineering declarations; current policy is target-free; final Tube may miss feasible states not discovered by the exploration family.

---

## 16. Engineering outline

### Layer A — immutable task/runtime

```text
XML / physics / action mapping / success events
```

No change.

### Layer B — raw replay support

```text
Soft Tube snapshots / Tube-RSI
```

Training artifact, not capability proof.

### Layer C — ground-connected acquisition

```text
natural reset -> frozen policy prefix -> lookback perturbation -> target slice
```

New authoritative reachability source.

### Layer D — continuation evaluation

```text
exact reached snapshot -> RSI -> success/failure
```

RSI permitted here.

### Layer E — causal physical Tube

```text
reachable + viable -> resolution-aware root/full cells -> x cross-sections
```

Primary scientific artifact.

### Layer F — continuation models

`C_up^k/C_down^k` remain proposal/filtering models; they do not estimate reachability.

### Layer G — unified policy

One deployable Actor trained from replay/curriculum support. Its coverage is evaluated separately from cumulative capability evidence.

### Layer H — orchestration

Recorded pre-outcome roles, baseline locking, stopping on failed scientific gates, no automatic repair sweep.

---

## 17. Current exact position

```text
Historical experts                             DONE
Historical Tube_0 / pi_0 / C^0                DONE
Historical Tube_1 / pi_1                      DONE
Historical C^1 engineering path               DONE
Historical Tube_2 / pi_2                      DONE
Locked pi_1 vs pi_2                           DONE
All-state physical-resolution analysis         DONE
Trajectory-centered semantic redesign          DONE in code
Causal ground-reachability redesign             DONE in code
Causal centerline v2                            DONE in code
Causal every-slice frontier plan                DONE in code
Causal ground-start acquisition                 DONE in code
Causal role runner                              DONE in code
Causal reachable∩viable analyzer                DONE in code
Causal provenance in new raw Tube rows          DONE in code
Local compile/tests                             NOT YET RUN by assistant
Locked real centerline artifact                 NOT YET GENERATED locally
First prospective causal frontier               NOT YET RUN
First causal capability summary                 NOT YET GENERATED
Next unified policy under causal method          NOT AUTHORIZED YET
Final TEST/JCE/JEL                              UNTOUCHED
```

---

## 18. Immediate next work

### Step 1 — local compile/tests

Run the regression suite before any simulation campaign.

### Step 2 — obtain/verify one successful natural-start reference rollout

Preferred source is current selected `pi_1` if it succeeds on the canonical natural start. If it does not, stop and choose another predeclared frozen successful source. Do not fabricate or interpolate the path and do not select a reference by looking at final TEST.

### Step 3 — build and lock `centerline_v2`

Verify:

```text
natural_start_connected=true
real_frames_only=true
RSI reachability=false
x step 0.1m
downstream vz<0
first landing terminal
```

### Step 4 — run the first causal discovery round before training a new policy

Use `pi_1` as the current control/proposal authority and the historical raw Tube only as runtime/replay context. Do **not** provide a previous causal summary in the first causal round; the locked centerline is the initial causal capability baseline.

The critical first result is not a PPO score. It is:

```text
How many new TRAIN-positive ground-connected root cells are found beyond the centerline?
Where along x are they found?
Which target slices remain narrow or produce only continuation-negative states?
```

### Step 5 — inspect causal geometry before candidate training

If the first causal discovery adds no meaningful new cells, do not train a new policy. Change the predeclared exploration family only after diagnosing reachability limits.

If it adds meaningful cells, continue through continuation fitting, raw replay expansion, baseline lock and candidate training.

### Step 6 — evaluate capability and realization separately

A candidate may expand causal capability but fail as the next single-policy authority if it loses too much prior coverage. Preserve both facts.

---

## 19. Stopping rule direction

The eventual stopping rule should be based on causal capability saturation rather than raw replay count. Candidate ingredients:

```text
new causal root cells per iteration becomes negligible
no x slice gains new cross-section support
repeated reachability attempts fail to expand frontier under declared proposal families
new reachable states are mostly continuation-negative
policy realization cannot absorb discovered support without unacceptable loss
resource budget reached
```

The exact thresholds must be predeclared before the final development iteration.

---

## 20. Claim boundary

Supported after code validation and future causal runs:

```text
empirical ground-connected reachable evidence under the declared exploration controller family
continuation viability under the declared frozen policy
resolution-aware causal Jump Capability Tube
just-in-time curriculum generated from causal TRAIN frontier
single-policy realization measured independently
```

Not supported:

```text
formal reachability proof
complete physical feasible set
viability kernel
certified safe set
invariant set
global physical jump limit proof
```

The scientific value of JIT comes from making those distinctions explicit rather than hiding them behind reward or RSI success.
