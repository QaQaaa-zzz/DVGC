# JIT trajectory-centered Jump-Tube task report — 2026-09-04

## 0. Executive summary

The project has completed the historical engineering chain through `pi_2`, but the latest Tube visualizations revealed a method-level problem: the old frontier logic expanded a **global set of continuation-support states**, not a longitudinally indexed **jump Tube around one successful jump trajectory**.

This distinction matters. Historical downstream-labelled states extend far into late recovery (for example around `x ~= 4.5 m`) and can even occupy regions that visually rise in `z`. Those states may be legitimate control/replay states, but they are not useful frontier evidence for the jump itself. Counting them as Jump-Tube expansion inflates downstream coverage and obscures the actual ascent -> apex -> descent band.

The mainline is therefore revised before any `pi_3` work.

New method:

```text
one successful real full-chain jump rollout
        ↓
nominal centerline from real captured frames only
        ↓
0.1 m longitudinal x slices, nominally x=2.5 ... 4.2 m
        ↓
actual end = first valid landing if it occurs earlier
        ↓
for each x slice, widen the local physical cross-section
        ↓
resolution-aware continuation labeling / C^k
        ↓
Jump-Tube grows as cross-sectional support around the trajectory
        ↓
train one unified policy from the resulting curriculum
```

The current mainline remains **without goal/intent conditioning**. No reward, Actor observation, XML, action semantics, TEST/JCE/JEL, or runtime expert-switching contract is changed by this redesign.

---

## 1. What JIT is now intended to identify

The central scientific object is not a bag of successful states and not one target-conditioned path.

For one fixed robot XML and one fixed jump task, JIT seeks an empirical state-space tube around a real successful jump trajectory:

```text
x = 2.5       x = 2.6       ...       x = 4.x / landing
   T(2.5)        T(2.6)                    T(x_end)
      \             \                         /
       \_____________ empirical Jump Tube ___/
```

At one longitudinal position `x`, the Tube cross-section may contain multiple valid combinations of:

```text
root y/z
root vx/vy/vz
roll/pitch/yaw
root wx/wy/wz
joint pose/rates
wheel tangential speed
```

The centerline is therefore a **scaffold**, not a fixed desired trajectory that the controller must reproduce exactly. JIT widens around it.

The scientific hierarchy remains:

```text
F*   unknown physical/task feasibility under fixed dynamics
E_k  cumulative successful real-dynamics evidence
J_k  trajectory-centered empirical Jump-Tube support at iteration k
R_k  realization coverage of one unified policy over accumulated support
```

JIT does not prove `F*` or a viability kernel.

---

## 2. Why the previous target-free Tube definition was insufficient

The previous revision correctly introduced physical state resolution, but it still allowed frontier parent selection by:

```text
newest Tube shell
+ low continuation score
+ distinct parent group
+ distinct exact state
+ distinct root geometry cell
```

The missing condition was **task progress geometry**.

The selector did not require:

```text
x within the actual jump corridor
one parent opportunity per 0.1 m longitudinal slice
downstream root_vz < 0
termination of Jump-Tube support at first valid landing
```

As a result, a dense late-recovery region could dominate the low-score parent pool. A state could be labelled `downstream` because the unified phase had already crossed Apex, even when the state was no longer part of the descending jump trajectory.

The latest physical plots made this visible and triggered the redesign.

---

## 3. Historical chain completed before this redesign

### 3.1 Phase experts

The bootstrap used two frozen discovery experts:

```text
pi_up_star   Propulsion-Ascent expert
pi_down_star Descent-Recovery expert
```

Known identities:

```text
pi_up_star
  9,977,856 transitions
  actor f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081

pi_down_star
  25,600 transitions
  actor 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

Experts are discovery probes/data sources only. Final runtime remains one unified policy.

### 3.2 Tube_0

Historical raw replay artifact:

```text
JIT/runs/soft_tube/soft_tube_train_v1_20260828
222 raw snapshots
117 upstream
105 downstream
```

Resolution-aware retrospective analysis already showed:

```text
100 unique root-geometry cells
112 unique full-physical cells
root duplicate fraction ~54.95%
13 occupied x slices
```

Therefore the original count `222` was never `222` independent macroscopic capability regions.

### 3.3 pi_0

Unified `pi_0` established that two-expert bootstrap evidence can be consumed by one Actor.

```text
10,009,600 transitions
actor 43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029
```

### 3.4 Tube_1

Historical raw artifact:

```text
3,119 raw snapshots
= 222 retained Tube_0
+ 2,897 raw expansion rows
```

Resolution-aware retrospective physical occupancy:

```text
2,142 unique root-geometry cells
2,404 unique full-physical cells
2,042 new root cells vs Tube_0
2,292 new full cells vs Tube_0
24 occupied x slices
```

Phase counts in the old all-state physical view:

```text
upstream   194 root cells
downstream 1,948 root cells
```

These numbers remain valid as **physical occupancy of the historical control/replay Tube**, but the downstream part must now be reclassified under Jump-Tube semantics. It is no longer acceptable to treat all 1,948 downstream cells as jump-boundary coverage.

### 3.5 pi_1 repair02

Selected engineering authority:

```text
Tube_0 quickcheck 222/222
upstream 117/117
downstream 105/105
historical boundary 26/260 across 4 parent groups
```

Historical formal Iteration-1 PASS is not claimed because three old baseline-reproduction mismatches remain quarantined.

### 3.6 C^1

Upstream 64x64 continuation model:

```text
AUC    0.6903137789904502
recall 0.5934515688949522
original AUC >= 0.70 formal gate: FAIL
engineering-selected: YES
```

Downstream 64x64:

```text
AUC 1.0
recall 1.0
formal calibration PASS
```

### 3.7 Tube_2

Historical raw artifact:

```text
3,776 raw snapshots
= 3,119 retained Tube_1
+ 657 raw expansion rows
```

Resolution-aware all-state physical occupancy:

```text
2,446 unique root-geometry cells
2,871 unique full-physical cells
304 new root cells vs Tube_1
467 new full cells vs Tube_1
24 occupied x slices
```

Old phase-labelled new root cells:

```text
upstream   +199
downstream +105
```

Again, these are valid physical-cell counts for the historical control Tube, not yet valid Jump-Tube expansion counts.

### 3.8 pi_2

`pi_2` completed 10,009,600 transitions.

Locked source-panel comparison:

```text
pi_1 3115/3119
pi_2 3002/3119
strict regressions 115

upstream:   423/427 -> 312/427
downstream: 2692/2692 -> 2690/2692
```

Locked pi_1-negative challenge:

```text
pi_2 13/14
upstream 4/5
downstream 9/9
successful parent groups 3
baseline reproduction failures 0
```

Interpretation retained:

- pi_2 learned clear new local frontier behavior;
- upstream single-policy realization substantially degraded;
- pi_2 remains empirical capability evidence;
- pi_2 is not retrospectively selected as the next policy authority.

No `pi_3` training is authorized yet.

---

## 4. New nominal centerline contract

Implementation:

```text
JIT/src/jit_dvgc/analysis/nominal_jump_centerline.py
JIT/cli/build_nominal_jump_centerline.py
```

Input is a **completed successful canonical natural evaluation** of a frozen unified policy. The canonical evaluator already stores every physical `qpos/qvel` frame in its trace artifact.

Centerline construction rules:

```text
x nominal start = 2.5 m
x hard maximum = 4.2 m
x spacing = 0.1 m
actual terminal = first valid landing/contact if earlier than 4.2 m
```

For every target x slice:

```text
choose nearest real captured simulator frame
NO qpos/qvel interpolation
maximum x mismatch <= 0.05 m
```

Branch semantics:

```text
pre-Apex slices    -> upstream
Apex-near slice    -> apex semantic marker
post-Apex slices   -> downstream only if root_vz < 0
first valid landing -> terminal of the nominal Jump-Tube centerline
post-landing frames -> excluded
```

The centerline does not modify reward and is not added as an Actor intent.

---

## 5. Physical resolution contract retained

The already declared capability resolution remains:

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

Two profiles remain:

```text
root_geometry_v1
  root pose + root twist
  primary Jump-Tube geometry / frontier diversity

full_physical_v1
  root geometry + joint pose/rates + wheel tangential speed
  fine physical deduplication
```

Actor FIFO/history/last-action bits do not define physical-cell identity.

---

## 6. Raw/Control Tube versus Jump Capability Tube

The redesign intentionally separates two views.

### 6.1 Raw/Control Tube

Purpose:

```text
replay
training support
historical provenance
exact restartable snapshots
```

It may contain landing/recovery states because the unified controller still has to finish the task.

Historical Tube_0/1/2 artifacts remain immutable.

### 6.2 Jump Capability Tube

Purpose:

```text
jump-boundary accounting
x-slice cross-section geometry
frontier parent eligibility
publication-level capability visualization
```

Implementation:

```text
JIT/src/jit_dvgc/analysis/jump_tube_view.py
JIT/cli/analyze_jump_tube_view.py
```

Semantic filter:

```text
x within nominal centerline support
upstream -> source upstream phase inside corridor
downstream -> source downstream phase AND root_vz < 0
post-landing recovery -> excluded
```

This does not delete late recovery from replay; it removes it from Jump-Tube claims.

---

## 7. New frontier acquisition rule

Implementation remains under:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

The CLI now requires:

```text
--nominal-centerline
```

A future parent must satisfy all of:

```text
1. source = newest raw Tube shell
2. parent_group_id unique
3. exact state unique
4. root_geometry_v1 cell unique
5. x slice exists on the nominal centerline
6. x is inside the actual jump corridor
7. downstream must have root_vz < 0
8. no post-landing / late-recovery frontier support
```

The previous global ranking has also changed.

Old:

```text
pool entire phase
sort all candidates by continuation score
select globally weakest states
```

New:

```text
split eligible candidates by x slice
rank weak-score frontier locally inside each slice
round-robin across x slices
only after every occupied slice had an opportunity may a slice contribute again
```

This prevents one dense region from monopolizing the acquisition budget.

Default parent budget remains up to 25 distinct cells per phase, with at least 5 required after semantic filtering.

---

## 8. What one JIT iteration should mean from now on

Prospective iteration `k -> k+1` should be interpreted as local Tube widening around the jump trajectory:

```text
selected pi_k
+ source raw/control Tube_k
+ successful nominal centerline
        ↓
physical projection of source Tube
        ↓
Jump-Tube semantic view
        ↓
for each supported x slice:
    identify local weak frontier cells
        ↓
real-dynamics action perturbations
        ↓
continuation labels under frozen pi_k
        ↓
TRAIN / CALIBRATION / ACCEPTANCE isolation
        ↓
C_up^k / C_down^k
        ↓
qualifying new replay states
        ↓
raw/control Tube_(k+1)
        ↓
rebuild Jump-Tube_(k+1) view
        ↓
measure new cross-section support by x slice
        ↓
train one unified pi_(k+1)
        ↓
locked capability-progression + policy-realization evaluation
```

The meaningful expansion question becomes:

> At which longitudinal slices did the physically resolved successful cross-section widen, by how many new root/full cells, and in which physical dimensions?

Not:

> How many raw JSON/snapshot rows were added?

---

## 9. Required Tube progression metrics

Every future iteration should report both raw replay and Jump-Tube metrics.

### Raw/control metrics

```text
raw snapshot count
retained count
new raw replay rows
```

### Jump-Tube metrics

```text
unique root-geometry cells
unique full-physical cells
new root cells vs source
new full cells vs source
cell duplicate fraction
occupied x slices
new cells per x slice
phase/branch counts
```

### Cross-section metrics

At every 0.1 m x slice:

```text
z support
vx/vz support
pitch support
pitch-rate support
other root twist/attitude support
```

The primary plots should show the nominal centerline and the occupied cross-section around it.

---

## 10. How the latest historical plots are now interpreted

The latest all-state physical plots showed:

```text
upstream cluster around the jump approach/ascent
large dense downstream-labelled cloud farther forward
sparse transition support between them
```

The project no longer interprets the full orange cloud as one valid downstream Jump-Tube branch.

The correct next action is retrospective semantic filtering:

```text
Tube_0 physical geometry -> Jump-Tube_0 view
Tube_1 physical geometry -> Jump-Tube_1 view
Tube_2 physical geometry -> Jump-Tube_2 view
```

This will quantify how many historical downstream cells disappear once:

```text
x > first valid landing
or downstream vz >= 0
```

is excluded from Jump-Tube accounting.

Until this filtered analysis is run, the historical `downstream 1,948 root cells` and related counts must not be described as downstream jump-envelope size.

---

## 11. Current automation maturity

The repository already has a generic automatic `k -> k+1` infrastructure for:

```text
outcome-blind role split
frontier acquisition/labeling
C^k fit/calibration
raw Tube construction
Tube-RSI smoke
role isolation
locked baseline
candidate train/freeze
paired evaluation
capability progression
prospective policy selection
```

The new trajectory-centered layer is implemented as production capabilities:

```text
successful rollout -> nominal centerline
physical Tube -> Jump-Tube view
nominal centerline + geometry -> x-balanced frontier plan revision
```

It has **not yet been executed in a complete new prospective iteration**. Therefore the project must not claim end-to-end trajectory-centered automatic JIT yet.

The next prospective workflow should explicitly record these extra stages before frontier outcomes:

```text
canonical successful rollout / centerline lock
-> source physical geometry analysis
-> source Jump-Tube semantic view
-> ordinary outcome-blind frontier plan
-> trajectory-centered x-balanced plan revision
-> frontier TRAIN/CALIBRATION/ACCEPTANCE
```

---

## 12. Immediate operator tasks

### A. Pull and verify code

```bash
cd ~/DVGC
git pull --ff-only origin agent/two-phase-soft-tube

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli
$PY -m pytest -q \
  JIT/tests/test_nominal_jump_centerline.py \
  JIT/tests/test_resolution_frontier_parent_selection.py \
  JIT/tests/test_capability_tube_resolution.py \
  JIT/tests/test_capability_progression.py
```

### B. Produce one successful canonical natural rollout for the selected pi_1 authority

Use the existing canonical natural evaluation CLI with the exact pi_1 formal config and completed checkpoint used by the frozen engineering authority:

```bash
$PY JIT/cli/evaluate_unified_natural.py \
  --config <PI1_FORMAL_CONFIG> \
  --checkpoint <PI1_COMPLETED_CHECKPOINT> \
  --output-dir JIT/runs/trajectory_centerline/pi1_canonical_natural_20260904
```

The centerline builder will refuse an evaluation that does not report full recovery success. If the deterministic canonical pi_1 rollout fails, stop and select a separately predeclared successful real rollout source; do not fabricate a path.

### C. Build the nominal centerline

```bash
$PY JIT/cli/build_nominal_jump_centerline.py \
  --canonical-evaluation-report JIT/runs/trajectory_centerline/pi1_canonical_natural_20260904/report.json \
  --output-dir JIT/runs/trajectory_centerline/pi1_nominal_centerline_v1
```

Primary artifact:

```text
JIT/runs/trajectory_centerline/pi1_nominal_centerline_v1/centerline.json
```

### D. Rebuild historical Jump-Tube views

Assuming the existing physical geometry analyses:

```text
JIT/runs/capability_geometry/tube0_resolution_v1/summary.json
JIT/runs/capability_geometry/tube1_vs_tube0_resolution_v1/summary.json
JIT/runs/capability_geometry/tube2_vs_tube1_resolution_v1/summary.json
```

Run:

```bash
CENTER=JIT/runs/trajectory_centerline/pi1_nominal_centerline_v1/centerline.json

$PY JIT/cli/analyze_jump_tube_view.py \
  --capability-geometry-summary JIT/runs/capability_geometry/tube0_resolution_v1/summary.json \
  --nominal-centerline ${CENTER} \
  --output-dir JIT/runs/jump_tube_geometry/tube0_jump_view_v1

$PY JIT/cli/analyze_jump_tube_view.py \
  --capability-geometry-summary JIT/runs/capability_geometry/tube1_vs_tube0_resolution_v1/summary.json \
  --source-capability-geometry-summary JIT/runs/capability_geometry/tube0_resolution_v1/summary.json \
  --nominal-centerline ${CENTER} \
  --output-dir JIT/runs/jump_tube_geometry/tube1_vs_tube0_jump_view_v1

$PY JIT/cli/analyze_jump_tube_view.py \
  --capability-geometry-summary JIT/runs/capability_geometry/tube2_vs_tube1_resolution_v1/summary.json \
  --source-capability-geometry-summary JIT/runs/capability_geometry/tube1_vs_tube0_resolution_v1/summary.json \
  --nominal-centerline ${CENTER} \
  --output-dir JIT/runs/jump_tube_geometry/tube2_vs_tube1_jump_view_v1
```

Then inspect:

```text
summary.json
jump_tube_x_z_cells.png
jump_tube_x_z_vx_cells_3d.png
```

### E. Only after the historical audit, prepare the next prospective frontier plan

The new plan revision requires the centerline explicitly:

```bash
$PY JIT/cli/prepare_resolution_aware_frontier_plan.py \
  --source-plan <OUTCOME_BLIND_ORDINARY_PLAN> \
  --source-tube <SOURCE_TUBE> \
  --capability-geometry-summary <SOURCE_GEOMETRY_SUMMARY> \
  --nominal-centerline ${CENTER} \
  --output <TRAJECTORY_CENTERED_PLAN>
```

Do not launch the role runs until the selected x-bin distribution and semantic rejection counts are inspected.

---

## 13. Decision gates before any new pi_3-like training

The project should not train a new policy merely because a new plan can be generated.

First require:

```text
1. nominal centerline full-chain success
2. centerline covers the actual jump with <=0.05 m x matching error
3. downstream centerline points all have vz < 0
4. first valid landing terminates the Jump Tube
5. historical Jump-Tube_0/1/2 views are recomputed
6. dense late-recovery contamination is quantified
7. next frontier parents are distributed across multiple x slices
8. TRAIN/CALIBRATION/ACCEPTANCE remain parent-disjoint and outcome-blind
```

Only then collect new frontier data.

---

## 14. What is still unresolved

The redesign does not yet answer:

1. How many of Tube_1's historical downstream cells remain after the trajectory/descending filter?
2. How many of Tube_2's +304 new root cells are true Jump-Tube expansion?
3. Which x slices remain thin or empty?
4. Is upstream realization loss in pi_2 associated with cross-sectional widening near specific x slices?
5. Does the 0.1 m longitudinal resolution need later refinement around Apex or landing?
6. Is one nominal centerline sufficient, or should later JIT versions maintain a small archive of successful centerlines without adding explicit Actor intent?

These are empirical next questions, not assumptions.

---

## 15. Claims that remain prohibited

Do not claim:

```text
Soft Tube = certified safe set
Jump Tube = exact viability kernel
current outer boundary = physical maximum jump limit
raw snapshot growth = capability-volume growth
pi_2 strict gate failure = no capability gain
all historical downstream cells = valid descending jump support
```

The correct current claim is narrower:

> JIT has produced substantial empirical real-dynamics support and demonstrated local frontier progression, but the historical global Tube definition mixed the actual jump trajectory with late downstream/recovery support. The method has therefore been revised to identify a physically resolved Tube around a successful real jump trajectory, slice-by-slice in longitudinal progress.

---

## 16. Immutable task identity

```text
repository  QaQaaa-zzz/DVGC
branch      agent/two-phase-soft-tube
XML         assets/orange_bike_4kg_horizontal.xml
XML SHA     0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a
payload     2 kg
control     50 Hz
hip/knee    +/-30 N m actuator range
actions     [steer, rear-wheel drive, hip, knee]
runtime     one unified Actor, no expert switching
TEST/JCE/JEL untouched
```

---

## 17. Current exact project position

```text
historical experts                 DONE
Tube_0 / pi_0 / C^0               DONE
Tube_1 / pi_1                     DONE
C^1 engineering path              DONE
Tube_2 / pi_2                     DONE
locked pi_1 vs pi_2 evaluation    DONE
physical resolution redesign      DONE
Tube_0/1/2 all-state geometry     DONE locally
trajectory-centered code          IMPLEMENTED
nominal centerline artifact        NEXT local task
filtered Jump-Tube_0/1/2 views    NEXT local task
prospective x-balanced frontier    NOT RUN
pi_3 / next unified candidate      NOT AUTHORIZED
```

The next work is therefore **measurement and geometric reconstruction**, not another PPO run.
