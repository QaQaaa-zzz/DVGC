# DVGC/JIT Resolution-Aware Capability-Tube vNext Outline — 2026-09-04

## 1. Executive conclusion

The project has completed the engineering chain:

```text
pi_up_star + pi_down_star
-> Tube_0
-> pi_0
-> C^0
-> Tube_1
-> pi_1
-> C^1
-> Tube_2
-> pi_2
-> locked pi_1 vs pi_2 evaluation
```

The next step is **not** pi_3 training, a replay-ratio sweep, or intent conditioning.

The next step is to reconstruct Tube_0/1/2 as a **resolution-aware physical capability Tube** and determine how much of the raw snapshot growth represents genuinely different regions of physical state space.

The central methodological correction is:

> Raw snapshot cardinality is a replay/data quantity. Capability expansion must be measured in physically resolved cells and their longitudinal Tube geometry.

Historical raw counts remain valid:

```text
Tube_0 =   222 snapshots
Tube_1 = 3,119 snapshots
Tube_2 = 3,776 snapshots
```

But they are no longer used as direct capability-envelope multipliers.

The first physical resolution contract is:

```text
position                     0.10 m
linear velocity              0.10 m/s
orientation                  0.50 deg
angular velocity             2.0 deg/s
joint angle                  0.50 deg
joint angular velocity       2.0 deg/s
wheel tangential speed       0.10 m/s
phase                        discrete
```

The Tube is sliced along root `x` every 0.10 m so its cross-section, widening, narrowing, branching and gaps become directly inspectable.

---

# 2. What JIT is actually studying

JIT separates four objects.

## 2.1 Physical/task feasibility `F*`

`F*` is the conceptual set of physical states from which some admissible control behavior could complete the fixed jumping task.

JIT does not prove or exactly compute this set.

## 2.2 Cumulative empirical capability evidence `E_k`

`E_k` is successful real-dynamics evidence accumulated from frozen phase experts and unified policies.

A later policy failing at an earlier successful state does not erase the historical successful evidence.

## 2.3 Resolution-aware capability Tube `T_k`

`T_k` is a sparse occupied-cell representation of the empirical jumping state space.

It answers:

> At the declared physical resolution, which parts of the jumping state space have actually been supported or demonstrated?

## 2.4 Single-policy realization `R(pi_k, T_k)`

The runtime target remains one unified Actor.

Therefore JIT separately measures how much of the accumulated capability support one particular policy can realize.

This separates:

```text
system/task capability evidence
from
latest-policy behavioral coverage
```

---

# 3. Why intent is not added now

The current method remains target-free.

No desired jump distance, apex target, landing target, or latent intent is added to the Actor in this revision.

Reason:

The desired research object is a **state-space Tube with nonzero cross-section**, not one fixed target trajectory.

At the same `x`, multiple states can all be valid:

```text
different z
different vx/vz
different pitch/pitch-rate
different joint configuration
```

Adding a fixed intent now would require changing reward/task semantics and would open a different goal-conditioned research question.

Therefore current vNext preserves:

```text
reward unchanged
Actor observation unchanged
success semantics unchanged
target-free task unchanged
```

Goal-conditioned JIT remains a later method option only if the reconstructed Tube geometry shows that multiple successful branches create a genuine representation problem for one target-free Actor.

---

# 4. Authoritative physical/task identity

Repository:

```text
QaQaaa-zzz/DVGC
```

Branch:

```text
agent/two-phase-soft-tube
```

XML:

```text
assets/orange_bike_4kg_horizontal.xml
```

XML SHA-256:

```text
0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a
```

Physical/runtime contract:

```text
payload                    2 kg
simulation substep         0.005 s
control interval           0.020 s
control frequency          50 Hz
hip actuator range         +/-30 N m
knee actuator range        +/-30 N m
action order               steer, rear-wheel drive, hip, knee
runtime policy switching   none
```

The `4kg` token in the XML filename is historical; the actual payload geom mass is 2 kg.

Older documents that stated hip/knee +/-50 N m were inconsistent with the authoritative XML and runtime validator. The XML itself has not been changed.

---

# 5. Completed chain from experts to pi_2

## 5.1 Propulsion-Ascent expert

`pi_up_star`:

```text
training transitions = 9,977,856
actor SHA = f218775e3cf99555ce524f1357a800172904bc815b06c54a53db8965204d9081
```

Role:

- bootstrap launch/ascent capability;
- produce real-dynamics continuation evidence;
- discovery probe only, not runtime switching.

## 5.2 Descent-Recovery expert

`pi_down_star`:

```text
training transitions = 25,600
actor SHA = 7b25f54bb1df3b97f63a15d011d66c2440682efb10b0510a266a9066725dd8be
```

Frozen expert manifest:

```text
JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json
```

## 5.3 Bootstrap V_up / V_down and Tube_0

Raw Tube_0:

```text
JIT/runs/soft_tube/soft_tube_train_v1_20260828
```

Composition:

```text
222 raw TRAIN snapshots
= 117 upstream
+ 105 downstream
```

Manifest SHA:

```text
c1c1161ebafd16716f2566aaccfe89169fe9cb0c2b090266c0e2bf90165df28b
```

## 5.4 pi_0

Frozen pi_0:

```text
JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json
```

```text
training transitions = 10,009,600
actor SHA = 43e82928c3643e5616a665b43814819a34b7a1a5bba5b6641f2a11ad4907e029
payload SHA = fb107a5f31b1455f9626c3be68efab36457fb801fcfbba9e99acc0deff3b5719
```

pi_0 established that the initial two-phase capability support could be learned by one unified Actor.

## 5.5 C^0 and Tube_1

Raw Tube_1:

```text
JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901
```

Composition:

```text
3,119 raw snapshots
= 222 retained Tube_0
+ 2,897 raw expansion snapshots

upstream   = 427
downstream = 2,692
```

Manifest SHA:

```text
817a980a5dd84f36507f762a913c21c1fc0913580d925ff9c68e982edfd82a80
```

The old `3,119 / 222` ratio is now treated only as raw replay-support cardinality growth.

## 5.6 pi_1 repair02

Selected engineering authority:

```text
JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json
```

```text
actor SHA = 85d6b4667364daf8e054af9bccbf155dda16a62518df19883057fcfcbbd6f86a
payload SHA = 3b9af512c7e389aade1c86ca76e9420a0bc687c499f2ff9cf7637701dd5d0cbc
```

Historical engineering quickcheck:

```text
Tube_0 retention      222/222
upstream              117/117
downstream            105/105
boundary success      26/260
successful groups     4
```

Historical strict formal PASS is not claimed because the old quickcheck contains 3 baseline-reproduction mismatches from the historical PRNG hierarchy.

## 5.7 Iteration-1 frontier evidence

v3 TRAIN:

```text
total        1,031
upstream       821 = 785 positive + 36 negative, 9 parent groups
downstream     210 = 182 positive + 28 negative, 3 parent groups
```

v3b upstream CALIBRATION:

```text
739 = 733 positive + 6 negative, 3 parent groups
```

Downstream CALIBRATION:

```text
70 = 61 positive + 9 negative
```

v3c ACCEPTANCE:

```text
upstream   516 = 511 positive + 5 negative
downstream  70 =  61 positive + 9 negative
```

## 5.8 C^1

Selected architecture:

```text
76 -> 64 tanh -> 64 tanh -> 1
9,153 parameters per phase
```

Upstream:

```text
AUC               0.6903137789904502
positive recall   0.5934515688949522
accepted negative 0
formal AUC >= 0.70 = FAIL
engineering selection = YES
```

Downstream:

```text
AUC               1.0
positive recall   1.0
formal calibration = PASS
```

Correct claim:

```text
C_up^1 = engineering-selected
C_down^1 = formal calibration PASS
C^1 all-phase formal PASS = false
```

## 5.9 Tube_2

Raw Tube_2:

```text
JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904
```

Composition:

```text
3,776 raw snapshots
= 3,119 retained Tube_1
+ 657 raw expansion snapshots

upstream   =   902 = 427 + 475
downstream = 2,874 = 2,692 + 182
```

Manifest SHA:

```text
135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff
```

Tube-RSI smoke: GO.

The 657 added rows are now treated as **raw expansion snapshots**, not automatically 657 independent capability states.

## 5.10 pi_2

Training run id:

```text
pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904
```

Training contract:

```text
10,009,600 transitions
outer reset: 90% Tube / 10% natural
inside Tube: 75% retained Tube_1 / 25% raw Tube_2 expansion
```

No expert switching, TEST, or validation data were used.

---

# 6. Current pi_1 vs pi_2 result

## 6.1 Locked source Tube_1 panel

```text
pi_1 baseline success = 3115 / 3119
pi_2 success          = 3002 / 3119
strict regressions    = 115
strict improvements   = 2
```

By phase:

```text
upstream:
  pi_1 423/427 = 99.06%
  pi_2 312/427 = 73.07%
  regressions = 113

downstream:
  pi_1 2692/2692 = 100.00%
  pi_2 2690/2692 = 99.93%
  regressions = 2
```

## 6.2 Locked pi_1-negative frontier challenge

```text
state count                    14
pi_2 successes                 13
successful parent groups        3
upstream                      4/5
downstream                    9/9
baseline reproduction failures  0
```

Current scientific interpretation:

```text
local frontier progression = strong
upstream single-policy realization = substantially degraded
```

Therefore pi_2 is retained as capability evidence but is not retrospectively promoted as the next formal policy authority.

---

# 7. Why raw snapshot growth is insufficient

Two snapshots can have different exact physical hashes while occupying essentially the same physically meaningful neighborhood.

They remain separate replay snapshots, but if their physical coordinates quantize to the same declared cell they do not count as two independent macroscopic capability regions.

This distinction is important because:

```text
more replay density != more independent capability coverage
```

The retrospective Tube analysis measures this directly.

---

# 8. Capability coordinate schema v1

## 8.1 Root geometry profile

Primary macroscopic Tube geometry:

```text
phase
root x/y/z
root vx/vy/vz
roll/pitch/yaw
root wx/wy/wz
```

Use for:

- physical Tube shape;
- x-slice cross-sections;
- macroscopic expansion count;
- frontier parent diversity.

## 8.2 Full physical profile

Adds:

```text
steering angle/rate
hip angle/rate
knee angle/rate
front/rear wheel tangential speed
```

Use for finer physical configuration diversity.

Wheel rotational phase is intentionally excluded from cell identity; wheel tangential speed is retained.

---

# 9. Resolution contract v1

| Coordinate family | Resolution |
|---|---:|
| root x/y/z | 0.10 m |
| root vx/vy/vz | 0.10 m/s |
| roll/pitch/yaw | 0.50 deg |
| root wx/wy/wz | 2.0 deg/s |
| steering/hip/knee angle | 0.50 deg |
| steering/hip/knee rate | 2.0 deg/s |
| wheel tangential speed | 0.10 m/s |
| phase | discrete |

Native MuJoCo state units remain unchanged. Degrees and degree/s are used only for the analysis convention.

Quantization:

```text
nearest grid
round half away from zero
```

---

# 10. Physical Tube geometry

## 10.1 Longitudinal coordinate

Use root `x` as the first progress coordinate.

```text
Delta x = 0.10 m
```

## 10.2 Cross-section

For each x slice summarize unique root-geometry cells over:

```text
y, z
vx, vy, vz
roll, pitch, yaw
wx, wy, wz
```

Statistics:

```text
occupied root-cell count
raw snapshot density
phase composition
min/q25/median/q75/max
```

## 10.3 Shape interpretation

The empirical Tube is not forced to be cylindrical.

Possible meaningful structures include:

- widening;
- bottlenecks;
- branches;
- disconnected lobes;
- gaps.

A multi-lobed upstream Tube would be especially interesting because it could help explain why pi_2 gains new frontier behavior while losing old upstream realization. That is a hypothesis to test, not a current conclusion.

---

# 11. New production code

## 11.1 Capability Tube analysis

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

Responsibilities:

- read replay snapshots;
- project qpos/qvel to physical coordinates;
- quaternion -> Euler attitude;
- wheel angular rate -> tangential speed;
- quantize by per-variable resolutions;
- construct sparse root/full cells;
- calculate x-slice geometry;
- compare source/target Tube generations;
- generate plots when matplotlib is available;
- retain provenance/self-hashed analysis artifacts.

Generated files:

```text
summary.json
entries.json
cells.json
x_slices.json
projected_points.csv
tube_x_z_cells.png
tube_x_vx_cells.png
tube_x_pitch_cells.png
tube_x_z_vx_cells_3d.png
```

## 11.2 Resolution-aware frontier parents

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

Future parent requirement:

```text
newest Tube shell only
+ unique parent group
+ unique exact state
+ unique root_geometry_v1 cell
```

Default:

```text
max distinct parent cells per phase = 25
minimum distinct parent cells per phase = 5
```

Role pattern remains outcome-blind:

```text
TRAIN, TRAIN, TRAIN, CALIBRATION, ACCEPTANCE, repeat
```

---

# 12. New quantitative Tube metrics

Per Tube:

```text
raw_snapshot_count
unique_root_geometry_cell_count
unique_full_physical_cell_count
root_geometry_duplicate_fraction
full_physical_duplicate_fraction
x_slice_count
x_min_center_m
x_max_center_m
phase-specific cell counts
```

Source-vs-target comparison:

```text
raw_snapshot_growth
new_root_geometry_cell_count
new_full_physical_cell_count
new cells by phase
new_root_geometry_cells_per_raw_added_snapshot
nearest source-cell L2 distance in resolution units
nearest source-cell Linf distance in resolution units
```

Primary macroscopic expansion metric:

```text
new root_geometry_v1 cells
```

A new full physical cell without a new root geometry cell may still represent useful internal posture diversity, but it should not be described as new macroscopic jumping-envelope geometry.

---

# 13. Exact local validation procedure

## 13.1 Pull and test

```bash
cd ~/DVGC
git pull --ff-only origin agent/two-phase-soft-tube

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli

$PY -m pytest -q \
  JIT/tests/test_capability_tube_resolution.py \
  JIT/tests/test_resolution_frontier_parent_selection.py \
  JIT/tests/test_capability_progression.py \
  JIT/tests/test_iteration_policy_selection_capability.py
```

No GPU training is required for the retrospective Tube projection.

## 13.2 Analyze Tube_0

```bash
T0=JIT/runs/soft_tube/soft_tube_train_v1_20260828

$PY JIT/cli/analyze_capability_tube.py \
  --tube ${T0} \
  --output-dir JIT/runs/capability_geometry/tube0_resolution_v1
```

## 13.3 Analyze Tube_1 versus Tube_0

```bash
T1=JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901

$PY JIT/cli/analyze_capability_tube.py \
  --tube ${T1} \
  --source-tube ${T0} \
  --output-dir JIT/runs/capability_geometry/tube1_vs_tube0_resolution_v1
```

## 13.4 Analyze Tube_2 versus Tube_1

```bash
T2=JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904

$PY JIT/cli/analyze_capability_tube.py \
  --tube ${T2} \
  --source-tube ${T1} \
  --output-dir JIT/runs/capability_geometry/tube2_vs_tube1_resolution_v1
```

## 13.5 Print compact summaries

```bash
$PY - <<'PY'
import json
from pathlib import Path

for name in [
    "tube0_resolution_v1",
    "tube1_vs_tube0_resolution_v1",
    "tube2_vs_tube1_resolution_v1",
]:
    path = Path("JIT/runs/capability_geometry") / name / "summary.json"
    d = json.load(open(path))
    print("\n===", name, "===")
    print("raw =", d["raw_snapshot_count"])
    print("root_cells =", d["unique_root_geometry_cell_count"])
    print("full_cells =", d["unique_full_physical_cell_count"])
    print("root_dup =", d["root_geometry_duplicate_fraction"])
    print("x_slices =", d["x_slice_count"])
    print("by_phase =", d["by_phase"])
    e = d.get("expansion_vs_source")
    if e:
        print("new_root_cells =", e["new_root_geometry_cell_count"])
        print("new_full_cells =", e["new_full_physical_cell_count"])
        print("new_root_per_raw_growth =", e["new_root_geometry_cells_per_raw_added_snapshot"])
        print("new_cells_by_phase =", e["by_phase"])
        print("nearest_source_distance =", e["new_root_geometry_nearest_source_distance"])
PY
```

---

# 14. Questions the retrospective analysis must answer

## 14.1 Tube_0

- How many unique root cells exist?
- How many unique full cells exist?
- What is the raw-to-cell duplicate fraction?
- What is the x support range?
- How thick are initial upstream/downstream cross-sections?

## 14.2 Tube_1 versus Tube_0

- Of 2,897 raw expansion snapshots, how many create new root cells?
- How many create only new internal/full cells?
- Which x slices widen?
- Does Tube_1 extend in x or mainly thicken existing x slices?

## 14.3 Tube_2 versus Tube_1

- Of 657 raw expansion snapshots, how many create new root cells?
- How many are new full cells only?
- Which phase contributes physical geometry growth?
- What is the nearest old-cell distance distribution?
- Do new cells form a coherent frontier shell or dense local clusters?

## 14.4 Relationship to pi_2 upstream degradation

After Tube geometry is known, investigate:

- whether upstream regressions concentrate in specific x slices;
- whether new Tube_2 cells form a separate branch/lobe;
- whether pi_2 succeeds on a new branch while losing old branch coverage.

This requires a later join between locked gate records and physical cell IDs; do not infer it from current aggregate counts alone.

---

# 15. How to interpret possible outcomes

## Scenario A — heavy raw duplication

If hundreds/thousands of raw added snapshots produce only a small number of new root cells:

```text
problem = acquisition density, not true coverage growth
```

Next priority:

- more physically separated parents;
- unexplored x slices/cells;
- candidate-level cell deduplication before expensive labeling, if prospectively designed.

## Scenario B — substantial new root-cell growth

Then Tube expansion is physically meaningful at the declared resolution.

If policy realization still degrades, the stronger problem becomes:

```text
single-policy representation/retention across a broader behavior family
```

## Scenario C — full cells grow but root cells do not

Then the project gained internal posture/configuration diversity but little macroscopic trajectory-state expansion.

That may help training, but should not be presented as large jumping-envelope growth.

## Scenario D — multiple upstream branches/lobes

Then the earlier intent discussion becomes relevant again.

A future goal/latent-mode conditioned Actor may be justified if geometry shows multiple successful behavior families that a target-free single policy struggles to represent simultaneously.

That decision must be made after geometry evidence, not before.

---

# 16. What the task has achieved so far

1. Built two phase-specific discovery experts without using runtime switching.
2. Converted the expert evidence into a unified-policy training pipeline.
3. Trained pi_0, pi_1 and pi_2 as single unified Actors.
4. Implemented policy-conditioned frontier acquisition and continuation fields.
5. Built structurally core-retaining raw Tube generations.
6. Demonstrated pi_2 success on 13/14 states locked as pi_1 failures, across both phases.
7. Identified severe upstream policy-realization loss despite frontier gain.
8. Separated cumulative capability evidence from latest-policy coverage.
9. Now separates raw replay density from physically independent capability coverage.
10. Adds a direct x-sliced geometric representation so the empirical Tube can actually be inspected as a Tube.

---

# 17. Revised JIT story

A compact scientific description is:

> JIT is a target-free, real-dynamics capability-discovery loop that uses frozen controllers as probes, identifies the current success/failure frontier, converts newly supported regions into just-in-time replay curriculum, and iteratively maps a resolution-aware jumping capability Tube while separately measuring how much of that accumulated Tube a single deployable policy can realize.

Compared with ordinary RL, JIT explicitly asks:

```text
Where in physical state space is success supported?
Where is the current frontier?
Which resolved physical cells are genuinely new?
How does the Tube cross-section change with x?
Does the newest unified policy realize the cumulative support?
```

---

# 18. What JIT does not claim

The project does not currently claim:

- a certified safe Tube;
- a viability kernel;
- formal reachability;
- a proven physical jump limit;
- a two-policy runtime system;
- that raw snapshot count equals physical state-space volume;
- that current pi_2 is formally selected under the revised prospective criterion;
- that C^1 passed all original formal calibration rules;
- that final JCE/JEL has been evaluated.

---

# 19. Automation status

## Already generic

```text
frontier roles
-> continuation fields
-> raw Tube construction
-> smoke/isolation
-> baseline lock
-> candidate train/freeze
-> locked paired evaluation
-> capability-progression analysis
-> prospective selection
```

## Implemented in vNext but not yet integrated into one prospective DAG

```text
physical Tube projection
resolution quantization
root/full sparse-cell accounting
x-slice geometry and plots
resolution-distinct frontier parent revision
```

Before another iteration the explicit sequence is:

```text
analyze source Tube geometry
-> prepare outcome-blind frontier plan
-> revise parents by distinct root geometry cells
-> run frontier roles
```

Do not describe the automatic workflow as fully resolution-aware until these stages are present in a recorded workflow artifact.

---

# 20. Next scientific decision after Tube reconstruction

Do not choose the next method until Tube_0/1/2 physical geometry exists.

Possible decisions after that evidence:

### If coverage growth is too dense/candidate-redundant

Improve acquisition diversity.

### If coverage is broad but one Actor cannot retain it

Investigate policy representation/retention, possibly including a later goal or latent behavior condition.

### If new-cell growth saturates

Consider a stopping criterion based on:

- negligible new root cells;
- no x-range extension;
- stable cross-section bounds;
- repeated frontier candidates landing in occupied cells;
- resource budget.

No final TEST/JCE/JEL is opened until the final method, stopping rule, and final policy are frozen.

---

# 21. Code added/changed in this revision

New production analysis:

```text
JIT/src/jit_dvgc/analysis/capability_tube.py
JIT/cli/analyze_capability_tube.py
```

New resolution-aware acquisition capability:

```text
JIT/src/jit_dvgc/acquisition/resolution_frontier.py
JIT/cli/prepare_resolution_aware_frontier_plan.py
```

New tests:

```text
JIT/tests/test_capability_tube_resolution.py
JIT/tests/test_resolution_frontier_parent_selection.py
```

Updated authority documentation:

```text
AGENTS.md
JIT/AGENTS.md
PROJECT.md
JIT/docs/CURRENT_STATUS.md
JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md
JIT/docs/CODEX_HANDOFF_20260904.md
JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md
JIT/docs/CODE_ORGANIZATION.md
```

---

# 22. Current position in one sentence

> JIT has reached pi_2 and demonstrated real frontier learning, but before any further policy training the project is now converting its raw replay Tubes into a resolution-aware physical capability Tube so future expansion claims correspond to genuinely different state-space regions rather than dense neighboring snapshots.

---

# 23. Authority read order

1. `AGENTS.md`
2. `JIT/AGENTS.md`
3. `JIT/docs/CURRENT_STATUS.md`
4. `JIT/docs/JIT_CAPABILITY_TUBE_VNEXT_OUTLINE_20260904.md`
5. `JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`
6. `JIT/docs/CODEX_HANDOFF_20260904.md`
7. `PROJECT.md`
8. `JIT/docs/ENVELOPE_ITERATION_PROTOCOL.md`
9. `JIT/docs/CODE_ORGANIZATION.md`

The 2026-09-03 handoff is superseded historical context only.
