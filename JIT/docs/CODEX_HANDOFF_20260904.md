# DVGC/JIT Technical Handoff — 2026-09-04

## Purpose

This is the active takeover guide after completion of the engineering
`pi_1 -> C^1 -> Tube_2 -> pi_2` round and the capability-progression method
revision.

Full scientific history and quantitative interpretation:

`JIT/docs/JIT_CAPABILITY_PROGRESS_REPORT_20260904.md`

Concise live state:

`JIT/docs/CURRENT_STATUS.md`

The 2026-09-03 handoff is superseded historical context.

---

## 1. Project definition

JIT is an iterative real-dynamics capability-discovery and just-in-time
curriculum framework for a fixed single-track two-wheeled robot task.

Frozen experts and unified policies act as capability probes. Successful
continuation/frontier evidence accumulates into empirical capability/training
support. The frontier generates the next curriculum. One unified policy is
trained for runtime use. Evaluation then separates:

1. empirical frontier progression; and
2. phase-aware single-policy realization coverage.

JIT does not prove the true physical feasibility set or a safe/viable set.

---

## 2. Immutable task identity

- repository: `QaQaaa-zzz/DVGC`
- branch: `agent/two-phase-soft-tube`
- local repo: `~/DVGC`
- Python: `/home/qy/mujoco_playground/.venv/bin/python`
- XML: `assets/orange_bike_4kg_horizontal.xml`
- XML SHA:
  `0b56d3672773ef05a2b5982117fa53a7fdffcaf2b7f3f04a7a7941233d6e9c8a`
- payload: 2 kg
- control: 50 Hz
- hip/knee torque: +/-50 Nm
- action order: `[steer, rear-wheel drive, hip, knee]`
- runtime target: one unified Actor, no expert switching
- final TEST/JCE/JEL: untouched

Do not alter physics/reward/action/snapshot/task geometry/TEST isolation during an
iteration without opening a new method question.

---

## 3. Current completed artifacts

### Experts

`JIT/runs/frozen_experts/pi_up9977856_pi_down25600_20260827/frozen_experts.json`

- `pi_up_star`: 9,977,856 transitions;
- `pi_down_star`: 25,600 transitions.

### Tube_0

`JIT/runs/soft_tube/soft_tube_train_v1_20260828`

```text
222 = 117 upstream + 105 downstream
```

### pi_0

`JIT/runs/frozen_unified/pi_0_round1_10009600_20260831/frozen_unified_policy.json`

### Tube_1

`JIT/runs/soft_tube/soft_tube_iter1_pi0_conditioned_20260901`

```text
3,119 total
= 222 retained Tube_0
+ 2,897 expansion

upstream   427
downstream 2,692
```

### selected pi_1

`JIT/runs/frozen_unified/pi_1_core_replay75_10009600_20260903/frozen_unified_policy.json`

Engineering authority only. Historical formal PASS remains unclaimed because the
old gate retains 3 baseline-reproduction mismatches from the old PRNG protocol.

### Iteration-1 -> 2 work root

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3`

### C^1 engineering selection

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3/continuation_C1_standard_mlp64x64_engineering_selected_v1`

Upstream 64x64:

- AUC `0.6903137789904502`;
- recall `0.5934515688949522`;
- original AUC >= 0.70 false;
- engineering override true.

Downstream 64x64:

- AUC 1.0;
- recall 1.0;
- formal calibration PASS.

### Tube_2

`JIT/runs/soft_tube/soft_tube_iter2_pi1_c1_64x64_engineering_20260904`

```text
3,776 total
= 3,119 retained Tube_1
+ 657 new expansion

upstream   902 = 427 + 475
downstream 2874 = 2692 + 182
```

Manifest SHA:

`135798c843a7acd9eb18cb44f9fd7a92ab39bf3df2d887b6c1fb8c629d480cff`

Tube_2 smoke: GO.

### pi_2

Training run:

`JIT/runs/pi_unified/pi_2_tube2_c1_64x64_engineering_core75_natural10_10009600_seed821101_20260904`

Frozen policy path used for the completed gate:

`JIT/runs/frozen_unified/pi_2_c1_64x64_engineering_10009600_20260904/frozen_unified_policy.json`

Training completed at 10,009,600 transitions.

### pi_1 -> pi_2 locked comparison

`JIT/runs/iteration_auto/pi_1_to_pi_2_20260903_phase_specific_twoaxis_v3/pi_1_to_pi_2_gate_c1_64x64_engineering/summary.json`

---

## 4. Current pi_2 evidence

Source Tube_1 panel:

```text
pi_1 baseline success = 3115/3119
pi_2 success          = 3002/3119
strict regressions    = 115
strict improvements   = 2
```

Phase split:

```text
upstream:
  pi_1 423/427
  pi_2 312/427
  regressions 113

downstream:
  pi_1 2692/2692
  pi_2 2690/2692
  regressions 2
```

Locked frontier:

```text
14 pi_1-negative challenge states
pi_2 success 13/14
3 successful parent groups
upstream 4/5
downstream 9/9
baseline reproduction failures 0
```

Interpretation:

```text
frontier progression: YES
single-policy upstream realization: DEGRADED
```

Do not summarize this as simply “pi_2 failed.”

---

## 5. New decision semantics

Stable analysis:

`JIT/src/jit_dvgc/analysis/capability_progression.py`

CLI:

`JIT/cli/analyze_capability_progression.py`

Future prospective selection contract:

### Frontier progression

- no baseline reproduction mismatch;
- candidate boundary success > 0;
- minimum independent parent groups;
- candidate success in both phases.

### Policy realization

- global locked Tube-panel coverage drop <= 5 percentage points;
- every phase coverage drop <= 10 percentage points.

Strict zero-regression remains diagnostic only.

A candidate becomes next-authority eligible only when frontier + policy
realization both pass prospectively.

Retrospective analyses cannot formally select a policy.

---

## 6. First commands after takeover

```bash
cd ~/DVGC
git pull --ff-only origin agent/two-phase-soft-tube

export PYTHONPATH="$PWD/JIT/src"
PY=/home/qy/mujoco_playground/.venv/bin/python

$PY -m compileall -q JIT/src JIT/cli
$PY -m pytest -q \
  JIT/tests/test_capability_progression.py \
  JIT/tests/test_iterative_envelope_automation.py \
  JIT/tests/test_role_isolation_engineering_override.py
```

Then create the retrospective current-pi2 decision:

```bash
TAG=20260903_phase_specific_twoaxis_v3
ROOT=JIT/runs/iteration_auto/pi_1_to_pi_2_${TAG}
GATE=${ROOT}/pi_1_to_pi_2_gate_c1_64x64_engineering

$PY JIT/cli/analyze_capability_progression.py \
  --gate-summary ${GATE}/summary.json \
  --output ${ROOT}/pi_1_to_pi_2_capability_progression_retrospective.json \
  --retrospective
```

Expected classification:

```text
empirical_envelope_expansion_observed = true
candidate_policy_authority_eligible = false
retrospective_analysis = true
```

Do not pass this retrospective artifact to policy selection.

---

## 7. Automatic workflow status

Future `prepare_iterative_envelope_workflow.py` generates:

```text
frontier plan
-> TRAIN
-> CALIBRATION
-> ACCEPTANCE
-> C^k
-> Tube_(k+1)
-> smoke
-> strict role isolation
-> baseline lock
-> candidate train/freeze
-> locked paired evaluation
-> capability-progression analysis
-> prospective selection only if frontier + realization pass
```

The completed pi_1 -> pi_2 round was not fully automatic because it required:

- phase-specific v3 frontier redesign;
- v3b upstream calibration repair;
- 64x64 same-data architecture engineering selection;
- explicit C_up^1 AUC override;
- engineering near-observation isolation continuation.

Do not claim hands-off automation for this historical round.

---

## 8. What not to do next

Do not:

- automatically run core90/expansion10 merely because strict regressions = 115;
- reopen old A/B warm-start studies;
- rewrite C_up^1 as AUC PASS;
- select current pi_2 retrospectively under the new gate;
- start pi_3 before a new method version is declared;
- touch final TEST/JCE/JEL;
- interpret Tube cardinality as physical envelope volume.

---

## 9. Recommended next scientific decision

The current policy is reward-guided but not told the desired jump behavior. This
is now a stronger candidate bottleneck than replay quantity alone.

Priority options:

1. **goal-/intent-conditioned unified Actor**
   - keep one runtime policy;
   - add a low-dimensional requested jump outcome/behavior variable;
   - candidates include desired horizontal travel, apex/clearance, landing region,
     recovery state, or a learned/normalized behavior code.

2. **multi-seed policy realization evaluation**
   - predeclare multiple seeds per locked state before candidate training;
   - estimate success rate/confidence rather than one rollout response.

3. **discovery-time frozen policy archive**
   - preserve successful frozen probes;
   - use them for capability discovery/frontier access only;
   - never turn this into runtime policy switching.

A new candidate should be launched only after the next method choice is
predeclared.

---

## 10. Claim boundary

Supported:

- empirical Tube support grew from 222 to 3,776 entries;
- pi_2 demonstrates strong locked local frontier progression;
- current upstream single-policy realization degrades substantially;
- the locked-baseline PRNG protocol works for the current gate;
- cumulative capability evidence and latest-policy coverage are different
  quantities.

Not supported:

- Tube_2 equals the physical jump limit;
- pi_2 is formally selected under the new prospective criterion;
- current C^1 passed all original calibration gates;
- the current round was fully automatic;
- final JCE/JEL has been evaluated;
- latest-policy failure proves physical infeasibility.
