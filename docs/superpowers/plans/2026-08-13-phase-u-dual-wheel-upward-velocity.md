# Phase U Dual-Wheel Upward-Velocity Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace solver-gap height credit with timing-explicit synchronized wheel upward-velocity credit so Phase U receives clean pre-liftoff action-direction feedback inside the legal jump window.

**Architecture:** `dvgc.two_phase_runtime` remains the pure-JAX authority for collision-wheel terrain clearances. `PhaseExpertEnvAdapter` stores the previous two-wheel clearance vector in its own info namespace, forms a current-minus-previous per-control-tick velocity, and passes the minimum wheel velocity into the bounded Phase U reward. The environment main loop and every physical event/success/failure contract remain unchanged.

**Tech Stack:** Python 3.11, JAX/JAX NumPy, MuJoCo/MJX, pytest, JSON configuration and SHA-256 contracts.

## Global Constraints

- Work only in `/home/qy/DVGC` with `/home/qy/mujoco_playground/.venv/bin/python`.
- Do not modify `dvgc/env.py`, XML, 2 kg payload, +/-50 N m limits, action mapping, observation/history, reset, thresholds, deadlines, PPO/network/optimizer/exploration/horizon, or fixed seeds.
- Keep pre-window task-progress reward exactly zero and early airborne nonterminal/nonsuccess.
- Do not resume the completed v6 run; any dynamic test after qualification must use a fresh run-bound authorization.
- Use explicit path staging and preserve the user-owned untracked `.vscode/` directory.

---

### Task 1: Pure-JAX collision-wheel clearance vector

**Files:**
- Modify: `dvgc/two_phase_runtime.py`
- Test: `tests/test_two_phase_runtime.py`

**Interfaces:**
- Consumes: `state.data.geom_xpos`, `state.data.geom_xmat`, and immutable `TwoPhaseGeometry`.
- Produces: `wheel_terrain_clearances(state: Any, geometry: TwoPhaseGeometry) -> Any`, with final dimension exactly two in the stable collision-geometry manifest order.

- [ ] **Step 1: Write the failing real-runtime tests**

Add tests that call the wished-for `wheel_terrain_clearances` API on synthetic and authoritative geometry. Assert exact two-wheel order, terrain-aware clearances, `minimum_wheel_terrain_clearance == min(vector)`, plus `jax.jit` and batched `jax.vmap` behavior.

- [ ] **Step 2: Run RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_two_phase_runtime.py -k 'wheel_terrain_clearances or minimum_wheel_clearance'
```

Expected: FAIL because `wheel_terrain_clearances` is not exported.

- [ ] **Step 3: Implement the minimal helper**

Factor only the existing physical geometry computation:

```python
def wheel_terrain_clearances(state: Any, geometry: TwoPhaseGeometry) -> Any:
    physical = _physical_geometry_values(state, geometry)
    return physical["terrain_clearances"][..., jp.asarray(geometry.wheel_mask)]
```

If boolean advanced indexing is not JIT-static under the real geometry type,
store the two stable wheel positions as immutable integer indices in
`TwoPhaseGeometry` and gather them. Do not read contact labels or legacy phase
metadata. Make `extract_apex_band_signals` derive its existing scalar minimum
from this helper.

- [ ] **Step 4: Run GREEN and regression**

Run the RED command and then:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_two_phase_runtime.py
```

Expected: PASS.

---

### Task 2: Timing-explicit adapter state and velocity reward

**Files:**
- Modify: `dvgc/phase_expert_training.py`
- Modify: `configs/phase_expert_phase_u.json`
- Modify: `configs/phase_expert_smoke.json`
- Test: `tests/test_phase_expert_training.py`

**Interfaces:**
- Consumes: `wheel_terrain_clearances`, `ctrl_dt`, current event state, and the immediately previous adapter-owned clearance vector.
- Produces: `phase_expert/previous_wheel_terrain_clearances` info state and `minimum_wheel_upward_velocity` input to `phase_u_reward_components`.

- [ ] **Step 1: Write failing schema and reward tests**

Replace v6 height-target expectations with exact fields:

```python
dual_wheel_upward_velocity_deadband = 0.02
dual_wheel_upward_velocity_target = 0.20
```

Assert finite non-negative deadband, finite positive target, and strict
`deadband < target`. Add parameterized reward assertions for pre-window zero,
0.01/0.02 m/s zero, 0.11 m/s half credit, 0.20/0.30 m/s full credit, and
unchanged angular-rate qualification. Verify exact-schema rejection and hash
drift when either field changes.

- [ ] **Step 2: Write failing adapter timing tests**

Extend the fake signal/runtime fixture to provide a two-wheel vector and
control `ctrl_dt`. Assert reset records the exact current vector, the first
step uses current minus reset, the next step uses current minus immediately
previous, only the minimum of the two wheel velocities is rewarded, and the
stored vector advances after every tick. Assert that one wheel rising while
the other is static earns zero, and that credit does not imply liftoff,
success, task failure, physical failure, timeout, or done.

- [ ] **Step 3: Run RED**

Run:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py -k \
  'dual_wheel or previous_wheel or reward_contract_hash or reward_manifest'
```

Expected: FAIL because v6 still consumes absolute height and has no adapter timing state.

- [ ] **Step 4: Implement minimal config and reward behavior**

In `PhaseURewardConfig`, remove `dual_wheel_lift_progress_target`; add the
deadband and target fields with the validations above. Extend
`phase_u_reward_components` with an explicit `minimum_wheel_upward_velocity`
argument and compute:

```python
progress = jp.clip(
    (minimum_wheel_upward_velocity - config.dual_wheel_upward_velocity_deadband)
    / (
        config.dual_wheel_upward_velocity_target
        - config.dual_wheel_upward_velocity_deadband
    ),
    0.0,
    1.0,
)
```

Keep component name `dual_wheel_lift_progress`, weight 4.0, window gate, and
rate quality. Bump `PHASE_U_REWARD_SEMANTICS` to a descriptive v7 value.

- [ ] **Step 5: Implement minimal timing state**

At adapter reset, compute and store the two current clearances. At step, read
the previous vector before calling the base environment, compute the current
vector after that one control tick, divide each difference by immutable
`ctrl_dt`, and pass the minimum into the reward. Update the info value with the
current vector. Test adapters may inject a clearance-vector extractor; do not
write event latch state into `dvgc/env.py`.

- [ ] **Step 6: Update only the stable JSON schemas**

In both stable Phase U configs, replace the old height target with:

```json
"dual_wheel_upward_velocity_deadband": 0.02,
"dual_wheel_upward_velocity_target": 0.20
```

Leave all other JSON values byte-equivalent after canonical parsing.

- [ ] **Step 7: Run GREEN and Phase U regressions**

Run the RED command and then:

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_phase_expert_training.py \
  tests/test_two_phase_runtime.py \
  tests/test_two_phase_semantics.py \
  tests/test_training_budget.py
```

Expected: PASS.

---

### Task 3: Full qualification and evidence update

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Modify only if existing contract assertions require it: `PROJECT.md`, `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`

**Interfaces:**
- Consumes: completed v6 audit, new reward hash, source hash, and validation reports.
- Produces: recoverable v7 experiment marker and exact next permitted run.

- [ ] **Step 1: Run static and full validation**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
```

Expected: all pass with no XML/environment fingerprint mismatch.

- [ ] **Step 2: Run the managed runtime gate**

Use the stable `cli.runtime_gate` interface with a new v7 output JSON under
`runs/two_phase/runtime_gate/`. It must execute exactly 64 update + 32 resume
transitions and pass snapshot/restore/determinism/reward fingerprint checks.

- [ ] **Step 3: Update experiment state**

Record v6 completed accounting, six-panel outcome table, 157/157 sidecars,
48 video/NPZ pairs, solver-gap evidence, selected v7 hypothesis, new contract
hash, test counts, preflight result, runtime-gate path, and zero new formal
training transitions.

- [ ] **Step 4: Commit and push qualified source/docs**

Explicitly stage only the implementation, tests, stable configs, and approved
docs. Do not stage `.vscode/` or `runs/`.

---

### Task 4: Fresh smoke and independently authorized formal retry

**Files:**
- Create ignored run records only under `runs/two_phase/`.
- Modify after audited smoke/formal evidence: `docs/EXPERIMENT_STATE.md`.

**Interfaces:**
- Consumes: current committed HEAD, source/model/config/threshold/reward hashes, and v7 runtime-gate PASS.
- Produces: one smoke result and, only if clean, one fresh v7 formal Phase U run capped at 998,400 PPO-training transitions.

- [ ] **Step 1: Build a fresh threshold/provenance manifest if the code hash invalidates the prior static manifest**

Reuse byte-identical approved threshold values. Do not rerun reference
open-loop dynamics or change any threshold.

- [ ] **Step 2: Create and validate one smoke authorization**

Bind purpose, exact HEAD/source/model/config/threshold/reward identities,
256 environments, 6,400 training transitions, fixed evaluation ceiling,
stopping conditions, and output path. Run preflight first and verify it reports
zero executed transitions.

- [ ] **Step 3: Run one smoke**

Verify finite reset/reward/update/checkpoint/resume/fixed evaluation, recursive
sidecar identity, closed outcome accounting, exact transition counts, and
failure-video hashes. Smoke is engineering evidence only.

- [ ] **Step 4: Create a new formal run input and authorization if smoke is clean**

Keep every v6 formal input byte-equivalent after canonical parsing except the
two v7 reward fields and derived hashes. Bind a new seed/run ID, 256 parallel
environments, effective 998,400 training ceiling, and checkpoints
0/102,400/256,000/505,600/755,200/998,400.

- [ ] **Step 5: Launch once and supervise sparsely**

Use a persistent resumable process and completion marker. Audit startup once,
then inspect only a fixed checkpoint, abnormal exit, or terminal state. Do not
poll full logs.

- [ ] **Step 6: At terminal state, audit before any next change**

Validate every sidecar and media hash; report Apex/liftoff/stable-airborne/
clearance/physical-failure/roll/pitch/return/reward decomposition and parent
diversity at every fixed checkpoint. If at least eight independent successful
Apex parents exist, proceed to candidate snapshot acquisition and bounded
continuation probing. Otherwise retain all failure videos and return to one
new evidence-backed hypothesis; do not stack reward/PPO changes.

