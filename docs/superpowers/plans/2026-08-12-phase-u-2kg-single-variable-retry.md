# Phase U 2 kg Single-Variable Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Change the single authoritative DVGC payload from 4.0 kg to 2.0 kg, close its provenance/runtime contracts, qualify the new dynamics, and run one fresh sparsely supervised Phase U retry capped at 1,000,000 training transitions.

**Architecture:** Retain the configured historical XML path and edit only its named `load` geom mass. Bind the resulting byte hash through the existing configuration, Gate B manifests, runtime gate, and phase-expert run authorization; old 4 kg artifacts remain immutable and incompatible. Reuse the stable 512-env Phase U training entrypoint with unchanged reward/reset/PPO settings and supervise only fixed checkpoint or terminal states.

**Tech Stack:** MuJoCo XML, Python 3.12, JAX/MJX Warp, Brax PPO, JSON manifests, pytest, shell preflight/runtime gates.

## Global Constraints

- Work only in `/home/qy/DVGC` on `agent/two-phase-soft-tube`; preserve `.vscode/`.
- The only physical edit is `geom name="load" mass="4.0" -> mass="2.0"` in `assets/orange_bike_4kg_horizontal.xml`.
- The post-edit XML SHA-256 must be `e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192`.
- Keep the historical XML filename and configured path; do not create a second model.
- Preserve geometry, meshes, obstacle, initial state, +/-50 N m hip/knee limits, action mapping, reward, reset, observation, thresholds, optimizer, network, horizon, and 512-env PPO layout.
- Never rewrite historical run/report provenance. Old 4 kg checkpoints and banks are incompatible inputs, not resume sources.
- Early airborne remains nonterminal, unpunished telemetry and never grants Phase U/Apex success.
- Formal training starts only after targeted/full/preflight, fresh Gate B, runtime gate, and a clean 12,800-transition smoke.
- The formal training ceiling is 998,400 aligned transitions with effective fixed checkpoints `0, 102400, 256000, 512000, 755200, 998400`.
- Stop or pause on existing numerical, collision, hash, timing/history, accounting, reward-hacking, action-saturation, repeated-degradation, and three-window plateau conditions.
- Do not start Phase D, formal feasibility training, Tube construction, or unified PPO in this plan.

---

### Task 1: Authoritative 2 kg Model Contract

**Files:**
- Modify: `tests/test_model.py`
- Modify: `assets/orange_bike_4kg_horizontal.xml`
- Modify: `dvgc/config.py`
- Modify: `cli/runtime_gate.py`

**Interfaces:**
- Consumes: `inspect_model(path) -> dict[str, Any]` and the stable historical XML path.
- Produces: `AUTHORITATIVE_PAYLOAD_MASS_KG = 2.0`, `AUTHORITATIVE_XML_SHA256 = "e276...9192"`, and runtime validation bound to both.

- [ ] **Step 1: Write the failing model behavior test**

First change `tests/test_model.py` so the parsed real XML must satisfy:

```python
assert model["named_masses_kg"]["load"] == 2.0
assert model["xml_sha256"] == (
    "e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192"
)
```

Retain the existing obstacle, mesh, knee range, actuator order, and exact
hip/knee force-range assertions. Production mutation caught: a payload other
than 2 kg, a stale hash, or collateral force/geometry change.

- [ ] **Step 2: Run RED**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_model.py
```

Expected: FAIL on the literal mass assertion because the real XML still parses
`load=4.0`. This is a behavioral failure, not an import or collection error.

- [ ] **Step 3: Apply the minimal authoritative model change**

In the one named XML geom, set `mass="2.0"`. In `dvgc/config.py`, add:

```python
AUTHORITATIVE_PAYLOAD_MASS_KG = 2.0
AUTHORITATIVE_XML_SHA256 = (
    "e2762bec49fdce61eff6ad01b6a67925934d8997b53929b0a67ace7f44109192"
)
```

Then import both authoritative constants in `tests/test_model.py` and add:

```python
assert AUTHORITATIVE_PAYLOAD_MASS_KG == 2.0
assert model["named_masses_kg"]["load"] == AUTHORITATIVE_PAYLOAD_MASS_KG
assert model["xml_sha256"] == AUTHORITATIVE_XML_SHA256
```

In `cli/runtime_gate.py`, import the mass constant and replace the 4 kg literal
check/message with comparison to `AUTHORITATIVE_PAYLOAD_MASS_KG` and a message
that reports the expected 2 kg contract. Do not change any other XML attribute.

- [ ] **Step 4: Run GREEN and mutation checks**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q tests/test_model.py
sha256sum assets/orange_bike_4kg_horizontal.xml
git diff -- assets/orange_bike_4kg_horizontal.xml
```

Expected: test PASS; hash equals `e276...9192`; XML diff contains exactly the
single mass value change.

- [ ] **Step 5: Commit the physical contract atomically**

```bash
git add -- tests/test_model.py assets/orange_bike_4kg_horizontal.xml dvgc/config.py cli/runtime_gate.py
git commit -m "model: set authoritative payload to 2kg"
```

---

### Task 2: Live Documentation and Current Report Provenance

**Files:**
- Modify: `AGENTS.md`
- Modify: `PROJECT.md`
- Modify: `docs/METHOD_TWO_PHASE_SOFT_TUBE.md`
- Modify: `docs/VERIFICATION_PROTOCOL.md`
- Modify: `docs/XML_AND_KNEE_MAPPING.md`
- Modify: `docs/model_report.json` via `cli.prepare_project`
- Modify: `docs/BUILD_VALIDATION.json`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: the parsed 2 kg model, new XML hash, and the immutable old-run boundary.
- Produces: current reader-facing contracts that identify 2 kg while preserving the historical filename and old evidence.

- [ ] **Step 1: Regenerate the structural model report**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.prepare_project \
  --xml assets/orange_bike_4kg_horizontal.xml \
  --reference data/reference_jump.csv \
  --docs docs
```

Verify `docs/model_report.json` reports `load: 2.0`, hash `e276...9192`, all
meshes present, unchanged obstacle geometry, and unchanged force ranges.

- [ ] **Step 2: Update only live method/governance claims**

Change current authoritative statements from 4 kg to 2 kg and explicitly state
that `orange_bike_4kg_horizontal.xml` is a retained historical filename. In
`docs/EXPERIMENT_STATE.md`, append the new experiment authorization without
altering the prior 4 kg run narrative. In `docs/BUILD_VALIDATION.json`, update
the authoritative XML hash and payload to 2.0 but leave its old validation
counts/timings marked stale until Tasks 3--4 refresh them; do not claim a fresh
PASS prematurely.

Do not edit historical files under `docs/experiments/`, old run directories, or
old hashes quoted as past evidence.

- [ ] **Step 3: Validate live/current versus historical occurrences**

```bash
rg -n "4 kg|4\.0 kg|payload_mass_kg|d7e9f43f" \
  AGENTS.md PROJECT.md docs dvgc cli tests \
  --glob '!docs/experiments/**' \
  --glob '!docs/superpowers/specs/2026-08-12-phase-u-2kg-single-variable-retry-design.md' \
  --glob '!docs/EXPERIMENT_STATE.md'
```

Classify every remaining hit: allowed historical evidence/path name or a stale
live contract to fix. Do not globally replace the old hash.

- [ ] **Step 4: Run affected static tests**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_model.py \
  tests/test_repository_contract.py \
  tests/test_source_contracts.py \
  tests/test_reference_joints.py \
  tests/test_reset_geometry.py \
  tests/test_two_phase_runtime.py
```

Expected: PASS with no model-path split or geometry/action regression.

- [ ] **Step 5: Commit current provenance contracts**

```bash
git add -- AGENTS.md PROJECT.md docs/METHOD_TWO_PHASE_SOFT_TUBE.md \
  docs/VERIFICATION_PROTOCOL.md docs/XML_AND_KNEE_MAPPING.md \
  docs/model_report.json docs/BUILD_VALIDATION.json docs/EXPERIMENT_STATE.md
git commit -m "docs: bind current method to 2kg model"
```

---

### Task 3: Static, Full, and Gate B Refresh

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored evidence: `runs/two_phase/gate_b_2kg_20260812/`

**Interfaces:**
- Consumes: the new authoritative XML/hash and unchanged guideline/runtime contracts.
- Produces: fresh 2 kg threshold, geometry, natural reset, snapshot/round-trip, and guideline-role evidence.

- [ ] **Step 1: Run static compilation and directly affected tests**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m compileall dvgc cli
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q \
  tests/test_model.py tests/test_two_phase_runtime.py \
  tests/test_two_phase_guideline.py tests/test_two_phase_snapshot_roundtrip.py \
  tests/test_phase_expert_training.py tests/test_prelaunch_continuation.py
```

- [ ] **Step 2: Run the fresh Gate B builder exactly once**

```bash
/home/qy/mujoco_playground/.venv/bin/python \
  -m cli.build_two_phase_guideline_banks \
  --config configs/default.json \
  --reference data/reference_jump.csv \
  --output runs/two_phase/gate_b_2kg_20260812 \
  --seed 4200 \
  --perturbations nominal \
  --geometry-tolerance 2e-4 \
  --event-max-control-ticks 100
```

Do not change seed, actions, thresholds, or window after observing the result.
The revised Gate B does not require guideline open-loop Apex/recovery success;
inspect the threshold, geometry, natural-reset, static snapshot, and round-trip
artifacts that exist under the revised contract. Any reference replay failure
is provenance, not a reason to tune reference actions.

- [ ] **Step 3: Audit Gate B artifacts**

Assert all current manifests use hash `e276...9192`, JAX/host geometry sign and
tolerance checks pass, natural Phase U reset is legal, every timing-explicit
snapshot has real ordered history/last action, and no old two-phase bank is
represented as authoritative. Record actual status and any diagnostic videos.

- [ ] **Step 4: Run full tests and preflight**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m pytest -q
bash scripts/local_preflight.sh
```

Expected: PASS. `local_preflight.sh` regenerates `docs/model_report.json`; verify
its content remains the same 2 kg model identity.

- [ ] **Step 5: Record and commit the static/Gate B result**

Update `docs/EXPERIMENT_STATE.md` with exact commands, test counts, Gate B
status, environment transitions, manifest hashes, and zero PPO training for
this new experiment so far.

```bash
git add -- docs/EXPERIMENT_STATE.md docs/model_report.json
git commit -m "docs: record 2kg gate b validation"
```

---

### Task 4: Dynamic Runtime Qualification

**Files:**
- Modify: `docs/RUNTIME_GATE.json` through `cli.runtime_gate`
- Modify: `docs/BUILD_VALIDATION.json`
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored evidence: `runs/two_phase/runtime_gate/phase_u_2kg_20260812/`

**Interfaces:**
- Consumes: new XML/hash and the actual PPO/reset/step source fingerprint.
- Produces: one fresh PASS report covering load/reset/step, 64+32 PPO update/resume, snapshot/policy round trips, and deterministic inference.

- [ ] **Step 1: Run the complete runtime gate**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate \
  --config configs/default.json \
  --output docs/RUNTIME_GATE.json \
  --work-dir runs/two_phase/runtime_gate/phase_u_2kg_20260812
```

Expected: PASS, parsed payload 2 kg, current XML hash, current source/config
fingerprints, finite zero/random rollouts, short PPO 64 transitions, resume 32
transitions, and passing snapshot/policy/determinism gates.

- [ ] **Step 2: Verify report freshness**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.runtime_gate \
  --config configs/default.json \
  --output docs/RUNTIME_GATE.json \
  --check-only
```

- [ ] **Step 3: Close build validation and record evidence**

Update `docs/BUILD_VALIDATION.json` from actual Task 3--4 results: date, test
counts, runtime elapsed time, physical failure/timeout causes, round-trip
tolerances, 64+32 transitions, new hash, and payload 2.0. Update experiment
state with the same evidence and label 96 transitions as runtime smoke only.

- [ ] **Step 4: Commit runtime evidence and push**

```bash
git add -- docs/RUNTIME_GATE.json docs/BUILD_VALIDATION.json docs/EXPERIMENT_STATE.md
git commit -m "test: qualify 2kg runtime"
git push origin agent/two-phase-soft-tube
```

---

### Task 5: Run-Bound 512-Environment Smoke

**Files:**
- Create ignored: `runs/two_phase/configs/phase_u_2kg_env512_smoke_20260812.json`
- Create ignored: `runs/two_phase/authorizations/gate_c1_phase_u_2kg_env512_smoke_20260812_seed720001.json`
- Create ignored: `runs/two_phase/phase_experts/gate_c1_phase_u_2kg_env512_smoke_20260812_seed720001/`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: unchanged stable Phase U config, fresh threshold manifest, new source/XML hashes.
- Produces: one completed 12,800-transition engineering smoke with checkpoint, evaluation, videos, and closed accounting.

- [ ] **Step 1: Create and preflight the run-specific smoke config**

Copy the previously validated 512-env smoke layout while binding the fresh 2 kg
threshold manifest. Keep `policy_initial_action_std = [0.05, 0.05, 0.25, 0.05]`,
one 12,800-transition block, fixed held-out seeds, and unchanged reward/PPO
fields. Set interaction ceilings for 12,800 training, 1,600 Brax evaluation,
and 1,600 fixed evaluation.

Run the stable CLI with `--preflight-only` and assert requested/effective
training transitions are both 12,800 and all hash/ceiling fields close.

- [ ] **Step 2: Issue one run-bound smoke authorization**

The authorization must include run ID, seed `720001`, producer HEAD and source
tree hash, new XML hash, threshold canonical hash, training-config hash,
absolute output directory, exact interaction ceilings, purpose, stopping
condition, `promotion_authorized=false`, and exclusions for formal Tube/Phase D/
unified PPO.

- [ ] **Step 3: Execute the smoke once**

```bash
/home/qy/mujoco_playground/.venv/bin/python -m cli.train_phase_expert \
  --phase propulsion_ascent \
  --experiment-level smoke \
  --requested-total-transitions 12800 \
  --seed 720001 \
  --config configs/default.json \
  --training-config runs/two_phase/configs/phase_u_2kg_env512_smoke_20260812.json \
  --threshold-manifest runs/two_phase/gate_b_2kg_20260812/threshold_manifest.json \
  --authorization-manifest runs/two_phase/authorizations/gate_c1_phase_u_2kg_env512_smoke_20260812_seed720001.json \
  --run runs/two_phase/phase_experts/gate_c1_phase_u_2kg_env512_smoke_20260812_seed720001
```

- [ ] **Step 4: Audit engineering integrity**

Require `status=completed`, training=12,800, finite rewards/gradients/std,
complete checkpoint, fixed evaluation with closed outcome accounting, videos
and NPZ traces for all failures, and no broadphase overflow, NaN/Inf, OOM,
traceback, timing/history mismatch, hash mismatch, or collision truncation.
Report learning metrics descriptively but do not require Apex success in one
block.

- [ ] **Step 5: Commit the smoke marker**

Update and commit only `docs/EXPERIMENT_STATE.md`; run outputs stay ignored.
Push the branch. If smoke integrity fails, stop before Task 6 and diagnose the
failure without changing multiple variables.

---

### Task 6: Fresh 998,400-Transition Formal Retry

**Files:**
- Create ignored: `runs/two_phase/configs/phase_u_2kg_env512_formal_20260812.json`
- Create ignored: `runs/two_phase/authorizations/phase_u_2kg_env512_998400_20260812_seed720002.json`
- Create ignored: `runs/two_phase/phase_experts/phase_u_2kg_env512_998400_20260812_seed720002/`
- Create ignored: `runs/two_phase/process_logs/phase_u_2kg_env512_998400_20260812_seed720002.control.txt`
- Modify: `docs/EXPERIMENT_STATE.md`

**Interfaces:**
- Consumes: clean smoke, new source/XML/threshold hashes, fresh policy initialization.
- Produces: a detached, resumable run capped at 998,400 training transitions with fixed physical checkpoint evaluations and evidence-gated acquisition.

- [ ] **Step 1: Create and validate formal config and authorization**

Keep the unchanged 512-env PPO/reward/reset/evaluation fields. Set requested
checkpoint schedule `[0,100000,250000,500000,750000,998400]`, effective schedule
`[0,102400,256000,512000,755200,998400]`, training ceiling 998,400, fixed-eval
ceiling 9,600, candidate ceiling 76,800, continuation ceiling 76,800, combined
training+fixed ceiling 1,008,000, and total environment ceiling 1,161,600.

Issue authorization seed `720002` with `cumulative_training_start=0`; explicitly
forbid `--resume-run` and any 4 kg checkpoint.

- [ ] **Step 2: Launch persistently**

Use `nohup setsid` with the stable CLI, redirect the complete log under
`runs/two_phase/process_logs/`, record PID, hashes, paths, ceilings, checkpoint
schedule, and exact resume template in the control file.

- [ ] **Step 3: Perform one startup health check**

Require a live PID, `status.json` with `running`, run manifest with the expected
2 kg XML/source/config/threshold/auth identities, transition-0 checkpoint, and
a clean log. Then stop active polling.

- [ ] **Step 4: Record and push startup evidence**

Commit only `docs/EXPERIMENT_STATE.md` and push. Record the estimated first
sparse inspection window as 8--15 minutes based on prior 512-env throughput and
checkpoint video overhead.

---

### Task 7: Sparse Terminal Audit and Single-Hypothesis Loop

**Files:**
- Modify: `docs/EXPERIMENT_STATE.md`
- Create ignored audit data under the formal run directory
- Modify only the files required by a later separately designed single hypothesis

**Interfaces:**
- Consumes: `status.json`, `metrics.jsonl`, checkpoint evaluations, MP4/NPZ traces, logs, and manifests.
- Produces: a terminal completion/Gate Pause audit, `pi_up_star`/acquisition decision, or one next falsifiable experiment.

- [ ] **Step 1: Inspect only milestone or terminal state**

At the scheduled continuation, read status once. If still running between fixed
checkpoints, record no new conclusion and wait for the next checkpoint-scale
interval. Do not repeatedly tail full logs.

- [ ] **Step 2: Audit every completed checkpoint**

Report Apex success, window reach, liftoff, ascending/clearance/forward retention,
roll/pitch/contact failure, physical failure/timeout/other outcome accounting,
return decomposition, action saturation, independent successful parent count,
candidate snapshots, continuation probes, and all interaction categories.
Inspect representative failure videos and timing traces.

- [ ] **Step 3: Decide the permitted branch**

- If at least eight independent successful online parents and clean Apex cases
  exist, allow candidate acquisition/continuation diagnostics under the existing
  gates; do not declare a formal Tube.
- If a suitable checkpoint has robust held-out physical coverage, document the
  evidence needed to select `pi_up_star`; formal selection/relabeling remains a
  separately validated step.
- If completed or Gate Paused without coverage, state one evidence-backed,
  falsifiable next hypothesis. Change one factor only, run red-green/full checks
  proportional to the fingerprint, qualify a new smoke, and issue a fresh run-
  bound authorization. Keep XML payload fixed at 2 kg and all safety limits.
- If a contract/numerical/collision/hash/state corruption occurs, diagnose and
  repair that defect before any scientific change.

- [ ] **Step 4: Verify, commit, and push the terminal marker**

Run exact audit assertions for counts/hashes/outcomes and `git diff --check`.
Commit only validated source/config/docs paths; never commit `runs/`, policies,
videos, logs, or `.vscode/`. Push the branch and keep the long-lived goal active
until Phase U has sufficient Apex coverage for the Soft-Tube data path or a
true user-decision blocker recurs under the blocked-audit rule.
